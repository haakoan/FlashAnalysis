import numpy as np
import h5py
import warnings
from scipy.interpolate import griddata

class FlashFile:
    """
    Class to read in a hdf5 Flash4 checkpoint file and do interpolation.
    """
    def __init__(self, path: str):
        """
        Initialize the FlashFile read.

        Parameters
        ----------
        path : str
            Path to a FLASH HDF5 output file.
        """
        self.path = path
        self.datafile = None
        self.read_file()
        self.set_grid()


    def read_file(self):
        """
        Open the HDF5 file in read-only mode and store the handle.

        Sets
        ----
        self.datafile : h5py.File
            Open file handle to the FLASH HDF5 data.

        Raises
        ------
        OSError
            If the file cannot be opened.
        """
        try:
            self.datafile = h5py.File(self.path, "r")
        except Exception as e:
            # keep it simple but fail loudly
            raise OSError(f"Could not open HDF5 file {self.path}: {e}")


    def __str__(self):
        """Return a one-line summary of the file."""
        return (f"FlashFile('{self.path}', "
                f"dim={self.mdim}D, blocks={self.nblocks}, "
                f"blocksize=({self.nxb},{self.nyb},{self.nzb}))")

    def print_info(self):
        """Print details about the file and grid setup."""
        print(f"FLASH file: {self.path}")
        print(f"  Dimensionality : {self.mdim}D")
        print(f"  Blocks         : {self.nblocks}")
        print(f"  Block size     : ({self.nxb}, {self.nyb}, {self.nzb})")
        print(f"  Top blocks     : {len(self.topblock)}")
        print(f"  Faces/Children : {self.nfaces} faces, {self.nchild} children")

    def set_grid(self):
        """
        Initialize grid metadata from the FLASH HDF5 file.

        Populates attributes used for interpolation:
        - coords   : (nblocks, 3) block center coordinates
        - bbox     : (nblocks, 6) bounding boxes [xmin, xmax, ymin, ymax, zmin, zmax]
        - refl     : (nblocks,) refinement level
        - gid      : (nblocks, ?) grid connectivity IDs
        - nodetype : (nblocks,) node type flags
        - nblocks  : total number of blocks
        - blockid  : block indices [1..nblocks]
        - topblock : IDs of top-level blocks (refinement level = 1)
        - mdim, nxb, nyb, nzb, nfaces, nchild, leaf_type : integer grid info

        Notes
        -----
        - Requires self.datafile to be an open h5py.File handle.
        """

        if not isinstance(self.datafile, h5py.File):
            raise TypeError("`datafile` must be an open h5py.File")

        # Coordinates and bounding boxes
        self.coords = np.array(self.datafile['coordinates'])
        bb = np.array(self.datafile['bounding box'])  # (nblocks, 3, 2)
        if bb.ndim != 3 or bb.shape[1:] != (3, 2):
            raise ValueError(f"Unexpected bounding box shape {bb.shape}, expected (nblocks, 3, 2)")
        self.bbox = np.column_stack([
            bb[:, 0, 0], bb[:, 0, 1],   # xmin, xmax
            bb[:, 1, 0], bb[:, 1, 1],   # ymin, ymax
            bb[:, 2, 0], bb[:, 2, 1],   # zmin, zmax
        ])

        # Refinement and block info
        self.refl = np.array(self.datafile['refine level'])
        self.gid = np.array(self.datafile['gid'])
        self.nodetype = np.array(self.datafile['node type'])
        self.nblocks = self.bbox.shape[0]
        self.blockid = np.arange(self.nblocks) + 1
        self.topblock = self.blockid[self.refl == 1]

        # Integer scalars (grid parameters)
        info = self.datafile['integer scalars'][:]
        result = {
            (n.decode().strip() if isinstance(n, bytes) else str(n).strip()): v
            for n, v in zip(info["name"], info["value"])
        }

        self.mdim = result['dimensionality']
        self.nxb = result['nxb']
        self.nyb = result['nyb']
        self.nzb = result['nzb']
        self.nfaces = 2 * self.mdim
        self.nchild = 2 ** self.mdim
        self.leaf_type = 1


    def _contains(self, bounds6, x, y, z):
        """
        Check whether a point lies inside a bounding box.

        The convention is inclusive lower bounds and exclusive upper bounds,
        this then defines how we deal with a point that lies exactly on a shared face between boxes.

        Parameters
        ----------
        bounds6 : array_like of length 6
            [xmin, xmax, ymin, ymax, zmin, zmax]
        x, y, z : float
            Coordinates of the query point.

        Returns
        -------
        bool
            True if (x, y, z) is inside the box, False otherwise.

        Raises
        ------
        ValueError
            If bounds6 does not have length 6.
        """
        import numpy as np

        bounds6 = np.asarray(bounds6, dtype=float)
        if bounds6.shape[0] != 6:
            raise ValueError(f"`bounds6` must have length 6; got shape {bounds6.shape}")

        xmin, xmax, ymin, ymax, zmin, zmax = bounds6
        return ((x >= xmin) and (x < np.nextafter(xmax, -np.inf)) and
                (y >= ymin) and (y < np.nextafter(ymax, -np.inf)) and
                (z >= zmin) and (z < np.nextafter(zmax, -np.inf)))

    def descend_tree(self,node_list,x,y,z): 
            """
            This function climbs down the tree to find the 
            leaf block that contains (x,y,z). 
            one might think that it risk looping over all the blocks,
            but the tree is structure in such a way that the first part of the
            search only looks in 8 blocks and from there the number of possible nodes
            is limited.
            """ 

            #The i-1 notation might seem confusing at first.
            #It is done like this because Flash is a Fortran code and 
            #labels block ids from 1 to nblocks. Python indexes will go from
            #0 to nblocks-1, we need the python index.
            for i in node_list:
                if(i <= 0):
                    continue
                if(self._contains(self.bbox[i-1], x, y, z)):
                    node_list_new = self.gid[i-1][self.nfaces+1:]
                    #node_list_new = node_list_new[node_list_new != i-1]
                    if(not self.nodetype[i-1] == self.leaf_type): #if its not a leaf then keep going
                        return self.descend_tree(node_list_new,x,y,z)
                    else: # If leaf, done 
                        return i-1
            warnings.warn(f"Did not find any block containing {x}, {y}, {z}; returning -1.")
            return -1

    def build_interpolator(self,block,x,y,z):
        """
        Find the points needed for interpolation to a given point (x, y, z).

        This function identifies the 8 nearest cell centers that form the 
        interpolation cube enclosing the target point. If one or more of 
        the required corners lie outside the current block, the function 
        locates the correct neighboring block and substitutes the 
        corresponding cell center from there.

        Parameters
        ----------
        block : int
            ID/index of the block containing the point (or nearest candidate).
        x, y, z : float
            Coordinates of the target point in simulation space.

        Returns
        -------
        centers : list of lists
            Each entry has the form:
            [xc, yc, zc, i, j, k, block_id]
            where (xc, yc, zc) are the coordinates of the cell center, 
            (i, j, k) are the integer indices of the cell within the block, 
            and block_id is the ID of the block where the cell belongs.
            The list contains 8 such entries, corresponding to the cube 
            corners around (x, y, z).
        """
        def center(xmin,dx,i): 
            return xmin + (i + 0.5)*dx

        xmin,xmax,ymin,ymax,zmin,zmax = self.bbox[block]
        dx = (xmax-xmin)/float(self.nxb)
        dy = (ymax-ymin)/float(self.nyb)
        dz = (zmax-zmin)/float(self.nzb)

        i_f = (x - (xmin + 0.5*dx)) / dx
        j_f = (y - (ymin + 0.5*dy)) / dy    
        k_f = (z - (zmin + 0.5*dz)) / dz

        i0 = int(np.floor(i_f))
        j0 = int(np.floor(j_f))
        k0 = int(np.floor(k_f))

        corners = [
        (i0,   j0,   k0,block),
        (i0+1, j0,   k0,block),
        (i0,   j0+1, k0,block),
        (i0+1, j0+1, k0,block),
        (i0,   j0,   k0+1,block),
        (i0+1, j0,   k0+1,block),
        (i0,   j0+1, k0+1,block),
        (i0+1, j0+1, k0+1,block),]

        centers = []
        for corner in corners:
            ii = corner[0]; jj = corner[1]; kk = corner[2]
            ib = corner[3]
            xc = center(xmin, dx, ii)
            yc = center(ymin, dy, jj)
            zc = center(zmin, dz, kk)

            if 0 <= ii < self.nxb and 0 <= jj < self.nyb and 0 <= kk < self.nzb:
                centers.append([xc,yc,zc,ii,jj,kk,ib])   # inside same block
            else:
                outside_block = self.descend_tree(self.topblock,xc,yc,zc)
                if outside_block < 1:
                    continue
                xmin_out,xmax_out,ymin_out,ymax_out,zmin_out,zmax_out = self.bbox[outside_block]
                dx_out = (xmax_out-xmin_out)/float(self.nxb)
                dy_out = (ymax_out-ymin_out)/float(self.nyb)
                dz_out = (zmax_out-zmin_out)/float(self.nzb)

                i_f_out = (xc - (xmin_out + 0.5*dx_out)) / dx_out
                j_f_out = (yc - (ymin_out + 0.5*dy_out)) / dy_out    
                k_f_out = (zc - (zmin_out + 0.5*dz_out)) / dz_out

                i0_out = int(np.floor(i_f_out))
                j0_out = int(np.floor(j_f_out))
                k0_out = int(np.floor(k_f_out))
                xc_out = center(xmin_out, dx_out, i0_out)
                yc_out = center(ymin_out, dy_out, j0_out)
                zc_out = center(zmin_out, dz_out, k0_out)
                centers.append([xc_out,yc_out,zc_out,i0_out,j0_out,k0_out,outside_block])
        
        return np.array(centers)


    def interpolate(self, field, centers, x, y, z):
        """
        Interpolate a scalar field at (x, y, z) from scattered cell centers.

        Parameters
        ----------
        field : str
            name of the field in self.datafile. 
        centers : array_like, shape (N, 7) Should come from self.build_interpolator.
                  In most cases N=8, but it could be smaller if some interpolation points
                  lie outside the domain, must be atleast 4.
            Rows contain:
                [:, 0:3] -> float XYZ coordinates of the sample points
                [:, 3]   -> int i-index in block
                [:, 4]   -> int j-index in block
                [:, 5]   -> int k-index in block
                [:, 6]   -> int block id
        x, y, z : float
            Query point.

        Returns
        -------
        float
            Linear-interpolated value at (x, y, z).

        Raises
        ------
        ValueError
            If centers has the wrong shape, if there are fewer than 4 points,
            or if the points are not full-rank in 3D (coplanar/collinear).
        """

        centers = np.asarray(centers)
        if centers.shape[0] < 4:
            raise ValueError(f"centers must have at least 4 points, got {centers.shape}")

        points = centers[:, 0:3].astype(np.float64)
        npoints = points.shape[0]

        # Coplanarity / rank check (3D needs rank 3)
        centered = points - points.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered) < 3:
            raise ValueError("Points are not full-rank in 3D (coplanar/collinear)")

        # Indices
        i0s = centers[:, 3].astype(np.int64)
        j0s = centers[:, 4].astype(np.int64)
        k0s = centers[:, 5].astype(np.int64)
        blocks = centers[:, 6].astype(np.int64)

        # Gather values for each sample point
        values = np.empty(npoints, dtype=np.float64)
        for j, b in enumerate(blocks):
            values[j] = np.array(self.datafile[field][b])[i0s[j], j0s[j], k0s[j]]

        #call to griddata, this is not optimal, but its fine for us
        result = griddata(points, values, (float(x), float(y), float(z)), method="linear")
        return float(result)

    def interpolate_fields(self, fields, points):
        """
        Interpolate multiple fields at multiple query points.

        Parameters
        ----------
        fields : sequence of str
            Field names present in self.datafile (e.g. ["dens", "pres", "velx"]).
        points : array_like, shape (M, 3)
            Query points as [[x1, y1, z1], ..., [xM, yM, zM]].

        Returns
        -------
        dict[str, np.ndarray]
            Mapping field name -> array of length M with interpolated values.

        Notes
        -----
        - Warns if any query point lies outside the domain.
        - Skips out-of-domain corners inside build_interpolator; requires ≥4 corners.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (M, 3); got {points.shape}")

        fields = list(fields)
        M = points.shape[0]
        out = {f: np.empty(M, dtype=float) for f in fields}

        for m in range(M):
            x, y, z = points[m]
            # Hard fail if point is outside the domain
            block = self.descend_tree(self.topblock,x, y, z)
            if block < 0:
                warnings.warn(f"Point ({x}, {y}, {z}) is outside the domain; filling with NaN.")
                for f in fields:
                    out[f][m] = np.nan
                continue

            # Build stencil (may skip out-of-domain corners, must leave ≥4)
            centers = self.build_interpolator(block, x, y, z)
            # Interpolate each requested field at this point
            for f in fields:
                out[f][m] = self.interpolate(f, centers, x, y, z)

        return out
    
    def interpolate_field(self, field, points):
        """
        A small helper for when we want a single field.
        Takes in a field, str, and the list of points.
        """
        return self.interpolate_fields([field], points)[field]
