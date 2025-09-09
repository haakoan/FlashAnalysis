# Fast FLASH Interpolator

Works by:

- Searching tree structure to find the block containing a given point, this is fast.
- Collecting the surrounding cell values needed for interpolation. They are few in number.
- Performing interpolation using scipy.griddata.
- Consider it a beta version, it requires testing.

The main source code can be found in flash_octree_interpolator.py, the notebook
contains a short demonstration.
