# MISCELLANEA

Various bits of code that seem useful, but not enough to deserve their own repo.

- bash
  - config.sh : parse files with "key=value" lines
  - logging : a reminder of how to use `tee` to save program output to a file and print to the shell
- cpp
  - bessel.hpp : approximation of J1(x)
  - copycount.hpp : boilerplate for reference counting
  - iniparser : parse ini files "\[Section]" and "key=value"
  - logger : thread-safe logging
  - options : half-baked options parsing
  - tsqueue : thread-safe queue
- python (these files are in the root to make importing easier...)
  - ansi.py : ansi formatting codes for colors etc
  - image_utils.py : various image processing functions, some accelerated using numba
    - loading/saving png
    - projections: lambert azimuthal equal area, cylindrical equal area, arbitrary warp with QMC subsampling
    - filtering: sepfir2d, sepfirnd, sobel_filters, farid_filters, circle_mask
  - mpl_utils.py : various matplotlib helpers
    - colorline : make a line with individually colored segments
    - symlog, isymlog : symlog & its inverse transform
    - lots of colormaps
    - mxticks, myticks : more options for xticks and yticks, particularly applying format strings to existing tick locations
  - progress.py : printing progress bars & task status
  - script_utils.py : find_files, emailer
  - sobol.py : Sobol sequence
  - stats_utils.py : numba accelerated statistics, angular stats
