#image_utils.py

#This is free and unencumbered software released into the public domain.

import numpy as np
from .sobol import sobol
#image derivative filters consist of 2 separable filters
# the interpolation filter and the derivative filter
# the interpolation filter is applied to the orthogonal axis to the derivative filter
def sobel_filters():
    """Sobel differentiation filters
    returns p, d : p is the interpolation filter, d is the derivative filter
    e.g:
    p,d = sobel_filters()
    darr_dcol = scipy.signal.sepfir2d(arr, p, d)
    darr_drow = scipy.signal.sepfir2d(arr, d, p)
    """
    return [1,2,1],[1,0,-1]

def farid_filters(n=3):
    """Farid's differentiation filters
    From Differentiation of Discrete Multidimensional Signals, H. Farid & E. P. Simoncelli, IEEE 2004
    n : number of taps, either 3 or 5
    returns p, d : p is the interpolation filter, d, is the derivative filter
    e.g:
    p,d = farid_filters(5)
    darr_dcol = scipy.signal.sepfir2d(arr, p, d)
    darr_drow = scipy.signal.sepfir2d(arr, d, p)
    """
    if n == 3:
        return [0.229879, 0.540242, 0.229879], [0.425287, 0.0, -0.425287]
    elif n == 5:
        return [0.037659, 0.249153, 0.426375, 0.249153, 0.037659], [0.109604, 0.276691, 0.0, -0.276691, -0.109604]

def isoutofbounds(indices, dims):
    """ isoutofbounds(indices, dims)
    Check whether indices are within the bounds of dims
    """
    indices = np.asarray(indices)
    dims = np.asarray(dims)
    z = np.zeros_like(dims)
    return np.any(np.logical_or(indices < z, indices >= dims), -1)

def warp_qmc(image, inverse_map, map_args={}, output_shape=None, nval=0., n=9, mode='constant', cval=0., preserve_range=False):
    """ Warp a 2D image according to a given coordinate transformation.
    
    This version of warp integrates across each output pixel area using
    the Quasi-Monte Carlo method. It may produce cleaner results when
    the warp involves non-linear stretching.
    
    Parameters
    ----------
    image : ndarray
        Input image
    inverse_map : callable ``rc = f(rc, **kwargs)``
        Inverse coordinate map, which transforms coordinates in the output
        image into the corresponding coordinates in the input image.
        This is a function which takes an ``(M, 2)`` array of ``(row, col)``
        coordinates in the output image to their corresponding coordinates
        in the input image. If either coordinate is non-finite (i.e. 
        ``any(!isfinite((row, col))) is True``) the output value will be
        set to `nval`. Extra parameters to the function can be specified
        through `map_args`.
        Note that `inverse_map` must operate on floating point coordinates.
        Pixels are treated as squares spanning ``(r, c)`` to 
        ``(r + 1, c + 1)`` and centered on ``(r + .5, c + .5)``.
    map_args : dict, optional
        Keyword arguments passed to `inverse_map`.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved. Note that, even for multi-band images, only rows
        and columns need to be specified.
    nval : float, optional
        The value given to pixels with any non-finite input coordinate.
    n : int, optional
        Number of integration samples. Sample points are chosen using a Sobol
        sequence in the unit square.
    mode : {'constant', 'raise', 'wrap', 'clip'}, optional
        Points outside the boundaries of the input are filled according to the
        given mode. Modes match the behavior of `ravel_multi_index`.
    cval : float, optional
        In mode 'constant', the value outside the image boundaries.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
    
    Returns
    -------
    warped : double ndarray
        The warped input image.
    
    Notes
    -----
    - The input image is converted to a `double` image.
    - `n` increases the time complexity linearly.
    """
    #check and sanitize inputs
#    if preserve_range:
#        image = np.asarray(image, np.double)
#    else:
#        image = ski.img_as_float(image)

    input_shape = np.array(image.shape)

    if output_shape is None:
        output_shape = input_shape
    else:
        if len(input_shape) == 3 and len(output_shape) == 2:
            output_shape = (output_shape[0], output_shape[1], input_shape[2])

    def coord_map(*args):
        return inverse_map(*args, **map_args)

    if n <= 0:
        n = 1
    nd = 2

    #replace out-of-bounds indices before ravel_multi_index causes problems
    replace_oob = False
    if mode == 'constant':
        mode = 'raise'
        replace_oob = True

    #generate output indices
    out_idx = np.moveaxis(np.indices(
        output_shape[:nd]), 0, -1).reshape((-1, nd))
    out_pts = out_idx.astype(np.float64)

    #generate sampling offsets
    if n > 1:
        # points in 0 to 1 on each dimension (ie. spanning the pixel)
        offsets = sobol(nd).generate(n,20)
    else:
        offsets = 0.5*np.ones(nd)

    #allocate output
    out = np.empty(output_shape, dtype=image.dtype)
    #partially flatten images for easy indexing
    out_flat = out.reshape((-1,)+output_shape[nd:])
    image_flat = image.reshape((-1,)+image.shape[nd:])

    #allocate temporary storage for accumulating values
    if n > 1:
        out_acc = np.empty(out_flat.shape, dtype=out.dtype)

    for o in offsets:
        #do inverse transform to get image points
        image_pts = coord_map(out_pts + o)

        #find non-finite and out-of-bounds indices
        image_bad = np.any(np.logical_not(np.isfinite(image_pts)), -1)
        if not np.any(image_bad):
            image_bad = False
        if replace_oob:
            image_oob = isoutofbounds(image_pts, image.shape[:nd])
            if not np.any(image_oob):
                image_oob = False
        else:
            image_oob = False

        #get indices by rounding down
        image_idx = np.floor(image_pts).astype(int)

        #set masked indices to 0 so that ravel_multi_index will work
        image_idx[np.logical_or(image_bad, image_oob)] = 0

        #ravel the indices so that we can take them from the flattened array
        image_idx = np.ravel_multi_index(np.moveaxis(
            image_idx, -1, 0), image.shape[:nd], mode)

        #allocate temporary storage for masking
        out_tmp = np.empty(out_flat.shape, dtype=out.dtype)
        out_tmp[:] = image_flat[image_idx]
        out_tmp[image_bad] = nval
        if replace_oob:
            out_tmp[image_oob] = cval

        if n == 1:  # no integration, store directly
            out_flat[:] = out_tmp.astype(out.dtype)
        else:  # accumulate
            out_acc[:] += out_tmp
    if n > 1:
        out_acc /= n
        out_flat[:] = out_acc.astype(out.dtype)
    return out


def circle_mask(radius,size=None,offset=None,inner=0,subsample_limit=4,center=False):
    """subsampled circle, with sub-pixel offset"""
    def subsample(x,y,sz,r,lim):
        d = np.hypot(x, y)
        if lim==0: #hit recursion limit
            #return area if x,y is inside circle
            return sz**2 if d < r else 0.0
        elif d + 0.70711*sz < r: #totally inside circle
            return sz**2
        elif d - 0.70711*sz > r: #totally outside circle
            return 0.0
        else: #on edge, recurse into quadrants
            s,o = sz/2, sz/4
            return subsample(x+o,y+o,s,r,lim-1) + \
                   subsample(x+o,y-o,s,r,lim-1) + \
                   subsample(x-o,y-o,s,r,lim-1) + \
                   subsample(x-o,y+o,s,r,lim-1)
    if offset is None:
        y0,x0 = 0,0
    else:
        y0,x0 = offset
    if size is None:
        size=2*radius+1
    if np.isscalar(size):
        size = (size,size)
    if center:
        y0 += 0.5*size[0]-0.5-radius
        x0 += 0.5*size[1]-0.5-radius
    coeffs = np.empty(size)
    for r in range(size[0]):
        for c in range(size[1]):
            x,y = c-radius,r-radius
            coeffs[r,c] = subsample(x-x0,y-y0,1,radius,subsample_limit)
            if inner > 0:   
                coeffs[r,c] -= subsample(x-x0,y-y0,1,inner,subsample_limit) 
    return coeffs