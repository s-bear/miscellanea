#image_utils.py

#This is free and unencumbered software released into the public domain.

import numpy as np
from scipy.ndimage.filters import convolve1d
from .sobol import sobol
from PIL import Image, PngImagePlugin
from numba import njit, prange


def save_png16(fname, arr, title=None):
    """save an array as a 16-bit png, the data range is stored in the text chunk with keys 'min' and 'max'
    Workaround for a PIL bug https://github.com/python-pillow/Pillow/issues/2970"""
    vmin, vmax = np.amin(arr), np.amax(arr)
    buf = ((arr-vmin)*((2**16-1)/(vmax-vmin))).astype(np.uint16)
    img = Image.frombytes('I', buf.T.shape, buf, 'raw', 'I;16')
    nfo = PngImagePlugin.PngInfo()
    if title is not None:
        nfo.add_text('Title', title)
    nfo.add_text('min', '{}'.format(vmin))
    nfo.add_text('max', '{}'.format(vmax))
    img.save(fname, 'png', pnginfo=nfo)


def load_png16(fname):
    """load a 16-bit png into an array, scaling data to 'min' and 'max' stored in the text chunk"""
    with Image.open(fname) as img:
        if hasattr(img, 'text') and 'min' in img.text and 'max' in img.text:
            vmin = float(img.text['min'])
            vmax = float(img.text['max'])
            arr = np.array(img).astype(np.float64) * \
                ((vmax-vmin)/(2**16-1)) + vmin
        else:
            arr = np.array(img)
        return arr

@njit(parallel=True)
def _laea(za,za0,out):
    """lambert azimuth equal area projection helper"""
    z0, a0 = za0[0], za0[1]
    z, a = za[0], za[1]

    ca, sa = np.cos(a-a0), np.sin(a-a0)
    cz, sz = np.cos(z), np.sin(z)
    cz0, sz0 = np.cos(z0), np.sin(z0)

    k = np.sqrt(2/(1+cz0*cz+sz0*sz*ca))

    out[0] = k*sz*sa
    out[1] = k*(sz0*cz - cz0*sz*ca)

def laea(za, za0=None, out=None):
    """Lambert Azimuthal Equal-Area Projection, per http://mathworld.wolfram.com/LambertAzimuthalEqual-AreaProjection.html
    N.B. This routine uses image coordinates with +x as East, and +y as South. That is, 0 azimuth is in the -y direction
    and 90 deg azimuth is in the +x direction.
    The equator (zenith=pi/2) is at radius sqrt(2) in the x-y plane. The antipole is at radius 2.
    za: shape = (2,...)
        za[0] = zenith, za[1] = azimuth
    za0: pole coordinates, defaults to [0,0] if None
    returns: xy
    """
    za = np.asarray(za)
    if out is None:
        xy = np.empty_like(za)
    else:
        xy = out
    if za0 is None:
        za0 = [0, 0]
    _laea(za,za0,xy)
    return xy

@njit(parallel=True)
def _laea_i(xy,za0,out):
    """inverse laea helper"""
    z0, a0 = za0[0], za0[1]
    cz0, sz0 = np.cos(z0), np.sin(z0)
    x, y = xy[0], xy[1]
    R2 = x**2 + y**2
    out[0] = np.arccos(0.5*(y*np.sqrt(4-R2)*sz0 - (R2-2)*cz0))
    out[1] = a0 + np.arctan2(x*np.sqrt(-R2*(R2-4)), -
                            np.sqrt(R2)*((R2-2)*sz0 + y*np.sqrt(4-R2)*cz0))

def laea_i(xy, za0=None, out=None):
    """inverse Lambert Azimuthal Equal-Area Projection, per http://mathworld.wolfram.com/LambertAzimuthalEqual-AreaProjection.html
    N.B. this routine uses image coordinates with +x as East, and +y as South. That is, 0 azimuth is in the -y direction
    and 90 deg azimuth is in the +x direction.
    The equator (zenith=pi/2) is at radius sqrt(2) in the x-y plane. The antipole is at radius 2.
    xy: shape = (2,...)
    za0: pole coordinats, defaults to [0,0] if None
    returns: za, za[0] = zenith, za[1] = azimuth
    """
    xy = np.asarray(xy)
    if out is None:
        za = np.empty_like(xy)
    else:
        za = out
    if za0 is None:
        za0 = [0, 0]
    _laea_i(xy,za0,za)
    return za

@njit(parallel=True)
def _cea(za,za0,out):
    z0, a0 = za0[0], za0[1]
    z, a = za[0], za[1]
    out[0] = (a-a0)*np.sin(z0)
    out[1] = -np.cos(z)/np.sin(z0)

def cea(za,za0=None,out=None):
    """Cylindrical Equal-Area Projection, per http://mathworld.wolfram.com/CylindricalEqual-AreaProjection.html
    N.B. This routine uses image coordinates with +x as East and +y as South. That is, zenith=0 is at y=-1/sin(z0) and zenith=pi
    is at y = 1/sin(z0).
    Parameters:
      za : shape = (2, ...)
        za[0] = zenith angle, za[1] = azimuth angle (in radians)
      za0 : standard zenith and azimuth.
        The standard aziumth maps to the center of the projection (x=0). The standard zenith angle is the zenith with minimal
        distortion. Standard values are pi/2 (Lambert), pi/4 (Gall), pi/6 (Behrmann), 5*pi/18 (Balthasart)
    Returns:
      xy : shape = za.shape
    """
    za = np.asarray(za)
    if out is None:
        xy = np.empty_like(za)
    else:
        xy = out
    if za0 is None:
        za0 = [np.pi/2,0.]
    _cea(za,za0,xy)
    return xy

@njit(parallel=True)
def _cea_i(xy,za0,out):
    x,y = xy[0], xy[1]
    z0,a0 = za0[0], za0[1]
    out[0] = np.arccos(-y*np.sin(z0))
    out[1] = x/np.sin(z0) + a0

def cea_i(xy, za0=None, out=None):
    """inverse Cylindrical Equal-Area Projection, per http://mathworld.wolfram.com/CylindricalEqual-AreaProjection.html
    N.B. This routine uses image coordinates with +x as East and +y as South. That is, zenith=0 is at y=-1/sin(z0) and zenith=pi
    is at y = 1/sin(z0).
    Parameters:
      xy : shape = (2, ...)
      za0 : standard zenith and azimuth.
        The standard aziumth maps to the center of the projection (x=0). The standard zenith angle is the zenith with minimal
        distortion. Standard values are pi/2 (Lambert), pi/4 (Gall), pi/6 (Behrmann), 5*pi/18 (Balthasart)
    Returns:
      za : shape = xy.shape
        za[0] = zenith angle, za[1] = azimuth angle (in radians)
    """
    xy = np.asarray(xy)
    if out is None:
        za = np.empty_like(xy)
    else:
        za = out
    if za0 is None:
        za0 = [np.pi/2,0.]
    _cea_i(xy,za0,za)
    return za

@njit(parallel=True)
def _corr1d_0(input, filter, output, wrap=True, cval=0.0):
    """Correlation along axis 0, parallelized for C-order arrays
    Assuming filter is much smaller than axis"""
    #3 loops: rows, cols, filter along rows
    rows, cols = input.shape
    N = len(filter)
    n = N//2
    #access scans whole col of output at once for better cache coherency
    for r in range(rows):
        for c in prange(cols):
            output[r,c] = 0
            for i in range(N):
                j = r-n+i
                if wrap:
                    j %= rows
                if j >= 0 and j < rows:
                    output[r,c] += input[j,c]*filter[i]
                else:
                    output[r,c] += cval*filter[i]
    return output

@njit(parallel=True)
def _corr1d_1(input, filter, output, wrap=True, cval=0.0):
    """Correlation along axis 1, parallelized for C-order arrays
    Assuming filter is much smaller than axis"""
    #3 loops: rows, cols, filter along cols
    rows, cols = input.shape
    N = len(filter)
    n = N//2
    #access pattern scans whole col of input & output at once for better cache coherency
    for r in range(rows):
        for c in prange(cols):
            output[r,c] = 0
            for i in range(N):
                j = c-n+i
                if wrap:
                    j %= cols
                if j >= 0 and j < cols:
                    output[r,c] += input[r,j]*filter[i]
                else:
                    output[r,c] += cval*filter[i]
    return output

def sepfir2d(input, filters, axes=None, output=None, wrap=True, cval=0.0):
    """Apply separable filters to a 2D input
     Parallelized with Numba
    input : array_like, must be 2D
    filters : sequence of array_like
      Sequence of filters to apply along axes. If length 1, will apply the same filter to all axes
    axes : optional, sequence of int: axes to apply filters to must be valid for a 2D array
    output : array, optional
    wrap : bool, default True. If True, wrap out-of-bounds access
    cval : scalar, optional. Used for out-of-bounds access if wrap is False
    """
    if output is None:
        output = np.empty_like(input)
    tmp = output
    if np.isscalar(filters[0]):
        filters = [np.asarray(filters)]
    if axes is None:
        axes = list(range(len(filters)))
    if np.isscalar(axes):
        axes = [axes]
    if len(axes) > 1:
        tmp = np.empty_like(output)
        if len(filters) == 1:
            filters = [filters[0]]*len(axes)
        if len(axes) & 1 == 1: #pre-swap buffers so that the last write goes to output
            output,tmp = tmp, output
    for filt, ax in zip(filters,axes):
        filt = filt[::-1]
        output,tmp = tmp, output #swap buffers
        ax %= 2
        if ax == 0: _corr1d_0(input,filt,output,wrap,cval)
        else: _corr1d_1(input,filt,output,wrap,cval)
        input = output
    return output


def sepfirnd(input,filters,axes,output=None,mode='reflect',cval=0.0,origin=0):
    """Apply multiple 1d filters to input using scipy.ndimage.filters.convolve1d
    input : array_like
    filters : sequence of array_like
        Sequence of filters to apply along axes. If length 1, will apply the same filter to all axes.
    axes : sequence of int
    output : array, optional
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    cval : scalar, optional
    origin : int, optional
    """
    if output is None:
        output = np.empty_like(input)
    tmp = output
    if np.isscalar(filters[0]):
        filters = [np.asarray(filters)]
    if np.isscalar(axes):
        axes = [axes]
    if len(axes) > 1:
        tmp = np.empty_like(output)
        if len(filters) == 1:
            filters = [filters[0]]*len(axes)
        if len(axes) & 1 == 1: #pre-swap so that last write goes to output
            output,tmp = tmp,output 
    for filt,ax in zip(filters,axes):
        output,tmp = tmp,output #swap buffers
        convolve1d(input,filt,ax,output,mode,cval,origin)
        input = output
    return output

#image derivative filters consist of 2 separable filters
# the interpolation filter and the derivative filter
# the interpolation filter is applied to the orthogonal axis to the derivative filter
def sobel_filters():
    """Sobel differentiation filters
    returns p, d : p is the interpolation filter, d is the derivative filter
    e.g:
    p,d = sobel_filters()
    darr_dx = sepfirnd(arr, (p, d), (0,1)) #apply p to axis 0, d to axis 1
    darr_dy = sepfirnd(arr, (p, d), (1,0)) #apply p to axis 1, d to axis 0
    """
    return [1,2,1],[1,0,-1]

def farid_filters(n=3):
    """Farid's differentiation filters
    From Differentiation of Discrete Multidimensional Signals, H. Farid & E. P. Simoncelli, IEEE 2004
    n : number of taps, either 3 or 5
    returns p, d : p is the interpolation filter, d, is the derivative filter
    e.g:
    p,d = farid_filters(5)
    darr_dx = sepfirnd(arr, (p, d), (0,1)) #apply p to axis 0, d to axis 1
    darr_dy = sepfirnd(arr, (p, d), (1,0)) #apply p to axis 1, d to axis 0
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
    #if preserve_range:
    #    image = np.asarray(image, np.double)
    #else:
    #    image = ski.img_as_float(image)

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
    """subsampled circle, with sub-pixel offset
    Each pixel's value is the area of the pixel covered by the circle.
    """
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
