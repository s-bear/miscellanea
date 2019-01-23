#stats_utils.py

#This is free and unencumbered software released into the public domain.

#Statistics stuff...
import numpy as np
import numba
from numba import njit, vectorize, prange
from scipy import stats

_types = ['i1','u1','i2','u2','i4','u4','i8','u8','f4','f8']
@vectorize(["{T}({T},{T},{T})".format(T=t) for t in _types],target='parallel')
def wrap(x, vmin, vmax):
    """Wrap a value x within bounds (vmin,vmax)"""
    return (x-vmin) % (vmax-vmin)+vmin


@vectorize(["{T}({T},{T},{T})".format(T=t) for t in _types],target='parallel')
def clip(x, vmin, vmax):
    """Clip a value x within bounds (vmin, vmax)"""
    if x < vmin:
        return vmin
    elif x > vmax:
        return vmax
    else:
        return x
clamp = clip

#@vectorize(["i4({T},{T},{T},{T},i4)".format(T=t) for t in ('f4','f8')], target='parallel')
@vectorize
def _digitize(x, vmin, vmax, d, n):
    """Helper for digitize"""
    if x < vmin:
        return -1
    elif x > vmax:
        return n
    elif x == vmax:
        return n-1
    else:
        return np.uint32((x-vmin)*d) #n/(vmax-vmin))

@njit(parallel=True)
def digitize(x, vmin, vmax, n):
    """Return the bin index of x for n equally spaced bins in (vmin,vmax)
    For out of bounds x, returns -1 or n"""
    d = n/(vmax-vmin)
    return _digitize(x,vmin,vmax,d,n)

def bin_edges(vmin, vmax, n):
    """Return the bin edges for n equally spaced bins in (vmin, vmax)"""
    return np.linspace(vmin,vmax,n+1)

def bin_centers(vmin,vmax,n):
    """return the bin centers for n equally spaced bins in (vmin, vmax)"""
    dx_2 = (vmax-vmin)/(2*n)
    return np.linspace(vmin+dx_2,vmax-dx_2,n)

@njit(parallel=True)
def centroid2d(x,y,w):
    """Return the centroid of a 2D grid
    x, y : 1d array_like
      the positions of each w[r,c]. 
    w : 2d array_like
      2D array of weights. y corresponds to axis 0
    Returns
    mx, my
    """
    mx = 0.0
    my = 0.0
    s = 0.0
    for r in prange(w.shape[0]):
        for c in range(w.shape[1]):
            s += w[r,c]
            mx += x[c]*w[r,c]
            my += y[r]*w[r,c]
    mx /= s
    my /= s
    return (mx, my)

@njit(parallel=True)
def _histogram2d(out,x,y,nx,ny,xmin,xmax,ymin,ymax,d,keep):
    """helper for histogram2d"""
    dx = nx/(xmax-xmin)
    dy = ny/(ymax-ymin)
    nt = numba.config.NUMBA_NUM_THREADS
    for i in prange(out.size):
        out.flat[i] = 0
    #make a buffer for each thread:
    buf = [out] + [np.zeros(out.shape,out.dtype) for i in range(nt-1)]
    ss = (x.shape[0] + nt - 1)//nt
    if keep:
        for b in prange(nt):
            for i in range(b*ss,(b+1)*ss):
                if i >= x.shape[0]: break
                #do the histogram within each thread
                r = _digitize(x[i], xmin, xmax, dx, nx)
                c = _digitize(y[i], ymin, ymax, dy, ny)
                buf[b][r+1,c+1] += 1
    else:
        for b in prange(nt):
            for i in range(b*ss,(b+1)*ss):
                if i >= x.shape[0]: break
                r = _digitize(x[i], xmin, xmax, dx, nx)
                c = _digitize(y[i], ymin, ymax, dy, ny)
                if r >= 0 and r < nx and c >= 0 and c < ny:
                    buf[b][r,c] += 1
    #reduce back to the output buffer
    for i in prange(out.size):
        for b in range(1,nt):
            out.flat[i] += buf[b].flat[i]
    #multiply by d if necessary
    if d != 1:
        for i in prange(out.size):
            out.flat[i] *= d

def histogram2d(x,y,n=10,range=None,density=False,keep_outliers=False,out=None):
    """2D histogram with uniform bins. Accelerated by numba
    x, y: array_like
      x and y coordinates of each point. x and y will be flattened
    n : scalar or (nx, ny)
      number of bins in x and y
    range : None or ((xmin,xmax),(ymin,ymax))
      range of bins. If any is None, the min/max is computed
    density : optional, bool
      if True, compute bin_count / (sample_count * bin_area)
    keep_outliers : optional, bool
      if True, add rows and columns to each edge of the histogram to count the outliers
    out : array_like, optional, shape = (nx, ny)
      Array to store output. Note that for compatibility with numpy's histogram2d, out
      is indexed out[x,y]. If keep_outliers is True, out must have shape (nx+2,ny+2)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise RuntimeError("x and y must be same shape")
    x = x.reshape(-1)
    y = y.reshape(-1)
    if range is None:
        xmin,xmax = None,None
        ymin,ymax = None,None
    else:
        xmin,xmax = range[0]
        ymin,ymax = range[1]
    if xmin is None or xmax is None:
        xmm = aminmax(x)
        if xmin is None: xmin = xmm[0]
        if xmax is None: xmax = xmm[1]
    if ymin is None or ymax is None:
        ymm = aminmax(y)
        if ymin is None: ymin = ymm[0]
        if ymax is None: ymax = ymm[1]
    if np.isscalar(n):
        nx,ny = n,n
    else:
        nx,ny = n
    if keep_outliers:
        out_shape = (nx+2,ny+2)
    else:
        out_shape = (nx,ny)
    if density:
        # 1/ (sample_count * bin_area)
        d = (nx*ny)/(len(x)*(xmax-xmin)*(ymax-ymin))
        if out is None:
            out = np.empty(out_shape,np.float64)
    else:
        d = 1
        if out is None:
            out = np.empty(out_shape,np.uint64)
    _histogram2d(out, x,y,nx,ny,xmin,xmax,ymin,ymax,d,keep_outliers)
    return out

@njit(parallel=True)
def _histogram(out,x,nx,xmin,xmax,d,keep):
    """helper for histogram2d"""
    dx = nx/(xmax-xmin)
    nt = numba.config.NUMBA_NUM_THREADS
    for i in prange(out.size):
        out[i] = 0
    #make a buffer for each thread:
    buf = [out] + [np.zeros(out.shape,out.dtype) for i in range(nt-1)]
    ss = (x.shape[0] + nt - 1)//nt
    if keep:
        for b in prange(nt):
            for i in range(b*ss,(b+1)*ss):
                if i >= x.shape[0]: break
                #do the histogram within each thread
                j = _digitize(x[i], xmin, xmax, dx, nx)
                buf[b][j+1] += 1
    else:
        for b in prange(nt):
            for i in range(b*ss,(b+1)*ss):
                if i >= x.shape[0]: break
                j = _digitize(x[i], xmin, xmax, dx, nx)
                if j >= 0 and j < nx:
                    buf[b][j] += 1
    #reduce back to the output buffer
    for i in prange(out.size):
        for b in range(1,nt):
            out[i] += buf[b][i]
        #multiply by d if necessary
        if d != 1:
            out[i] *= d

def histogram(x,n=10,range=None,density=False,keep_outliers=False):
    """1D histogram with uniform bins. Accelerated by numba
    x: array_like
      x coordinates of each point. x will be flattened
    n : scalar
      number of bins in x
    range : None or (xmin,xmax)
      range of bins. If None, the min/max is computed
    density : optional, bool
      if True, compute bin_count / (sample_count * bin_size)
    keep_outliers : optional, bool
      if True, add rows and columns to each edge of the histogram to count the outliers
    """
    x = np.asarray(x)
    x = x.reshape(-1)
    if range is None:
        xmin,xmax = None,None
    else:
        xmin,xmax = range
    if xmin is None or xmax is None:
        xmm = aminmax(x)
        if xmin is None: xmin = xmm[0]
        if xmax is None: xmax = xmm[1]
    if keep_outliers:
        out_shape = (n+2,)
    else:
        out_shape = (n,)
    if density:
        # 1/ (sample_count * bin_area)
        d = n/(len(x)*(xmax-xmin))
        out = np.empty(out_shape,np.float64)
    else:
        d = 1
        out = np.empty(out_shape,np.uint64)
    _histogram(out, x,n,xmin,xmax,d,keep_outliers)
    return out

@njit(parallel=True)
def aminmax(a):
    """Return (min,max) of array using np.min, np.max
    The many options of np.min & max are not supported by Numba for now.
    """
    vmin = np.min(a)
    vmax = np.max(a)
    return vmin,vmax

def lanczos(a,n):
    """lanczos filter kernel, order a, n points"""
    x = np.linspace(-a+.5,a-.5,n)
    y = np.sinc(x)*np.sinc(x/a)
    return y/sum(y)

def bin(x,y,bins,xmin=None,xmax=None,min_count=1,keep_oob=True,keep_empty=True):
    """x and y are 1d ndarrays"""
    x = np.asarray(x)
    y = np.asarray(y)
    if np.isscalar(bins):
        #a number of bins, not the bins 
        if xmin is None:
            xmin = x.min()
        if xmax is None:
            xmax = x.max()
        nbins = bins
        #set up bins for digitize
        w = (xmax-xmin)/nbins
        bins = np.linspace(xmin, xmax, nbins+1)  # includes endpoint
        #bin midpoints
        mids = bins[:-1] + (w/2)
    else:
        nbins = len(bins)-1
        mids = (bins[:-1] + bins[1:])/2
    #digitize x -- get bin index for each x
    xidx = np.digitize(x, bins)
    #send ys to the appropriate bin
    y_binned = []
    not_empty = []
    if keep_oob:
        rng = range(nbins+2) #include out-of-bounds bins
    else:
        rng = range(1,nbins+1) #don't include out-of-bounds bins
    
    for i in rng:
        y_b = y[xidx==i]
        if len(y_b) >= min_count:
            y_binned.append(y_b)
            if not keep_empty:
                #we're not keeping empty bins, so the non-empty mids need to be tracked
                if i == 0: #the 0th bin is all numbers less than xmin
                    not_empty.append(-float('inf'))
                elif i == nbins+1: #the last bin is all greater than xmax
                    not_empty.append(float('inf'))
                else: 
                    not_empty.append(mids[i-1])
        elif keep_empty:
            y_binned.append(np.array([]))
    if keep_empty:
        return mids, y_binned
    else:
        return np.array(not_empty), y_binned



def bin_wrap(x,y,bins,xmin=None,xmax=None,center=0,min_count=1,keep_empty=True):
    """for binning periodic data (like angles)"""
    #center is a relative offset of the 1st bin's center from xmin
    #center should be in [0,0.5]
    #center = 0 -> 1st bin is centered on xmin
    #center = 0.5 -> 1st bin's left edge is at xmin
    x = np.asarray(x)
    y = np.asarray(y)
    if np.isscalar(bins):
        nbins = bins
        if xmin is None:
            xmin = x.min()
        if xmax is None:
            xmax = x.max()
        w = (xmax-xmin)/nbins
        bins = np.linspace(xmin, xmax, nbins+1) + w*(center-0.5)
    
    bmin = bins[0]
    bmax = bins[-1]
    brange = bmax - bmin
    x_wrapped = ((x-bmin) % brange)+bmin
    mids,y_binned = bin(x_wrapped,y,bins,min_count=min_count,keep_oob=False,keep_empty=keep_empty)
    #because the data was wrapped, we don't need the out-of-bounds bins
    return mids,y_binned

def likelihood_same_mean(mahal_dist,ndims=1):
    """What is the likelihood that you drew x from N(mu,sigma)
    mahal_dist = Mahalanobis distance (x-mu)/sigma or sqrt( (x-mu).T.dot(inv(S).dot((x-mu))), S is covariance matrix
    ndims = number of dimensions of x
    """
    return stats.chi2.cdf(mahal_dist**2,ndims)

def mahal_distance_for_likelihood(likelihood,ndims=1):
    return np.sqrt(stats.chi2.isf(1-likelihood, ndims))

def rms(x,axis=None,keepdims=False):
    return np.sqrt(np.mean(x**2, axis, keepdims=keepdims))

def mean_conf(data,conf=0.90):
    n = len(data)
    #mean, and std error of mean
    m,se = np.mean(data), stats.sem(data)
    #get the interval from Student's t dist
    iv = stats.t.interval(conf,n-1,scale=se)
    return m,iv

##########
# Angles #
##########

def angle_diff(a1, a2, period=2*np.pi):
    """(a1 - a2 + d) % (2*d) - d; d = period/2"""
    d = period/2
    return ((a1 - a2 + d) % (period)) - d

def angle_mean(ang, axis=None, period=2*np.pi):
    """returns the circular mean of angles"""
    #uses the 1st angular moment:
    a = 2*np.pi/period
    m1 = np.mean(np.exp(1j*ang*a),axis=axis)
    return np.angle(m1)/a

def angle_std(ang,axis=None,period=2*np.pi):
    """Returns the circular standard deviation of angles"""
    a = 2*np.pi/period
    m1 = np.mean(np.exp(1j*ang*a),axis=axis)
    return np.sqrt(-2*np.log(np.abs(m1)))/a

def angle_var(ang,axis=None,period=2*np.pi):
    a = 2*np.pi/period
    m1 = np.mean(np.exp(1j*ang*a),axis=axis)
    return -2*np.log(np.abs(m1))/a**2

def angle_mean_std(ang, axis=None, period=2*np.pi):
    """returns the circular mean and standard deviation of angles"""
    #take the 1st angular moment
    # nth moment: m_n = mean(exp(1j*ang)**n)
    a = 2*np.pi/period
    m1 = np.mean(np.exp(1j*ang*a),axis=axis)
    mean = np.angle(m1)/a
    std = np.sqrt(-2*np.log(np.abs(m1)))/a
    return mean, std

def angle_rms(ang, axis=None, period=2*np.pi):
    """returns the rms of angles, uses the property that rms(x)**2 = mean(x)**2 + std(x)**2"""
    #rms(x)**2 = mean(x)**2 + std(x)**2
    #sqrt(E[X**2]) = E[X]**2 + sqrt(E[(X - E[X])**2])
    m,s = angle_mean_std(ang,axis,period)
    return np.hypot(m, s)

def angle_wrap(p,period=2*np.pi):
    if np.isscalar(period):
        vmin = -period/2
        vmax = period/2
    else:
        vmin = period[0]
        vmax = period[1]
    return (p-vmin)%(vmax-vmin)+vmin

def angle_unwrap(p,axis=-1,period=2*np.pi):
    """angle_unwrap(p,axis=-1,period=2*pi)
    Unwrap angles by changing deltas between values to their minimum relative to the period
    
    Parameters
    ----------
    p : array_like
        Angles to be unwrapped
    period : float, default = 2*pi
        The maximum angle. Differences between subsequent p are set to their minimum relative to this value. Set to 2*pi for radians, 360 for degrees, 1 for turns, etc.
    axis : int, default = -1
        Axis along which unwrap will be performed.
    
    Returns
    -------
    out : ndarray
        unwrapped angles
    """
    p = np.asarray(p)
    nd = len(p.shape)
    dd = np.diff(p, axis=axis)
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    hc = period/2
    ddmod = np.mod(dd + hc, period) - hc
    np.copyto(ddmod, hc, where=(ddmod == -hc) & (dd > 0))
    ph_correct = ddmod - dd
    np.copyto(ph_correct, 0, where=abs(dd) < hc)
    up = np.array(p, copy=True, dtype='d')
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up


