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


