#image_utils.py

#This is free and unencumbered software released into the public domain.

import numpy as np

#image derivative filters consist of 2 separable filters
# the interpolation filter and the derivative filter
# the interpolation filter is applied to the orthogonal axis to the derivative filter
def sobel_filters():
    return [1,2,1],[1,0,-1]

def farid_filters(n=3):
    """From Differentiation of Discrete Multidimensional Signals, H. Farid & E. P. Simoncelli, IEEE 2004
    """
    if n == 3:
        return [0.229879, 0.540242, 0.229879], [0.425287, 0.0, -0.425287]
    elif n == 5:
        return [0.037659, 0.249153, 0.426375, 0.249153, 0.037659], [0.109604, 0.276691, 0.0, -0.276691, -0.109604]

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
