import numpy as np

class sobol:
    """Sobol sequence
    Adapted from code by Corrado Chisari under the MIT License
    http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
    """
    @staticmethod
    def i4_bit_hi1(n):
        i = int ( n )
        bit = 0
        while i > 0:
            bit += 1
            i //= 2 #divide is faster than right shift ?!
        return bit
    
    @staticmethod
    def i4_bit_lo0(n):
        bit = 1
        i = int ( n )
        while (i & 1) != 0:
            i //= 2 #divide by 2 is faster than right shift ?!
            bit += 1
        return bit

    _dim_max = 40
    _log_max = 30
    _poly = [1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109, 103, 115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171, 213, 191, 253, 203, 211, 239, 247, 285, 369, 299]
    _atmost = 2**_log_max - 1
    _maxcol = i4_bit_hi1.__func__(_atmost) #.__func__ is necessary because sobol isn't defined yet!
    
    _v = np.zeros((_dim_max,_log_max),np.int32)
    _v[0:40,0] = np.transpose([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    _v[2:40,1] = np.transpose([1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3])
    _v[3:40,2] = np.transpose([7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 3, 1, 7, 5, 1, 3, 3, 7, 5, 1, 1, 5, 7, 7, 5, 1, 3, 3])
    _v[5:40,3] = np.transpose([1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3, 15, 7, 9, 13, 9, 1, 11, 7, 5, 15, 1, 15, 11, 5, 3, 1, 7, 9])
    _v[7:40,4] = np.transpose([9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1, 25, 29, 3, 31, 11, 5, 23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 9])
    _v[13:40,5] = np.transpose([37, 33, 7, 5, 11, 39,63, 27, 17, 15, 23, 29, 3, 21, 13, 31, 25, 9, 49, 33, 19, 29, 11, 19, 27, 15, 25])
    _v[19:40,6] = np.transpose([13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3, 113, 61, 89, 45, 107])
    _v[37:40,7] = np.transpose([7, 23, 39 ])
    _v[0,0:_maxcol] = 1

    def __init__(self, d, seed=0):
        if d < 1 or sobol._dim_max < d:
            raise RuntimeError('d ({}) out of range: 1 <= d <= {}'.format(d, sobol._dim_max))
        self.d = d
        #initialize the remaining values of v given d
        self.v = np.copy(sobol._v)
        for i in range(1, d):
            #the bits of _poly[i] gives the form of polynomial I
            #find the degree of _polynomial I from binary encoding
            j = sobol._poly[i]
            m = sobol.i4_bit_hi1(j)
            #expand this bit pattern to separate components
            includ = np.zeros(m,np.int32)
            for k in range(m-1, -1, -1):
                includ[k] = (j & 1)
                j //= 2
            #calculate the remaining elements of row I as explained in Bratley & Fox, section 2
            for j in range(m, sobol._maxcol):
                newv = self.v[i,j-m]
                l = 1
                for k in range(m):
                    l *= 2
                    if includ[k]:
                        newv ^= l*self.v[i,j-k-1]
                self.v[i,j] = newv
        #multiply columns of V by appropriate power of 2
        l = 1
        for j in range(sobol._maxcol-2, -1, -1):
            l *= 2
            self.v[0:d,j] *= l
        #recipd is 1/(common denominator of elements in v)
        self.recipd = 1.0/(2*l)
        self.seed = None
        self.lastq = None
        self.reseed(seed)

    def reseed(self, seed):
        """set a new seed"""
        if seed == self.seed: #nothing to do
            return
        seed = int(seed)
        if seed < 0:
            seed = 0
        if seed == 0 or seed < self.seed:
            #full reset
            self.lastq = np.zeros(self.d,np.int32)
            self.seed = 0
        #advance lastq until we hit the new seed
        for s in range(self.seed, seed):
            l = sobol.i4_bit_lo0(s)
            for i in range(self.d):
                self.lastq[i] ^= self.v[i,l-1]
        self.seed = seed
        
    def next(self):
        """generate the next value in the sequence"""
        l = sobol.i4_bit_lo0(self.seed)
        if sobol._maxcol < l:
            raise RuntimeError('Too many calls: L = {} >= MAXCOL = {}'.format(l, sobol._maxcol))
        quasi = np.zeros(self.d)
        for i in range(self.d):
            quasi[i] = self.lastq[i]*self.recipd
            self.lastq[i] ^= self.v[i,l-1]
        self.seed += 1
        return quasi
    
    __next__ = next
    __call__ = next

    def generate(self, n, skip=0):
        """generate a sobol sequence
        n : the number of points to generate
        skip : the number of initial points to skip
        """
        r = np.zeros((self.d,n))
        self.reseed(self.seed + skip)
        for j in range(n):
            r[:,j] = self.next()
        return r