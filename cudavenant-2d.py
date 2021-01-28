# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:16:47 2021

@author: basti
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy



a = numpy.random.randn(1000,1000)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)
  
func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
print(a)

# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import numpy as np

# x = np.random.rand(5).astype(np.float32)
# x_gpu = gpuarray.to_gpu(x)
# h = cublasCreate()
# m = cublasIsamax(h, x_gpu.size, x_gpu.gpudata, 1)
# cublasDestroy(h)
# np.allclose(m, np.argmax(abs(x.real) + abs(x.imag)))



