#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif


void checkCUDAError(const char *msg, int line);
void cudaPBFUpdateWrapper(float dt);
void initCuda(int N);
void cudaUpdateVBO(float * vbodptr, int width, int height);
void freeCuda();
void setLockNum(int x);
#endif
