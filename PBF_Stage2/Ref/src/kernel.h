#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <string>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"

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
void setMeshFile(std::string s);
void setGravity(const glm::vec3& g);
#endif
