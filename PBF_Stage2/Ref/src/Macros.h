#ifndef MACROS_H_
#define MACROS_H_
#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 0
#define PRESSURE 1
#define HEAP 0

#define BOX_X 20
#define BOX_Y 20
#define BOX_Z 100

#define SOLVER_ITERATIONS 3
#define MAX_NEIGHBORS 40
#define REST_DENSITY 2000.0f
#define H 2.0f // smoothing radius

#define POW_H_9 (float)(H*H*H*H*H*H*H*H*H) // h^9
#define POW_H_6 (float)(H*H*H*H*H*H) // h^6
#define RELAXATION .01 // relaxation term in lambda calculation



#define PI_FLOAT				3.141592653589793f
#define DELTA_Q				(float)(0.1*core_radius)

#define SQR(x)					((x) * (x))
#define CUBE(x)					((x) * (x) * (x))
#define POW6(x)					(CUBE(x) * CUBE(x))
#define POW9(x)					(POW6(x) * CUBE(x))


#if PRESSURE == 1
	#define DELTA_Q (float)(0.1*H)
	#define PRESSURE_K 0.1
	#define PRESSURE_N 6
#endif


#endif