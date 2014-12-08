#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <thrust/transform.h> 
#include <thrust/sequence.h> 
#include <thrust/copy.h> 
#include <thrust/fill.h> 
#include <thrust/replace.h> 
#include <thrust/functional.h>

//#include <thrust\device_vector.h>
#include "utilities.h"
#include "kernel.h"
#include "gridStruct.h"


#include "smallObjLoader.h"
#include "Macros.h"

//GLOBALS
dim3 threadsPerBlock(blockSize);

int totalGridSize = (2 * (BOX_X + 2)) * (2 * (BOX_Y + 2)) * (BOX_Z + 2);
int numParticles;
string MeshFileName;// = "..\\..\\Models\\bunny_fu_low2.obj";
__device__ int LockNum=1000;
vec3 gravity(.0f);
int numGenerated;

const float scene_scale = 1; //size of the height map in simulation space

particle* particles;
int* neighbors;
int* num_neighbors;
int* grid_idx;
int* grid;



bool hitonce = false;

float wallMove = 0.0f;

bool cleanupFixedPoints=false;
bool ExtForceSet = false;

rigidbodyObj rigtest;
vec4* rigPredictedPos;
mat3* rigPredictedRot;
mat3  rigRotMat;

using namespace glm;

struct particleMassCenter{
	__host__ __device__ vec4 operator()(const particle& x) const{
		return x.position;
	}
};

struct particlePredictMassCenter{
	__host__ __device__ vec4 operator()(const particle& x) const{
		return x.pred_position;
	}
};
void setLockNum(int x){
	LockNum=x;
}
void setGravity(const vec3& g){
	ExtForceSet = true;
	gravity = g;
}
void setMeshFile(string s){
	MeshFileName = s;
}
void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 

__device__ bool Conditions(int index, int N, int LockNum){
	return index < N&&index>=LockNum;
}

__device__ bool ParticleConditions(int index, int N, particle* p, int LockNum, int LayerMask){
	return index < N && (p[index].LayerMask&LayerMask)&&!(p[index].LayerMask&FROZEN);
	//return index<N && (p[index].ID>LockNum);
}

__host__ __device__ unsigned int devhash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(float time, int index)
{
    thrust::default_random_engine rng(devhash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__ void sendToVBO(int N, particle* particles, float * vbo, int width, int height, float s_scale, unsigned int LockNum)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale_w = 1.0f;
    float c_scale_h = 1.0f;
	float c_scale_z = 1.0f;

    if(index<N)
    {
        vbo[4*index+0] = particles[index].position.x*c_scale_w;
		vbo[4*index+1] = particles[index].position.y*c_scale_h;
        vbo[4*index+2] = particles[index].position.z*c_scale_z;
		//the w component is used as a selector of render color
		vbo[4 * index + 3] = (particles[index].ID >= LockNum)?1.0f:0.0f;
    }
}

/*************************************
 * Device Methods for Solver
 *************************************/

__device__ float wPoly6Kernel(glm::vec3 p_i, glm::vec3 p_j){
	vec3 r(p_i-p_j);
	if (length(r) > H) return 0.000001f;
	return 315.0f / (64.0f * PI_FLOAT * POW9(H)) * CUBE(SQR(H) - dot(r, r));
}

__device__ glm::vec3 wGradientSpikyKernel(glm::vec3 p_i, glm::vec3 p_j){
	glm::vec3 r = p_i - p_j;

	float hr_term = H - glm::length(r);
	if (hr_term < 0.0f) return vec3(0.0f);
	float gradient_magnitude = 45.0f / (PI * POW_H_6) * hr_term * hr_term;
	float div = (glm::length(r) + 0.001f);
	return gradient_magnitude * 1.0f / div * r;
}

__device__ float calculateRo(particle* particles, glm::vec3 p, int* p_neighbors, int p_num_neighbors, int index){
	glm::vec3 p_j;
	float ro = 0.0f;
	for(int i = 0; i < p_num_neighbors; i++){
		glm::vec3 p_j(particles[p_neighbors[i + index * MAX_NEIGHBORS]].pred_position);
		double kv=wPoly6Kernel(p,p_j);
		if (kv < K_EPSILON) kv = 0.0f;
		ro+=kv;
	}
	return ro;
}

__device__ glm::vec3 calculateCiGradient(glm::vec3 p_i, glm::vec3 p_j){
	//glm::vec3 Ci = -1.0f / float(REST_DENSITY) * wGradientSpikyKernel(p_i, p_j);
	//vec3 p_j((*pit)->PredictedPos);
				//if(Particles[i]->id>(*pit)->id) continue;
	//Ci=pow(spikyGradient(p-p_j,core_radius).Length()/material.rest_density,2);
	//sum_gradients+=C_i_gradient;
	return wGradientSpikyKernel(p_i,p_j)/REST_DENSITY;
}

__device__ glm::vec3 calculateCiGradientAti(particle* particles, glm::vec3 p_i, int* neighbors, int p_num_neighbors, int index){
	glm::vec3 accum = glm::vec3(0.0f);
	for(int i = 0; i < p_num_neighbors; i++){
		accum += wGradientSpikyKernel(p_i, glm::vec3(particles[neighbors[i + index * MAX_NEIGHBORS]].pred_position));
	}
	glm::vec3 Ci = 1.0f / float(REST_DENSITY) * accum;
	return Ci;
}

/*************************************
 * Finding Neighboring Particles 
 *************************************/

struct GridElement{
	particle *particles[4*MAX_NEIGHBORS];
	//int lock;
	int size;
	GridElement(){
		//particles = new particle[MAX_NEIGHBORS];
		//lock = 0;
		size = 0;
	}
};

GridElement *grid_elements;
int *grid_lock;
//__device__ GridElement *sleeping_grid_elements;
const int grid_width = 2 * BOX_X / H + 1;
const int grid_depth = 2 * BOX_Y / H + 1;
const int grid_height = BOX_Z / H + 1;

__device__ int grid_index(int i, int j, int k){
	if (i < 0 || i >= grid_width ||
		j < 0 || j >= grid_depth ||
		k < 0 || k >= grid_height)
		return -1;
	return grid_width*(k*grid_depth + j) + i;
}
//__device__ GridElement & gridContains(int i, int j, int k){
//	return grid_elements[grid_index(i, j, k)];
//}

__device__ void add2grid(GridElement *target_grid, particle* p, int* grid_lock){
	bool wait = true;
	int i = (int)(float(p->pred_position[0] + BOX_X) / H);
	i = clamp(i, 0, grid_width - 1);
	int j = (int)(float(p->pred_position[1] + BOX_Y) / H);
	j = clamp(j, 0, grid_depth - 1);
	int k = (int)(float(p->pred_position[2]) / H);
	k = clamp(k, 0, grid_height - 1);
	int id = grid_index(i, j, k);
	

	while (wait){
		if (0 == atomicExch(&(grid_lock[id]), 1)){
			int size = target_grid[id].size;
			if (size<MAX_NEIGHBORS) target_grid[grid_index(i, j, k)].particles[size++] = p;
			target_grid[id].size = size;
			grid_lock[id] = 0;
			wait = false;
			
		}
		else if (target_grid[id].size >= MAX_NEIGHBORS) {
			//printf("Too many particles in one grid!!!!");
			wait = false;
		}
	}
}

__global__ void update_grid(GridElement* grid_elements, particle* particles, int N,int* grid_lock){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < N){
		add2grid(grid_elements, &particles[index],grid_lock);
	}
}

__global__ void clearHistory(GridElement* grid_elements, int totalGridSize){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < totalGridSize){
		//printf("I'm good %d\n", index);
		grid_elements[index].size = 0;
	}
}

__global__ void findParticleNeighbors(GridElement* grid_elements, particle* particles, int* neighbors, int* num_neighbors, int N, int LockNum){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ParticleConditions(index, N, particles, LockNum,FLUID)){
		num_neighbors[index] = 0;
		particle p = particles[index];
		int i = (int)(float(p.pred_position[0] + BOX_X) / H);
		i = clamp(i, 0, grid_width - 1);
		int j = (int)(float(p.pred_position[1] + BOX_Y) / H);
		j = clamp(j, 0, grid_depth - 1);
		int k = (int)(float(p.pred_position[2]) / H);
		k = clamp(k, 0, grid_height - 1);
		int id = grid_index(i, j, k);
		int neighborsNum = 0;
		int offset[] = {0,-1, 1 };
		for (int ioff = 0, iiter = i; ioff < 3; iiter = i + offset[++ioff]){
			if (iiter < 0 || iiter >= grid_width) continue;
			for (int joff = 0, jiter = j ; joff < 3; jiter = j + offset[++joff]){
				if (jiter<0 || jiter>grid_depth - 1) continue;
				for (int koff = 0, kiter = k ; koff < 3; kiter = k + offset[++koff]){
					if (kiter<0 || kiter>grid_height - 1) continue;
					GridElement* thisGrid = &grid_elements[grid_index(iiter, jiter, kiter)];
					int thisGridSize = thisGrid->size;
					for (int pi = 0; pi < thisGridSize&&neighborsNum<MAX_NEIGHBORS; pi++){
						particle piter = *thisGrid->particles[pi];
						if (p.ID == piter.ID) continue;
						if (length(p.pred_position - piter.pred_position) < H)
							neighbors[p.ID*MAX_NEIGHBORS + neighborsNum++] = piter.ID;
					}
				}
			}
		}

		num_neighbors[index] = neighborsNum;
	}
}

void findParticleNeighborsWrapper(particle* particles, int* neighbors, int* num_neighbors, int N, int LockNum){
	dim3 fullBlocksPerGrid((int)ceil(float(grid_width*grid_depth*grid_height) / float(blockSize)));
	dim3 fullBlocksPerGridParticles((int)ceil(float(N) / float(blockSize)));
	//printf("fullblockPerGrid x=%d y=%d z=%d\n", fullBlocksPerGrid.x, fullBlocksPerGrid.y, fullBlocksPerGrid.z);
	clearHistory << <fullBlocksPerGrid, blockSize >> >(grid_elements, grid_width*grid_depth*grid_height);
	checkCUDAErrorWithLine("clearGrid failed!");

	update_grid << <fullBlocksPerGridParticles, blockSize >> >(grid_elements, particles, N,grid_lock);
	checkCUDAErrorWithLine("findParticleGridIndex failed!");

	findParticleNeighbors << <fullBlocksPerGridParticles, blockSize >> >(grid_elements, particles, neighbors, num_neighbors, N, LockNum);
	checkCUDAErrorWithLine("findKNearestNeighbors failed!");
}
//
//
//// Clears grid from previous neighbors
//__global__ void clearGrid(int* grid, int totalGridSize){
//	int index = threadIdx.x + (blockIdx.x * blockDim.x);
//	if(index < totalGridSize){
//		grid[index] = -1;
//	}
//}
//
//
//// Matches each particles the grid index for the cell in which the particle resides
//__global__ void findParticleGridIndex(particle* particles, int* grid_idx, int N){
//	int index = threadIdx.x + (blockIdx.x * blockDim.x);
//	if(index < N){
//		int x, y, z;
//		glm::vec4 p = particles[index].pred_position;
//		x = int(p.x) + BOX_X + 2;
//		y = int(p.y) + BOX_Y + 2;
//		z = int(p.z) + 2;
//		grid_idx[index] = x + (2 * (BOX_X + 2) * y) + (4 * (BOX_X + 2) * (BOX_Y + 2) * z);
//	}
//}
//
//// Matches the sorted index to each of the cells
//__global__ void matchParticleToCell(int* gridIdx, int* grid, int N, int totalGridSize){
//	int index = threadIdx.x + (blockIdx.x * blockDim.x);
//	if(index < N){
//		if(index == 0){
//			grid[gridIdx[index]] = index;
//		}else if(gridIdx[index] != gridIdx[index - 1]){
//			if(gridIdx[index] >= 0 && gridIdx[index] < totalGridSize) grid[gridIdx[index]] = index;
//		}
//	}
//}
//
//// Finds the nearest K neighbors within the smoothing kernel radius
//__global__ void findKNearestNeighbors(particle* particles, int* gridIdx, int* grid, int* neighbors, int* num_neighbors, int N, int totalGridSize,int LockNum){
//	int index = threadIdx.x + (blockIdx.x * blockDim.x);
//	if (ParticleConditions(index, N, particles, LockNum)){
//		int heap_size = 0;
//		int x,y,z,idx;
//		float r;
//		glm::vec4 p_j, p = particles[index].pred_position;
//
//		// Find particle index
//		x = int(p.x) + BOX_X + 2;
//		y = int(p.y) + BOX_Y + 2;
//		z = int(p.z) + 2;
//
//		float max;
//		int m, max_index, begin, cell_position;
//
//		// Examine all cells within radius
//		// NOTE: checks the cube that circumscribes the spherical smoothing kernel
//		for(int i = int(-H + z); i <= int(H + z); i++){
//			for(int j = int(-H + y); j <= int(H + y); j++){
//				for(int k = int(-H + x); k <= int(H + x); k++){
//					idx = k + (2 * (BOX_X + 2) * j) + (4 * (BOX_X + 2) * (BOX_Y + 2) * i);
//
//					if(idx >= totalGridSize || idx < 0){
//						continue;
//					}
//
//					begin = grid[idx];
//
//					if(begin < 0) continue;
//
//					cell_position = begin;
//					while(cell_position < N && gridIdx[begin] == gridIdx[cell_position]){
//						if(cell_position == index){
//							++cell_position;
//							continue;
//						}
//						p_j = particles[cell_position].pred_position;
//						r = glm::length(p - p_j);
//
//						if(heap_size < MAX_NEIGHBORS){
//							if(r < H){
//								neighbors[index * MAX_NEIGHBORS + heap_size] = cell_position;
//								++heap_size;
//							}
//						}else{
//							max = glm::length(p - particles[neighbors[index * MAX_NEIGHBORS]].pred_position);
//							max_index = 0;
//							for(m = 1; m < heap_size; m++){
//								float d = glm::length(p - particles[neighbors[index * MAX_NEIGHBORS + m]].pred_position); 
//								if(d > max){
//									max = d;
//									max_index = m;
//								}
//							}
//
//							if(r < max && r < H){
//								neighbors[index * MAX_NEIGHBORS + max_index] = cell_position;
//							}
//						}
//
//						++cell_position;
//					}
//				}
//			}
//		}
//		num_neighbors[index] = heap_size;
//	}
//}
//
//// Wrapper to find neighbors using hash grid
//void findNeighbors(particle* particles, int* grid_idx, int* grid, int* neighbors, int N,int LockNum){
//	dim3 fullBlocksPerGrid((int)ceil(float(totalGridSize) / float(blockSize)));
//	dim3 fullBlocksPerGridParticles((int)ceil(float(N)/float(blockSize)));
//
//	// Clear Grid
//	clearGrid<<<fullBlocksPerGrid, blockSize>>>(grid, totalGridSize);
//	checkCUDAErrorWithLine("clearGrid failed!");
//
//	// Match particle to index
//	findParticleGridIndex<<<fullBlocksPerGridParticles, blockSize>>>(particles, grid_idx, N);
//	checkCUDAErrorWithLine("findParticleGridIndex failed!");
//
//	// Cast to device pointers
//	thrust::device_ptr<int> t_grid_idx = thrust::device_pointer_cast(grid_idx);
//	thrust::device_ptr<particle> t_particles = thrust::device_pointer_cast(particles);
//
//	// Sort by key
//	thrust::sort_by_key(t_grid_idx, t_grid_idx + N, t_particles);
//	checkCUDAErrorWithLine("thrust failed!");
//
//	// Match sorted particle index
//	matchParticleToCell<<<fullBlocksPerGridParticles, blockSize>>>(grid_idx, grid, N, totalGridSize);
//	checkCUDAErrorWithLine("matchParticletoCell failed!");
//
//	// Find K nearest neighbors
//	findKNearestNeighbors<<<fullBlocksPerGridParticles, blockSize>>>(particles, grid_idx, grid, neighbors, num_neighbors, N, totalGridSize,LockNum);
//	checkCUDAErrorWithLine("findKNearestNeighbors failed!");
//}


/*************************************
 * Kernels for Jacobi Solver
 *************************************/

__global__ void calculateLambda(particle* particles, int* neighbors, int* num_neighbors, int N,int LockNum){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(ParticleConditions(index,N,particles,LockNum,FLUID)){
		int k = num_neighbors[index];
		glm::vec3 p = glm::vec3(particles[index].pred_position);

		float p_i = calculateRo(particles, p, neighbors, k, index);
		float C_i = (p_i / REST_DENSITY) - 1.0f;

		
		float C_i_gradient, sum_gradients = 0.0f;
		for(int i = 0; i < k; i++){
			// Calculate gradient when k = j
			C_i_gradient = glm::length(calculateCiGradient(p, glm::vec3(particles[neighbors[i + index * MAX_NEIGHBORS]].pred_position)));
			sum_gradients += (C_i_gradient * C_i_gradient);

		}

		// Add gradient when k = i
		C_i_gradient = glm::length(calculateCiGradientAti(particles, p, neighbors, k, index));
		sum_gradients += (C_i_gradient * C_i_gradient);

		float sumCi = sum_gradients + RELAXATION;
		particles[index].lambda = -1.0f * (C_i / sumCi); 
	}
}

__global__ void calculateDeltaPi(particle* particles, int* neighbors, int* num_neighbors, int N,int LockNum){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (ParticleConditions(index, N, particles, LockNum,FLUID)){
		int k = num_neighbors[index];
		glm::vec3 p = glm::vec3(particles[index].pred_position);
		float l = particles[index].lambda;
		
		glm::vec3 delta = glm::vec3(0.0f);
		int p_j_idx;
#if PRESSURE == 1
		float k_term;
		glm::vec3 d_q = DELTA_Q * glm::vec3(1.0f) + p;
#endif
		float s_corr = 0.0f;
		for(int i = 0; i < k; i++){
			p_j_idx = neighbors[i + index * MAX_NEIGHBORS];
#if PRESSURE == 1
			float poly6pd_q = wPoly6Kernel(p, d_q);
			if(poly6pd_q < EPSILON) k_term = 0.0f;
			else k_term = wPoly6Kernel(p, glm::vec3(particles[p_j_idx].pred_position)) / poly6pd_q;
			s_corr = -1.0f * PRESSURE_K * pow(k_term, PRESSURE_N);
#endif
			delta += (l + particles[p_j_idx].lambda + s_corr) * wGradientSpikyKernel(p, glm::vec3(particles[p_j_idx].pred_position));
		}
		particles[index].delta_pos = 1.0f / REST_DENSITY * delta;
	}
}

__global__ void calculateCurl(particle* particles, int* neighbors, int* num_neighbors, int N,int LockNum){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (ParticleConditions(index, N, particles, LockNum,FLUID)){
		int k = num_neighbors[index];
		glm::vec3 p = glm::vec3(particles[index].pred_position);
		glm::vec3 v = particles[index].velocity;

		int j_idx;
		glm::vec3 v_ij, gradient, accum = glm::vec3(0.0f);
		for(int i = 0; i < k; i++){
			j_idx = neighbors[i + index * MAX_NEIGHBORS];
			v_ij = particles[j_idx].velocity - v;
			gradient = wGradientSpikyKernel(p, glm::vec3(particles[j_idx].pred_position));
			accum += glm::cross(v_ij, gradient);
		}
		particles[index].curl = accum;
	}
}

__global__ void applyVorticity(particle* particles, int* neighbors, int* num_neighbors, int N,int LockNum){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(ParticleConditions(index,N,particles,LockNum,FLUID)){
		int k = num_neighbors[index];
		glm::vec3 p = glm::vec3(particles[index].pred_position);
		glm::vec3 w = particles[index].curl;

		int j_idx;
		float mag_w;
		glm::vec3 r, grad = glm::vec3(0.0f);
		for(int i = 0; i < k; i++){
			j_idx = neighbors[i + index * MAX_NEIGHBORS];
			r = glm::vec3(particles[j_idx].pred_position) - p;
			mag_w = glm::length(particles[j_idx].curl - w);
			grad.x += mag_w / r.x;
			grad.y += mag_w / r.y;
			grad.z += mag_w / r.z;
		}
		
		glm::vec3 vorticity, N;
		N = 1.0f/(glm::length(grad) + .001f) * grad;
		vorticity = float(RELAXATION) * (glm::cross(N, w));
		particles[index].external_forces += vorticity;
	}
}


__global__ void initializeParticles(int N, particle* particles,int LockNum=INT_MAX)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	float gravity = -9.8f;
	if (Conditions(index, N, LockNum))
    {
		particle p = particles[index];
		glm::vec3 rand = (generateRandomNumberFromThread(1.0f, index)-0.5f);
		p.ID=index;
		p.LayerMask = FLUID;
		p.position.x = (index%20)-9.5f;
		p.position.y = ((index/20)%20)-9.5f;
		p.position.z = (index/400)+10.0f+0.05f*rand.z;
		p.position.w = 1.0f;
		//p.position=glm::vec4(index%9-3.5f,(index/9)%20-9.5f,5.0f+index/180,1.0f);
		p.pred_position = p.position;

		p.velocity = glm::vec3(0.0f);
		
		p.external_forces = glm::vec3(0.0f,0.0f,gravity);
		
		particles[index] = p;
    }
	else if(index<N){
		particle p=particles[index];
		p.ID=index;
		//p.LayerMask = CONTAINER;
		p.velocity=glm::vec3(0.0f);
		p.external_forces=glm::vec3(0.0f,0.0f,gravity);
		particles[index]=p;
	}
}

__global__ void setExternalForces(int N, particle* particles, int LockNum,vec3 extForce){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (ParticleConditions(index,N,particles,LockNum,FLUID|RIGID_BODY)){
		particles[index].external_forces = extForce;
	}
}

//Simple Euler integration scheme
__global__ void applyExternalForces(int N, float dt, particle* particles,int LockNum)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (ParticleConditions(index,N,particles,LockNum,FLUID|RIGID_BODY)){
		particle p = particles[index];

		p.velocity+=dt*p.external_forces;
		
		p.delta_pos=glm::vec3(0.0f);
		p.pred_position=p.position+vec4(p.velocity*dt,0.0);
		



		particles[index] = p;
	}
}

__global__ void updatePosition(int N, particle* particles,int LockNum=0)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ParticleConditions(index, N, particles, LockNum,FLUID|RIGID_BODY)){
		//if (length(particles[index].position - particles[index].pred_position) > frozenDistance)
			particles[index].position = particles[index].pred_position;
		
	}
	/*if (index < N){
		particles[index].LayerMask &= ~FROZEN;
	}*/
	//if(particles[index].ID<=LockNum){
	//	particles[index].velocity=vec3(0.0f);
	//	particles[index].curl=vec3(0.0f);
	//	//particles[index].external_forces=vec3(0.0f);
	//}
}

__global__ void updatePredictedPosition(int N, particle* particles,int LockNum=0)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ParticleConditions(index, N, particles, LockNum,FLUID)){
		particles[index].pred_position += glm::vec4(particles[index].delta_pos,0.0f);
	}
}

__global__ void updateVelocity(int N, particle* particles, float dt,int LockNum)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ParticleConditions(index,N,particles,LockNum,FLUID|RIGID_BODY)){
		particles[index].velocity = glm::vec3((1.0f/dt)*(particles[index].pred_position - particles[index].position));
		if (length(particles[index].velocity) > 20.0f)
			particles[index].velocity = 20.0f*normalize(particles[index].velocity);
		/*if (length(particles[index].velocity) < frozenDistance)
			particles[index].LayerMask |= FROZEN;*/
	}
}

__global__ void boxCollisionResponse(int N, particle* particles, int LockNum){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ParticleConditions(index,N,particles,LockNum,FLUID|RIGID_BODY)){
		vec3 randv = generateRandomNumberFromThread(N, index);
		if( particles[index].pred_position.z < 0.0f){
			particles[index].pred_position.z = 0.001f*randv.z+0.01f;
			glm::vec3 normal = glm::vec3(0,0,1);
			
			particles[index].velocity.z = collision_restitution*abs(particles[index].velocity.z);
		}
		if( particles[index].pred_position.z > BOX_Z){
			particles[index].pred_position.z = BOX_Z - 0.001f*randv.z-0.01f;
			glm::vec3 normal = glm::vec3(0,0,-1);
			
			particles[index].velocity.z = -collision_restitution*abs(particles[index].velocity.z);
		}
		if( particles[index].pred_position.y < -BOX_Y){
			particles[index].pred_position.y = -BOX_Y + 0.001f*randv.y+0.01f;
			glm::vec3 normal = glm::vec3(0,1,0);
			
			particles[index].velocity.y = collision_restitution*abs(particles[index].velocity.y);
		}
		if( particles[index].pred_position.y > BOX_Y){
			particles[index].pred_position.y = BOX_Y - 0.001f*randv.y-0.01f;
			glm::vec3 normal = glm::vec3(0,-1,0);
			
			particles[index].velocity.y = -collision_restitution*abs(particles[index].velocity.y);
		}
		if( particles[index].pred_position.x < -BOX_X){
			particles[index].pred_position.x = -BOX_X + 0.001f*randv.x+0.01f;
			glm::vec3 normal = glm::vec3(1,0,0);
			
			particles[index].velocity.x = collision_restitution*abs(particles[index].velocity.x);
		}
		if( particles[index].pred_position.x > BOX_X){
			particles[index].pred_position.x = BOX_X - 0.001f*randv.x-0.01f;
			glm::vec3 normal = glm::vec3(-1,0,0);
			
			particles[index].velocity.x = -collision_restitution*abs(particles[index].velocity.x);
		}
		
	}
}


/*************************************
* shape matching *
*************************************/

void jacobiRotate(mat3 &A, mat3 &R, int p, int q){
	// rotates A through phi in pq-plane to set A(p,q) = 0
	// rotation stored in R whose columns are eigenvectors of A
	float d = (A[p][p] - A[q][q]) / (2.0f*A[p][q]);
	float t = 1.0f / (abs(d) + sqrt(d*d + 1.0f));

	if (d < 0.0f) t = -t;
	float c = 1.0f / sqrt(t*t + 1.0f);
	float s = t*c;
	A[p][p] += t*A[p][q];
	A[q][q] -= t*A[p][q];
	A[p][q] = A[q][p] = 0.0f;

	//transform A
	int k;
	for (k = 0; k < 3; k++){
		if (k != p&&k != q){
			float Akp = c*A[k][p] + s*A[k][q];
			float Akq = -s*A[k][p] + c*A[k][q];
			A[k][p] = A[p][k] = Akp;
			A[k][q] = A[q][k] = Akq;
		}
	}

	//store rotation in R
	for (k = 0; k < 3; k++){
		float Rkp = c*R[k][p] + s*R[k][q];
		float Rkq = -s*R[k][p] + c*R[k][q];

		R[k][p] = Rkp;
		R[k][q] = Rkq;
	}
}

void eigenDecompposition(mat3 &outA, mat3& outR){
	//only for symmetric matrices!
	//A=RA'R^T, where A' is diagnoal and R orthonormal

	//identity;
	mat3 A(outA);
	mat3 R = mat3(0.0f);
	R[0][0] = R[1][1] = R[2][2] = 1.0f;
	/*mat3 view = A;
	printf("view[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
		view[0][0], view[0][1], view[0][2],
		view[1][0], view[1][1], view[1][2],
		view[2][0], view[2][1], view[2][2]);*/
	int iter = 0;
	while (iter < JACOBI_ITERATIONS){
		int p, q;
		float a, maxval;
		maxval = -1.0f;
		for (int i = 0; i < 2; i++){
			for (int j = i+1; j < 3; j++){
				a = abs(A[i][j]);
				if (maxval<0.0f || a>maxval){
					p = i;
					q = j;
					maxval = a;
				}
			}
		}

		//all small enough->done
		if (maxval < 0.0001f) break;
		//rotate matrix with respect to that element
		jacobiRotate(A, R, p, q);
		/*printf("---------------------------------------------\n");
		view = A;
		printf("A[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
			view[0][0], view[0][1], view[0][2],
			view[1][0], view[1][1], view[1][2],
			view[2][0], view[2][1], view[2][2]);

		view = R;
		printf("R[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
			view[0][0], view[0][1], view[0][2],
			view[1][0], view[1][1], view[1][2],
			view[2][0], view[2][1], view[2][2]);*/
		iter++;
	}
	outA = A;
	outR = R;
}

void polarDecomposition(mat3 A, mat3 &R, mat3 &S){
	//A=RS, where S is symmetric and R is orthonormal
	//-> S=(A^T A)^(1/2)
	//mat3 A, R, S;
	//A = mat3(vec3(1.0f,-.3333f,.959f),vec3(.495f,1.0f,0.0f),vec3(.5f,-.247f,1.5f));


	//identity;
	R = mat3(0.0f);
	R[0][0] = R[1][1] = R[2][2] = 1.0f;

	mat3 ATA(0.0f);
	ATA = glm::transpose(A)*A;
	mat3 view = transpose(A);
	/*printf("AT[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
		view[0][0], view[0][1], view[0][2],
		view[1][0], view[1][1], view[1][2],
		view[2][0], view[2][1], view[2][2]);*/

	mat3 U;
	eigenDecompposition(ATA, U);
	view = U;
	/*printf("QT[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
		view[0][0], view[0][1], view[0][2],
		view[1][0], view[1][1], view[1][2],
		view[2][0], view[2][1], view[2][2]);*/
	float l0 = ATA[0][0];
	ATA[0][0] = l0 = l0 <= 0.0f ? 0.0f : sqrt(l0);

	float l1 = ATA[1][1];
	ATA[1][1]=l1 = l1 <= 0.0f ? 0.0f : sqrt(l1);

	float l2 = ATA[2][2];
	ATA[2][2]=l2 = l2 <= 0.0f ? 0.0f : sqrt(l2);
	view = ATA;
	/*printf("ATA[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
		view[0][0], view[0][1], view[0][2],
		view[1][0], view[1][1], view[1][2],
		view[2][0], view[2][1], view[2][2]);*/
	view = transpose(U)*ATA*U;
	/*printf("U[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
		view[0][0], view[0][1], view[0][2],
		view[1][0], view[1][1], view[1][2],
		view[2][0], view[2][1], view[2][2]);*/
	mat3 S1=inverse(view);
	/*S1[0][0] = l0*U[0][0] * U[0][0] + l1*U[0][1] * U[0][1] + l2*U[0][2] * U[0][2];
	S1[0][1] = l0*U[0][0] * U[1][0] + l1*U[0][1] * U[1][1] + l2*U[0][2] * U[1][2];
	S1[0][2] = l0*U[0][0] * U[2][0] + l1*U[0][1] * U[2][1] + l2*U[0][2] * U[2][2];

	S1[1][0] = S1[0][1];
	S1[1][1] = l0*U[1][0] * U[1][0] + l1*U[1][1] * U[1][1] + l2*U[1][2] * U[1][2];
	S1[1][2] = l0*U[1][0] * U[2][0] + l1*U[1][1] * U[2][1] + l2*U[1][2] * U[2][2];

	S1[2][0] = S1[0][2];
	S1[2][1] = S1[1][2];
	S1[2][2] = l0*U[2][0] * U[2][0] + l1*U[2][1] * U[2][1] + l2*U[2][2] * U[2][2];*/


	R = A*S1;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (abs(R[i][j] < 0.001f)) R[i][j] = 0.0f;
	R[0] = normalize(R[0]);
	R[1] = normalize(R[1]);
	R[2] = normalize(R[2]);

	view = R;
	/*printf("view[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
		view[0][0], view[0][1], view[0][2],
		view[1][0], view[1][1], view[1][2],
		view[2][0], view[2][1], view[2][2]);*/
	S = transpose(R)*A;
}

__global__ void SetGoalPosition(particle* particles, int N, mat3 R, vec4 MassCenter0, vec4 MassCenter1){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ParticleConditions(index,N,particles,0,RIGID_BODY)){
		particles[index].pred_position=vec4(R*(vec3(particles[index].position - MassCenter0) + vec3(MassCenter1)),1.0f);
	}
}

__global__ void MassCenterPredictedPosition(vec4* rigPredictedPos, particle* particles, int N, int startID){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ParticleConditions(index, N, particles, 0, RIGID_BODY)){
		rigPredictedPos[particles[index].ID - startID] = particles[index].pred_position;
	}
}

__global__ void MassCenterPredictedMatrix(mat3* rigPredictedRot, particle* particles, int N, int startID, vec4 oldMC, vec4 newMC){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ParticleConditions(index, N, particles, 0, RIGID_BODY)){
		vec3 p = vec3(particles[index].position - oldMC);
		vec3 q = vec3(particles[index].pred_position - newMC);
		rigPredictedRot[particles[index].ID - startID] = mat3(q.x*p, q.y*p, q.z*p);
	}
}
/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals
void initCuda(int N)
{
	
	numParticles = N;
	numGenerated = 0;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));
	//mat3 A, R, S;
	//polarDecomposition << <1, 1 >> >();
	LockNum = 0;
    cudaMalloc((void**)&particles, N * sizeof(particle));
	SmallObjMesh som(MeshFileName);
	LockNum+=som.position.size();
	
	printf("%d Vertices\n",LockNum);
	SmallObjMesh rigidtest("D:\\workspace\\PhysiAnim\\FinalProj\\Position_Based_Fluids\\PBF_Stage2\\Ref\\Models\\cont.obj");
	rigtest.start = LockNum;
	rigtest.size = rigidtest.position.size();
	LockNum += rigidtest.position.size();
	printf("%d Vertices\n", LockNum);
	particle* par=new particle[LockNum/*som.position.size()*/];
	for (int i = 0; i<som.position.size(); i++){
		par[i].position=vec4(som.position[i]+vec3(0.0,0.0,10.0),1.0);
		par[i].pred_position = par[i].position;
		par[i].LayerMask = CONTAINER;
	}

	if (rigidtest.position.size() > 0){
		rigtest.ID = 0;
		rigtest.size = rigidtest.position.size();
		rigtest.newMassCenter = rigtest.oldMassCenter = vec4(0.0);
	}
	vec4 testrig(0.0);
	for (int i = som.position.size(); i < som.position.size() + rigidtest.position.size(); i++){
		par[i].position = vec4(rigidtest.position[i - som.position.size()]+vec3(0,0,70.0f), 1.0f);
		par[i].pred_position = par[i].position;
		testrig += par[i].position;
		par[i].LayerMask = RIGID_BODY;
	}
	
	
	if (LockNum > N){
		printf("The mesh file need %d particles but the total particle number is set to %d!\n", LockNum, N);
		printf("Program down!\n");
		exit(-1);
	}
	if(LockNum>0){
		cudaMemcpy(particles,par,LockNum*sizeof(particle),cudaMemcpyHostToDevice);
	}
	
	//delete [] par;
	checkCUDAErrorWithLine("particles cudamalloc failed");

	cudaMalloc((void**)&neighbors, MAX_NEIGHBORS*N*sizeof(int));
	cudaMalloc((void**)&num_neighbors, N*sizeof(int));

	cudaMalloc((void**)&grid_idx, N*sizeof(int));
	checkCUDAErrorWithLine("grid idx cudamalloc failed!");
	cudaMalloc((void**)&grid, totalGridSize*sizeof(int));
	checkCUDAErrorWithLine("grid cudamalloc failed!");
	cudaMalloc((void**)&grid_elements, grid_width*grid_depth*grid_height*sizeof(GridElement));
	cudaMalloc((void**)&grid_lock, grid_width*grid_depth*grid_height*sizeof(int));
	cudaMemset(grid_lock, 0, grid_width*grid_depth*grid_height*sizeof(int));
	checkCUDAErrorWithLine("grid_elements cudamalloc failed!");
	cudaMalloc((void**)&rigPredictedPos, rigidtest.position.size()*sizeof(vec4));
	cudaMalloc((void**)&rigPredictedRot, rigidtest.position.size()*sizeof(mat3));
	initializeParticles<<<fullBlocksPerGrid, blockSize>>>(N, particles,LockNum);

	MassCenterPredictedPosition << <fullBlocksPerGrid, blockSize >> >(rigPredictedPos,particles, N, som.position.size());
	thrust::device_ptr<vec4> begin(rigPredictedPos);
	//begin+=som.position.size();
	thrust::device_ptr<vec4> end = begin + rigtest.size;
	//end += rigidtest.position.size();
	rigtest.oldMassCenter = thrust::reduce(begin, end,vec4(0.0), thrust::plus<vec4>())/rigtest.size;
	rigtest.oldMassCenter.w = 1.0f;
	printf("zhehuo: [%f, %f, %f]\n", rigtest.oldMassCenter.x, rigtest.oldMassCenter.y, rigtest.oldMassCenter.z);
	printf("tester: [%f, %f, %f]\n", testrig.x, testrig.y, testrig.z);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaPBFUpdateWrapper(float dt)
{
	
    dim3 fullBlocksPerGrid((int)ceil(float(numParticles)/float(blockSize)));

	int innerLockNum=0;
	if(cleanupFixedPoints&&innerLockNum!=LockNum){
		//cleanup<<<fullBlocksPerGrid, blockSize>>>(numParticles,particles,inn
	}
	innerLockNum=LockNum;
	/*if (ExtForceSet){
		ExtForceSet = false;
		setExternalForces << < fullBlocksPerGrid, blockSize >> >(numParticles, particles,innerLockNum,gravity);
	}*/
	//printf("Good\n");
	applyExternalForces << <fullBlocksPerGrid, blockSize >> >(numParticles, dt, particles, innerLockNum);
    checkCUDAErrorWithLine("applyExternalForces failed!");
	//findNeighbors(particles, grid_idx, grid, neighbors, numParticles,innerLockNum);
	findParticleNeighborsWrapper(particles, neighbors, num_neighbors, numParticles, innerLockNum);
    checkCUDAErrorWithLine("findNeighbors failed!");
	boxCollisionResponse << <fullBlocksPerGrid, blockSize >> >(numParticles, particles, innerLockNum);
	



	MassCenterPredictedPosition << <fullBlocksPerGrid, blockSize >> >(rigPredictedPos, particles, numParticles, rigtest.start);
	checkCUDAErrorWithLine("MassCenterPredictedPosition failed!");
	thrust::device_ptr<vec4> begin(rigPredictedPos);
	thrust::device_ptr<vec4> end = begin + rigtest.size;
	rigtest.newMassCenter = thrust::reduce(begin, end, vec4(0.0), thrust::plus<vec4>())/rigtest.size;
	rigtest.newMassCenter.w = 1.0f;
	//printf("mass center: [%f, %f, %f]\n", rigtest.newMassCenter.x, rigtest.newMassCenter.y, rigtest.newMassCenter.z);

	MassCenterPredictedMatrix << <fullBlocksPerGrid, blockSize >> >(rigPredictedRot, particles, numParticles, rigtest.start, rigtest.oldMassCenter, rigtest.newMassCenter);
	checkCUDAErrorWithLine("MassCenterPredictedMatrix failed!");
	thrust::device_ptr<mat3> rotBegin(rigPredictedRot);
	thrust::device_ptr<mat3> rotEnd = rotBegin + rigtest.size;
	
	rigRotMat = thrust::reduce(rotBegin, rotEnd, mat3(0.0), thrust::plus<mat3>());
	mat3 rotationDecomposition;
	mat3 scaleDecomposition;
	//mat3 view = rigRotMat;
	/*printf("view[3][3]=\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
	view[0][0], view[0][1], view[0][2],
	view[1][0], view[1][1], view[1][2],
	view[2][0], view[2][1], view[2][2]);*/
	polarDecomposition(rigRotMat, rotationDecomposition, scaleDecomposition);
	checkCUDAErrorWithLine("polarDecomposition failed!");
	SetGoalPosition << <fullBlocksPerGrid, blockSize >> >(particles, numParticles, rotationDecomposition, rigtest.oldMassCenter, rigtest.newMassCenter);
	checkCUDAErrorWithLine("SetGoalPosition failed!");

	MassCenterPredictedPosition << <fullBlocksPerGrid, blockSize >> >(rigPredictedPos, particles, numParticles, rigtest.start);
	checkCUDAErrorWithLine("MassCenterPredictedPosition failed!");
	begin=thrust::device_ptr<vec4>(rigPredictedPos);
	end = begin + rigtest.size;
	rigtest.oldMassCenter = thrust::reduce(begin, end, vec4(0.0), thrust::plus<vec4>()) / rigtest.size;
	rigtest.oldMassCenter.w = 1.0f;


	for(int i = 0; i < SOLVER_ITERATIONS; i++){
		calculateLambda<<<fullBlocksPerGrid, blockSize>>>(particles, neighbors, num_neighbors, numParticles,innerLockNum);
		calculateDeltaPi<<<fullBlocksPerGrid, blockSize>>>(particles, neighbors, num_neighbors, numParticles,innerLockNum);
		//PEFORM COLLISION DETECTION AND RESPONSE
		
		
		updatePredictedPosition<<<fullBlocksPerGrid, blockSize>>>(numParticles, particles,innerLockNum);
	}
	
	updateVelocity << <fullBlocksPerGrid, blockSize >> >(numParticles, particles, dt, innerLockNum);
	calculateCurl << <fullBlocksPerGrid, blockSize >> >(particles, neighbors, num_neighbors, numParticles, innerLockNum);
	applyVorticity << <fullBlocksPerGrid, blockSize >> >(particles, neighbors, num_neighbors, numParticles, innerLockNum);
	updatePosition<<<fullBlocksPerGrid, blockSize>>>(numParticles, particles,innerLockNum);
    checkCUDAErrorWithLine("updatePosition failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numParticles)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numParticles, particles, vbodptr, width, height, scene_scale,LockNum);
    cudaThreadSynchronize();
}

void freeCuda(){
	cudaFree(particles);
	cudaFree(neighbors);
	cudaFree(num_neighbors);
	cudaFree(grid_idx);
	cudaFree(grid);
}

