#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
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

using namespace glm;


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

__device__ bool ParticleConditions(int index, int N, int ID, int LockNum){
	return index < N&&ID >= LockNum;
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
	//if (length(r) > H) return 0.000001f;
	return max(315.0f / (64.0f * PI_FLOAT * POW9(H)) * CUBE(SQR(H) - dot(r, r)), .000001f);
}

__device__ glm::vec3 wGradientSpikyKernel(glm::vec3 p_i, glm::vec3 p_j){
	glm::vec3 r = p_i - p_j;
	float hr_term = H - glm::length(r);
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
// Clears grid from previous neighbors
__global__ void clearGrid(int* grid, int totalGridSize){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < totalGridSize){
		grid[index] = -1;
	}
}

// Matches each particles the grid index for the cell in which the particle resides
__global__ void findParticleGridIndex(particle* particles, int* grid_idx, int N){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < N){
		int x, y, z;
		glm::vec4 p = particles[index].pred_position;
		x = int(p.x) + BOX_X + 2;
		y = int(p.y) + BOX_Y + 2;
		z = int(p.z) + 2;
		grid_idx[index] = x + (2 * (BOX_X + 2) * y) + (4 * (BOX_X + 2) * (BOX_Y + 2) * z);
	}
}

// Matches the sorted index to each of the cells
__global__ void matchParticleToCell(int* gridIdx, int* grid, int N, int totalGridSize){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < N){
		if(index == 0){
			grid[gridIdx[index]] = index;
		}else if(gridIdx[index] != gridIdx[index - 1]){
			if(gridIdx[index] >= 0 && gridIdx[index] < totalGridSize) grid[gridIdx[index]] = index;
		}
	}
}

// Finds the nearest K neighbors within the smoothing kernel radius
__global__ void findKNearestNeighbors(particle* particles, int* gridIdx, int* grid, int* neighbors, int* num_neighbors, int N, int totalGridSize){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < N){
		int heap_size = 0;
		int x,y,z,idx;
		float r;
		glm::vec4 p_j, p = particles[index].pred_position;

		// Find particle index
		x = int(p.x) + BOX_X + 2;
		y = int(p.y) + BOX_Y + 2;
		z = int(p.z) + 2;

		float max;
		int m, max_index, begin, cell_position;

		// Examine all cells within radius
		// NOTE: checks the cube that circumscribes the spherical smoothing kernel
		for(int i = int(-H + z); i <= int(H + z); i++){
			for(int j = int(-H + y); j <= int(H + y); j++){
				for(int k = int(-H + x); k <= int(H + x); k++){
					idx = k + (2 * (BOX_X + 2) * j) + (4 * (BOX_X + 2) * (BOX_Y + 2) * i);

					if(idx >= totalGridSize || idx < 0){
						continue;
					}

					begin = grid[idx];

					if(begin < 0) continue;

					cell_position = begin;
					while(cell_position < N && gridIdx[begin] == gridIdx[cell_position]){
						if(cell_position == index){
							++cell_position;
							continue;
						}
						p_j = particles[cell_position].pred_position;
						r = glm::length(p - p_j);

						if(heap_size < MAX_NEIGHBORS){
							if(r < H){
								neighbors[index * MAX_NEIGHBORS + heap_size] = cell_position;
								++heap_size;
							}
						}else{
							max = glm::length(p - particles[neighbors[index * MAX_NEIGHBORS]].pred_position);
							max_index = 0;
							for(m = 1; m < heap_size; m++){
								float d = glm::length(p - particles[neighbors[index * MAX_NEIGHBORS + m]].pred_position); 
								if(d > max){
									max = d;
									max_index = m;
								}
							}

							if(r < max && r < H){
								neighbors[index * MAX_NEIGHBORS + max_index] = cell_position;
							}
						}

						++cell_position;
					}
				}
			}
		}
		num_neighbors[index] = heap_size;
	}
}

// Wrapper to find neighbors using hash grid
void findNeighbors(particle* particles, int* grid_idx, int* grid, int* neighbors, int N){
	dim3 fullBlocksPerGrid((int)ceil(float(totalGridSize) / float(blockSize)));
	dim3 fullBlocksPerGridParticles((int)ceil(float(N)/float(blockSize)));

	// Clear Grid
	clearGrid<<<fullBlocksPerGrid, blockSize>>>(grid, totalGridSize);
	checkCUDAErrorWithLine("clearGrid failed!");

	// Match particle to index
	findParticleGridIndex<<<fullBlocksPerGridParticles, blockSize>>>(particles, grid_idx, N);
	checkCUDAErrorWithLine("findParticleGridIndex failed!");

	// Cast to device pointers
	thrust::device_ptr<int> t_grid_idx = thrust::device_pointer_cast(grid_idx);
	thrust::device_ptr<particle> t_particles = thrust::device_pointer_cast(particles);

	// Sort by key
	thrust::sort_by_key(t_grid_idx, t_grid_idx + N, t_particles);
	checkCUDAErrorWithLine("thrust failed!");

	// Match sorted particle index
	matchParticleToCell<<<fullBlocksPerGridParticles, blockSize>>>(grid_idx, grid, N, totalGridSize);
	checkCUDAErrorWithLine("matchParticletoCell failed!");

	// Find K nearest neighbors
	findKNearestNeighbors<<<fullBlocksPerGridParticles, blockSize>>>(particles, grid_idx, grid, neighbors, num_neighbors, N, totalGridSize);
	checkCUDAErrorWithLine("findKNearestNeighbors failed!");
}


/*************************************
 * Kernels for Jacobi Solver
 *************************************/

__global__ void calculateLambda(particle* particles, int* neighbors, int* num_neighbors, int N){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N){
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

__global__ void calculateDeltaPi(particle* particles, int* neighbors, int* num_neighbors, int N){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N){
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
	if (ParticleConditions(index, N, particles[index].ID, LockNum)){
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
	if(ParticleConditions(index,N,particles[index].ID,LockNum)){
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
		p.position.x = (index%20)-9.5f;
		p.position.y = ((index/20)%20)-9.5f;
		p.position.z = (index/400)+30.0f+0.05f*rand.z;
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
		p.velocity=glm::vec3(0.0f);
		p.external_forces=glm::vec3(0.0f,0.0f,gravity);
		particles[index]=p;
	}
}

__global__ void setExternalForces(int N, particle* particles, int LockNum,vec3 extForce){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (Conditions(index, N, LockNum)){
		particles[index].external_forces = extForce;
	}
}

//Simple Euler integration scheme
__global__ void applyExternalForces(int N, float dt, particle* particles,int LockNum)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (Conditions(index, N, LockNum)){
		particle p = particles[index];
		//p.velocity += dt * p.external_forces;
		//p.pred_position = p.position + dt * glm::vec4(p.velocity,0.0f);

		p.velocity+=dt*p.external_forces;
		//Particles[i]->PredictedPos=Particles[i]->position+dt*Particles[i]->velocity;
		p.delta_pos=glm::vec3(0.0f);
		p.pred_position=p.position+vec4(p.velocity*dt,0.0);
		



		particles[index] = p;
	}
}

__global__ void updatePosition(int N, particle* particles,int LockNum=0)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < N&&particles[index].ID >= LockNum){
		particles[index].position = particles[index].pred_position;
	}
	if(particles[index].ID<=LockNum){
		particles[index].velocity=vec3(0.0f);
		particles[index].curl=vec3(0.0f);
		//particles[index].external_forces=vec3(0.0f);
	}
}

__global__ void updatePredictedPosition(int N, particle* particles,int LockNum=0)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if(index < N&&particles[index].ID>=LockNum){
		particles[index].pred_position += glm::vec4(particles[index].delta_pos,0.0f);
	}
}

__global__ void updateVelocity(int N, particle* particles, float dt,int LockNum)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (Conditions(index, N, LockNum)){
		particles[index].velocity = glm::vec3((1.0f/dt)*(particles[index].pred_position - particles[index].position));
	}
}

__global__ void boxCollisionResponse(int N, particle* particles, float move,int LockNum){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (Conditions(index, N, LockNum)){
		if( particles[index].pred_position.z < 0.0f){
			particles[index].pred_position.z = 0.0001f;
			glm::vec3 normal = glm::vec3(0,0,1);
			glm::vec3 reflectedDir = particles[index].velocity - glm::vec3(2.0f*(normal*(glm::dot(particles[index].velocity,normal))));
			particles[index].velocity.z = reflectedDir.z;
		}
		if( particles[index].pred_position.z > BOX_Z){
			particles[index].pred_position.z = BOX_Z-0.0001f;
			glm::vec3 normal = glm::vec3(0,0,-1);
			glm::vec3 reflectedDir = particles[index].velocity - glm::vec3(2.0f*(normal*(glm::dot(particles[index].velocity,normal))));
			particles[index].velocity.z = reflectedDir.z;
		}
		if( particles[index].pred_position.y < -BOX_Y+move){
			particles[index].pred_position.y = -BOX_Y+move+0.01f;
			glm::vec3 normal = glm::vec3(0,1,0);
			glm::vec3 reflectedDir = particles[index].velocity - glm::vec3(2.0f*(normal*(glm::dot(particles[index].velocity,normal))));
			particles[index].velocity.y = reflectedDir.y;
		}
		if( particles[index].pred_position.y > BOX_Y){
			particles[index].pred_position.y = BOX_Y-0.01f;
			glm::vec3 normal = glm::vec3(0,-1,0);
			glm::vec3 reflectedDir = particles[index].velocity - glm::vec3(2.0f*(normal*(glm::dot(particles[index].velocity,normal))));
			particles[index].velocity.y = reflectedDir.y;
		}
		if( particles[index].pred_position.x < -BOX_X){
			particles[index].pred_position.x = -BOX_X+0.01f;
			glm::vec3 normal = glm::vec3(1,0,0);
			glm::vec3 reflectedDir = particles[index].velocity - glm::vec3(2.0f*(normal*(glm::dot(particles[index].velocity,normal))));
			particles[index].velocity.x = reflectedDir.x;
		}
		if( particles[index].pred_position.x > BOX_X){
			particles[index].pred_position.x = BOX_X-0.01f;
			glm::vec3 normal = glm::vec3(-1,0,0);
			glm::vec3 reflectedDir = particles[index].velocity - glm::vec3(2.0f*(normal*(glm::dot(particles[index].velocity,normal))));
			particles[index].velocity.x = reflectedDir.x;
		}
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

    cudaMalloc((void**)&particles, N * sizeof(particle));
	SmallObjMesh som(MeshFileName);
	LockNum=som.position.size();
	printf("%d Vertices",LockNum);
	particle* par=new particle[som.position.size()];
	for(int i=0;i<LockNum;i++){
		par[i].position=vec4(som.position[i],1.0);
		par[i].pred_position = par[i].position;
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


	initializeParticles<<<fullBlocksPerGrid, blockSize>>>(N, particles,LockNum);

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
	applyExternalForces << <fullBlocksPerGrid, blockSize >> >(numParticles, dt, particles, innerLockNum);
    checkCUDAErrorWithLine("applyExternalForces failed!");
	findNeighbors(particles, grid_idx, grid, neighbors, numParticles);

    checkCUDAErrorWithLine("findNeighbors failed!");



	for(int i = 0; i < SOLVER_ITERATIONS; i++){
		calculateLambda<<<fullBlocksPerGrid, blockSize>>>(particles, neighbors, num_neighbors, numParticles);
		calculateDeltaPi<<<fullBlocksPerGrid, blockSize>>>(particles, neighbors, num_neighbors, numParticles);
		//PEFORM COLLISION DETECTION AND RESPONSE
		boxCollisionResponse << <fullBlocksPerGrid, blockSize >> >(numParticles, particles, wallMove, innerLockNum);
		
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

