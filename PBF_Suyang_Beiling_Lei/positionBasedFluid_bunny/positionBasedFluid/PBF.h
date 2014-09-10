#include <iostream>
#include <math.h>
#include <list>
#include<stdio.h>
#include <unordered_set>
#include <vector>

#include "vec.h"
#include "matrix.h"
#include "objLoader.h"
#include "Constant.h"

using namespace std;

//extern vec3 gravity;

#ifndef _PBF_H_
#define _PBF_H_

struct Particle {

	int id;
	float inv_mass;
	float density;
	float invRestDensity;
	vec3 position;
	vec3 velocity;
	vec3 force;
	bool isOnSurface;

	vec3 accum_vorticity;
	vec3 grad_vorticity;
	//vec3 color_gradient;
	//float color_laplacian;

	vec3 viscosity_force;
	//vec3 pressure_force;

	vec3 PredictedPos;

	float lambda;
	float C;
	float grad_C;
	vec3 accum_Grad_C;

	vec3 deltaP;

	unordered_set<Particle*> myNeighbor;

	Particle() { 
		inv_mass = 1.0f;
		density=0.0f;
		lambda=0.0f;
		C=0.0f;
		grad_C=0.0f;
		isOnSurface=false;
		invRestDensity=1/REST_DENSITY;
	}
};

struct GridElement {
	list<Particle*> particles;
};

struct FluidMaterial {

	const float gas_constant;
	const float mu;
	const float rest_density;
	const float sigma;
	const float point_damping;

	FluidMaterial()
		:	gas_constant(0),
			mu(0),
			rest_density(0),
			sigma(0),	
			point_damping(0){}

	FluidMaterial(
			float gas_constant,
			float mu,
			float rest_density,
			float sigma,
			float point_damping)
	  : gas_constant(gas_constant),
	    mu(mu),
	    rest_density(rest_density),
	    sigma(sigma),
	    point_damping(point_damping) {}
};

class PbfSolver{
	const int grid_width;
	const int grid_height;
	const int grid_depth;

	const float core_radius;
	const float timestep;

	const FluidMaterial material;

	unsigned int ParticleCont;
	Particle** Particles;

	GridElement *grid_elements;
	GridElement *sleeping_grid_elements;

	ModeFlag mf;

	float kernel(const vec3 &r, const float h);

	float kernel_viscosity(const vec3 &r,const float h);

	vec3 gradient_kernel(const vec3 &r, const float h);

	vec3 spikyGradient(const vec3 &r, const float h);

	void insert_into_grid(int i, int j, int k);

	void update_grid();

	GridElement &grid(int i, int j, int k);

	void findNeighbor();

public:
	Mesh objContainer;
	bool dropToGround;

public:
	PbfSolver(): 
		grid_width(0),
		grid_height(0),
		grid_depth(0),
		core_radius(0),
		timestep(0),
		material(FluidMaterial()),
		ParticleCont(0),
		objContainer(""){}
		
	PbfSolver(
		float domain_width,
		float domain_height,
		float domain_depth,
		float core_radius,
		float timestep,
		FluidMaterial material,
		int PaCont)
			: 
		grid_width((int) (domain_width / core_radius) + 1),
		grid_height((int) (domain_height / core_radius) + 1),
		grid_depth((int) (domain_depth / core_radius) + 1),
		core_radius(core_radius),
		timestep(timestep),
		material(material),
		ParticleCont(PaCont),
		objContainer(ContainerFileName){
			dropToGround=false;
		}

	void init_particles(Particle* p, unsigned int ParCont=0);

	void addExtForce(double dt,vec3 extF=vec3(0,100.0,0));

	void solveLambda();

	bool solveCollision(Particle&);

	void solvePosition();

	void solveConstraints(unsigned int MaxIter);

	void apply_vorticity(double dt);

	void apply_viscosity();

	void final_update(double dt);

	void update(double );

	inline void add_to_grid(GridElement *target_grid, Particle &particle);

	inline int grid_index(const int&, const int&, const int&);

	void update_vorticity(const int&, const int&, const int&);

	void sum_all_vorticity(const int&, const int&, const int&, Particle & );

	void sum_vorticity(GridElement &grid_element, Particle &particle);

	void add_vorticity(Particle &particle, Particle &neighbour);

	void drop();

	template <typename Function>
	void foreach_particle(Function function) {
		for (int k = 0; k < grid_depth; k++) {
			for (int j = 0; j < grid_height; j++) {
				for (int i = 0; i < grid_width; i++) {
					GridElement &grid_element = grid_elements[grid_width * (k * grid_height + j) + i];

					list<Particle*> &plist = grid_element.particles;
					for (list<Particle*>::iterator piter = plist.begin(); piter != plist.end(); piter++) {
						function(**piter);
					}
				}
			}
		}
	}

};
#endif
