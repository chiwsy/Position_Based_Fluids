#include "PBF.h"

//#define _disp_nbor
extern vec3 gravity;

#define PI_FLOAT				3.141592653589793f
#define DELTA_Q				(float)(0.1*core_radius)

#define SQR(x)					((x) * (x))
#define CUBE(x)					((x) * (x) * (x))
#define POW6(x)					(CUBE(x) * CUBE(x))
#define POW9(x)					(POW6(x) * CUBE(x))


inline float PbfSolver::kernel(const vec3 &r, const float h) {
	if(r.Length()>h) return 0.00001f;
	return 315.0f / (64.0f * PI_FLOAT * POW9(h)) * CUBE(SQR(h) - Dot(r,r));
}

inline vec3 PbfSolver::gradient_kernel(const vec3 &r, const float h) {
	if(r.Length()>h)
		return vec3(0.00001,0.00001,0.00001);
	return -945.0f / (32.0f * PI_FLOAT * POW9(h)) * SQR(SQR(h) - Dot(r, r)) * r;
}

inline float PbfSolver::kernel_viscosity(const vec3& r, const float h){
	if(r.Length()>h) return 0.00001f;
	return 15.0f/(2*PI_FLOAT*CUBE(h))*(-CUBE(r.Length()/h)/2+SQR(r.Length()/h)+h/2/r.Length()-1.0f);
}

inline vec3 PbfSolver::spikyGradient(const vec3 &r, const float h){
	if(r.Length()>h)
		return vec3(0.00001,0.00001,0.00001);
	return 45.0f / (PI_FLOAT * pow(h,6)) * pow(h - r.Length(),2) *r/(r.Length()+0.0001);
}

void PbfSolver::init_particles(Particle* p, unsigned int ParCont){
	ParticleCont=ParCont;
	Particles=new Particle*[ParCont];
	grid_elements = new GridElement[grid_width * grid_height * grid_depth];
	sleeping_grid_elements = new GridElement[grid_width * grid_height * grid_depth];

	for (int x = 0; x < ParCont; x++) {
		p[x].id = x;
		Particles[x]=&p[x];
		add_to_grid(grid_elements, p[x]);
	}
}

inline void PbfSolver::insert_into_grid(int i, int j, int k) {
	GridElement &grid_element = grid(i, j, k);

	list<Particle*> &plist = grid_element.particles;
	for (list<Particle*>::iterator piter = plist.begin(); piter != plist.end(); piter++) {
		add_to_grid(sleeping_grid_elements, **piter);
	}
}

void PbfSolver::update_grid() {
	for (int k = 0; k < grid_depth; k++) {
		for (int j = 0; j < grid_height; j++) {
			for (int i = 0; i < grid_width; i++) {
				insert_into_grid(i, j, k);
				grid(i, j, k).particles.clear();
			}
		}
	}

	/* Swap the grids. */
	swap(grid_elements, sleeping_grid_elements);
}

void PbfSolver::addExtForce(double dt, vec3 extF){
	for(unsigned int i=0;i<ParticleCont;i++){
		if(!dropToGround&&Particles[i]->isOnSurface)
			continue;
		//if(Particles[i]->position[1]<gravity.Length()*dt*dt) continue;
		Particles[i]->velocity+=dt*(Particles[i]->force+extF)*Particles[i]->inv_mass;
		//Particles[i]->PredictedPos=Particles[i]->position+dt*Particles[i]->velocity;
		Particles[i]->deltaP=vec3();
		Particles[i]->PredictedPos=Particles[i]->position+Particles[i]->velocity*dt;
		//solveCollision((*Particles[i]));
		Particles[i]->force=vec3();
	}
	
	/*for(unsigned int i=0;i<ParticleCont;i++){
		Particles[i]->PredictedPos+=Particles[i]->deltaP;
	}*/
}

void PbfSolver::solveLambda(){
	/////////////////Solve C_i///////////////////////////////////////////////
	for(unsigned int i=0;i<ParticleCont;i++){
		//Particles[i]->density=0.0f;
		vec3 p(Particles[i]->PredictedPos);
		float ro=0.0f;
		for(unordered_set<Particle*>::iterator pit=Particles[i]->myNeighbor.begin();
			pit!=Particles[i]->myNeighbor.end();
			pit++){
				if(Particles[i]->id>(*pit)->id) continue;
				vec3 p_j((*pit)->PredictedPos);
				double kv=kernel(p-p_j,core_radius);
				//ro+=kv/(*pit)->inv_mass;
				Particles[i]->density+=kv/(*pit)->inv_mass;
				(*pit)->density+=kv/Particles[i]->inv_mass;
		}
		//Particles[i]->density=ro+kernel(vec3(),core_radius)/Particles[i]->inv_mass;
		Particles[i]->density+=kernel(vec3(),core_radius)/Particles[i]->inv_mass;
	}

	for(unsigned int i=0;i<ParticleCont;i++){
		Particles[i]->C=Particles[i]->density*Particles[i]->invRestDensity-1.0f;
		Particles[i]->density=0.0f;
	}
	/////////////////////////////////////////////////////////////////////////

	////////////////////////Solve C_i_gradient///////////////////////////////
	
	for(unsigned int i=0;i<ParticleCont;i++){
		//Particles[i]->density=0.0f;
		vec3 p(Particles[i]->PredictedPos);
		float C_i_gradient, sum_gradients=0.0f;
		//Gradient:
		for(unordered_set<Particle*>::iterator pit=Particles[i]->myNeighbor.begin();
			pit!=Particles[i]->myNeighbor.end();
			pit++){
				
				if(Particles[i]->id>(*pit)->id) continue;
				vec3 p_j((*pit)->PredictedPos);
				C_i_gradient=pow(spikyGradient(p-p_j,core_radius).Length()*Particles[i]->invRestDensity,2);
				//sum_gradients+=C_i_gradient;
				
				Particles[i]->grad_C+=C_i_gradient;
				(*pit)->grad_C+=C_i_gradient;
			}

		//Anti-Gradient:
		//vec3 accum;
		for(unordered_set<Particle*>::iterator pit=Particles[i]->myNeighbor.begin();
			pit!=Particles[i]->myNeighbor.end();
			pit++){
				
				if(Particles[i]->id>(*pit)->id) continue;
				vec3 p_j((*pit)->PredictedPos);
				//accum+=spikyGradient(p-p_j,core_radius);
				vec3 agc=spikyGradient(Particles[i]->PredictedPos-(*pit)->PredictedPos,core_radius)*Particles[i]->invRestDensity;
				Particles[i]->accum_Grad_C+=agc;
				(*pit)->accum_Grad_C-=agc;

		}
		//C_i_gradient=1.0f/material.rest_density*accum.Length();
		C_i_gradient=Particles[i]->accum_Grad_C.Length()*Particles[i]->invRestDensity;
		Particles[i]->grad_C+=C_i_gradient*C_i_gradient;

		float sum_Ci=Particles[i]->grad_C+RELAXATION;
		Particles[i]->lambda=-1.0f*(Particles[i]->C/sum_Ci);
		Particles[i]->grad_C=0.0;
		Particles[i]->accum_Grad_C=vec3();
	}

	

	/*for(unsigned int i=0;i<ParticleCont;i++){
		Particles[i]->grad_C+=pow(Particles[i]->accum_Grad_C.Length(),2);
		Particles[i]->lambda=-Particles[i]->C/(Particles[i]->grad_C+RELAXATION);
		Particles[i]->grad_C=0.0;
		Particles[i]->accum_Grad_C=vec3(0,0,0);
	}*/
}

bool PbfSolver::solveCollision(Particle& particle){
	double &px = particle.PredictedPos[0];
	double &py = particle.PredictedPos[1];
	double &pz = particle.PredictedPos[2];

	double &vx = particle.velocity[0];
	double &vy = particle.velocity[1];
	double &vz = particle.velocity[2];

	    if( pz <= 0.0f){
			pz = jt;
			vec3 normal = vec3(0,0,1);
			vec3 reflectedDir = particle.velocity - vec3(2.0f*(normal*(Dot(particle.velocity,normal))));
			particle.velocity[2] = reflectedDir[2]*collision_restitution;
			return true;
		}
		if( pz >= DEPTH-1){
			pz = DEPTH-1-jt;
			vec3 normal = vec3(0,0,-1);
			vec3 reflectedDir = particle.velocity - vec3(2.0f*(normal*(Dot(particle.velocity,normal))));
			particle.velocity[2] = reflectedDir[2]*collision_restitution;
			return true;
		}
		if( py <= 0){
			py = jt;
			vec3 normal = vec3(0,1,0);
			vec3 reflectedDir = particle.velocity - vec3(2.0f*(normal*(Dot(particle.velocity,normal))));
			particle.velocity[1] = reflectedDir[1]*collision_restitution;
			return true;
		}
		if( py >= HEIGHT-1){
			py = HEIGHT-1-jt;
			vec3 normal = vec3(0,-1,0);
			vec3 reflectedDir = particle.velocity - vec3(2.0f*(normal*(Dot(particle.velocity,normal))));
			particle.velocity[1] = reflectedDir[1]*collision_restitution;
			return true;
		}
		if( px <= 0){
			px = jt;
			vec3 normal = vec3(1,0,0);
			vec3 reflectedDir = particle.velocity - vec3(2.0f*(normal*(Dot(particle.velocity,normal))));
			particle.velocity[0] = reflectedDir[0]*collision_restitution;
			return true;
		}
		if( px >= WIDTH-1){
			px = WIDTH-1-jt;
			vec3 normal = vec3(-1,0,0);
			vec3 reflectedDir = particle.velocity - vec3(2.0f*(normal*(Dot(particle.velocity,normal))));
			particle.velocity[0] = reflectedDir[0]*collision_restitution;
			return true;
		}

		return false;

}

void PbfSolver::solvePosition(){
	for(unsigned int i=0;i<ParticleCont;i++){
		//Particles[i]->density=0.0f;
		vec3 p(Particles[i]->PredictedPos);
		float l=Particles[i]->lambda;
		vec3 delta;
		
		float k_term;
		vec3 d_q= DELTA_Q*vec3(1.0,1.0,1.0)+p;

		float s_corr=0.0f;

		for(unordered_set<Particle*>::iterator pit=Particles[i]->myNeighbor.begin();
			pit!=Particles[i]->myNeighbor.end();
			pit++){
				
				if(Particles[i]->id>(*pit)->id) continue;
				vec3 p_j((*pit)->PredictedPos);
				float poly6pd_q=kernel(p-d_q,core_radius);

				if(poly6pd_q<EPSILON) k_term=0.0f;
				else k_term= kernel(p-p_j,core_radius)/poly6pd_q;
				s_corr = -1.0f * PRESSURE_K * pow(k_term, PRESSURE_N);
				//s_corr=0.0f;
				//delta+=(l+(*pit)->lambda+s_corr)*spikyGradient(p-p_j,core_radius);
				vec3 dp=(Particles[i]->lambda+(*pit)->lambda+s_corr)* spikyGradient(Particles[i]->PredictedPos-(*pit)->PredictedPos,core_radius)*Particles[i]->invRestDensity;
				Particles[i]->deltaP+=dp;
				(*pit)->deltaP-=dp;

		}
		//Particles[i]->deltaP+=2*Particles[i]->lambda*spikyGradient(vec3(),core_radius)/material.rest_density;
		//Particles[i]->deltaP=1.0f/material.rest_density*delta;
		
	}

	for(int i=0;i<ParticleCont;i++){
		if(Particles[i]->isOnSurface)
			continue;
		Particles[i]->PredictedPos+=Particles[i]->deltaP;
		Particles[i]->deltaP=vec3();
		solveCollision((*Particles[i]));
	}

	/*for(int i=0;i<ParticleCont;i++){
		Particles[i]->PredictedPos+=Particles[i]->deltaP;
	}*/
}

void PbfSolver::solveConstraints(unsigned int MaxIter){
	for(unsigned int iter=0;iter<MaxIter;iter++){
		solveLambda();
		solvePosition();
	}
}


void PbfSolver::apply_viscosity(){
	/*mf=VISCOSITY;
	for (int k = 0; k < grid_depth; k++) {
		for (int j = 0; j < grid_height; j++) {
			for (int i = 0; i < grid_width; i++) {
				update_vorticity(i, j, k);
			}
		}
	}*/

	for(int i=0;i<ParticleCont;i++){
		for(unordered_set<Particle*>::iterator pit=Particles[i]->myNeighbor.begin();
			pit!=Particles[i]->myNeighbor.end();
			pit++){
				if(Particles[i]->id>(*pit)->id) continue;
				//vec3 omega=((*pit)->velocity-Particles[i]->velocity).Cross(spikyGradient(Particles[i]->PredictedPos-(*pit)->PredictedPos,core_radius));
				vec3 v_accu=((*pit)->velocity-Particles[i]->velocity) * kernel_viscosity(Particles[i]->PredictedPos-(*pit)->PredictedPos,core_radius);
				Particles[i]->viscosity_force+=v_accu;
				(*pit)->viscosity_force-=v_accu;

		}
	}
	for(int i=0;i<ParticleCont;i++){
		Particles[i]->velocity+=c_viscocity*Particles[i]->viscosity_force;
		Particles[i]->viscosity_force=vec3();
	}
}

void PbfSolver::apply_vorticity(double dt){
	//curl
	for(int i=0;i<ParticleCont;i++){
		/*vec3 p = Particles[i]->PredictedPos;
		vec3 v = Particles[i]->velocity;

		int j_idx;
		vec3 v_ij, gradient, accum;*/

		for(unordered_set<Particle*>::iterator pit=Particles[i]->myNeighbor.begin();
			pit!=Particles[i]->myNeighbor.end();
			pit++){
				if(Particles[i]->id>(*pit)->id) continue;
				vec3 omega=((*pit)->velocity-Particles[i]->velocity).Cross(spikyGradient(Particles[i]->PredictedPos-(*pit)->PredictedPos,core_radius));
				Particles[i]->accum_vorticity+=omega;
				(*pit)->accum_vorticity+=omega;
				/*v_ij = (*pit)->velocity - v;
				gradient = spikyGradient(p-(*pit)->PredictedPos,core_radius);
				accum += v_ij.Cross( gradient);*/

		}
		//Particles[i]->accum_vorticity=accum;
	}

	for(int i=0;i<ParticleCont;i++){
		//vec3 p(Particles[i]->PredictedPos);
		//vec3 w(Particles[i]->accum_vorticity);

		////int j_idx;
		//float mag_w;
		//vec3 r, grad;
		for(unordered_set<Particle*>::iterator pit=Particles[i]->myNeighbor.begin();
			pit!=Particles[i]->myNeighbor.end();
			pit++){
				if(Particles[i]->id>(*pit)->id) continue;
				vec3 r=(*pit)->PredictedPos-Particles[i]->PredictedPos;
				double mag=((*pit)->accum_vorticity-Particles[i]->accum_vorticity).Length();
				vec3 grad=vec3(mag/r[0],mag/r[1],mag/r[2]);
				Particles[i]->grad_vorticity+=grad;
				(*pit)->grad_vorticity-=grad;
				/*r =(*pit)->PredictedPos - p;
				mag_w = ((*pit)->accum_vorticity - w).Length();
				grad[0] += mag_w / r[0];
				grad[1] += mag_w / r[1];
				grad[2] += mag_w / r[2];*/
		}
		/*vec3 vorticity, N;
		N = 1.0f/(grad.Length() + .001f) * grad;
		vorticity = float(RELAXATION) * (N.Cross( w));*/
		//particles[index].external_forces += vorticity;
		//Particles[i]->force+=vorticity;
	}
	
	
	for(int i=0;i<ParticleCont;i++){
		vec3 Ndir=Particles[i]->grad_vorticity.Normalize();
		/*Particles[i]->force=epsilon*(Ndir.Cross(Particles[i]->accum_vorticity));
		Particles[i]->velocity+=Particles[i]->force*Particles[i]->inv_mass*dt;*/
		Particles[i]->accum_vorticity=vec3();
		Particles[i]->grad_vorticity=vec3();
	}


}

void PbfSolver::update_vorticity(const int& i, const int& j,const int& k) {
	GridElement &grid_element = grid(i, j, k);
	list<Particle*>&plist = grid_element.particles;
	for (list<Particle*>::iterator piter = plist.begin(); piter != plist.end(); piter++) {
		sum_all_vorticity(i, j, k, **piter);
	}
}

void PbfSolver::sum_all_vorticity(const int& i, const int& j, const int& k, Particle &particle) {
	for (int z = k - 1; z <= k + 1; z++) {
		for (int y = j - 1; y <= j + 1; y++) {
			for (int x = i - 1; x <= i + 1; x++) {
				if (   (x < 0) || (x >= grid_width)
					|| (y < 0) || (y >= grid_height)
					|| (z < 0) || (z >= grid_depth)) {
					continue;
				}

				sum_vorticity(grid(x, y, z), particle);
			}
		}
	}
}

void PbfSolver::sum_vorticity(GridElement &grid_element, Particle &particle) {
	list<Particle*>  &plist = grid_element.particles;
	for (list<Particle*>::iterator piter = plist.begin(); piter != plist.end(); piter++) {
		add_vorticity(particle, **piter);
	}
}

inline void PbfSolver::add_vorticity(Particle &particle, Particle &neighbour) {
	if (particle.id >= neighbour.id) {
		return;
	}

	vec3 r = particle.position - neighbour.position;
	if (Dot(r, r) > SQR(core_radius)) {
		return;
	}

	switch(mf){
	case ModeFlag::VORTICITY:
		{
			//accumulative omiga, if the Spiky is not odd function, this is wrong!!!!!!
			vec3 omega=(neighbour.velocity-particle.velocity).Cross(spikyGradient(particle.PredictedPos-neighbour.PredictedPos,core_radius));
			particle.accum_vorticity+=omega;
			neighbour.accum_vorticity+=omega;
		}
		break;
	case ModeFlag::ETA:
		{
			vec3 r=neighbour.PredictedPos-particle.PredictedPos;
			double mag=(neighbour.accum_vorticity-particle.accum_vorticity).Length();
			vec3 grad=vec3(mag/r[0],mag/r[1],mag/r[2]);
			particle.grad_vorticity+=grad;
			neighbour.grad_vorticity-=grad;
		}
		break;
	case ModeFlag::VISCOSITY:
		{
			vec3 v_accu=(neighbour.velocity-particle.velocity) * kernel_viscosity(particle.PredictedPos-neighbour.PredictedPos,core_radius);
			particle.viscosity_force+=v_accu;
			neighbour.viscosity_force-=v_accu;
		}
		break;
	case ModeFlag::RHO:
		{
			double kernelValue=kernel(particle.PredictedPos-neighbour.PredictedPos,core_radius);
			particle.density+=kernelValue/neighbour.inv_mass;
			neighbour.density+=kernelValue/particle.inv_mass;
		}
		break;
	case ModeFlag::GRADIENT:
		{
			particle.grad_C+=pow(spikyGradient(particle.PredictedPos-neighbour.PredictedPos,core_radius).Length()*particle.invRestDensity,2);
		}
		break;
	case ModeFlag::GRADIENTANTI:
		{
			particle.accum_Grad_C+=spikyGradient(particle.PredictedPos-neighbour.PredictedPos,core_radius)*particle.invRestDensity;
		}
		break;
	case ModeFlag::DELTAP:
		{
			particle.deltaP+=(particle.lambda+neighbour.lambda)*spikyGradient(particle.PredictedPos-neighbour.PredictedPos,core_radius)*particle.invRestDensity;
		}
		break;
	default:
		cout<<"Unknown Mode detected!"<<endl;
		exit(mf);
		break;
	}
}

void PbfSolver::final_update(double dt){
	for(unsigned int i=0;i<ParticleCont;i++){
		if(Particles[i]->isOnSurface)
			continue;
		Particles[i]->velocity=(Particles[i]->PredictedPos-Particles[i]->position)/dt;
	}

	//apply_vorticity(dt);
	//apply_viscosity();
	for(unsigned int i=0;i<ParticleCont;i++){
		if(!dropToGround&&Particles[i]->isOnSurface)
			continue;
		Particles[i]->position=Particles[i]->PredictedPos;
	}
}

void PbfSolver::drop(){
	for(unsigned int i=0;i<ParticleCont;i++){
		Particles[i]->isOnSurface=false;
		Particles[i]->invRestDensity=1/REST_DENSITY;
	}
}

void PbfSolver::update(double dt){
	if(dropToGround){
		for(unsigned int i=0;i<ParticleCont;i++){			
			if(Particles[i]->isOnSurface){
				if(solveCollision((*Particles[i]))){
					Particles[i]->isOnSurface=false;
					Particles[i]->invRestDensity=1/REST_DENSITY;
				}
			}
		}
		//dropToGround=false;
	}

	for(unsigned int i=0;i<ParticleCont;i++){
		//Particles[i]->density=0.0;
		Particles[i]->force=vec3(0.0,0.0,0.0);
		//Particles[i]->accum_vorticity=vec3(0,0,0);
	}
	//Apply external force
	//predicted_position+=delta_t*v_i;
	addExtForce(dt,gravity);
    
	findNeighbor();
#ifdef _disp_nbor
	for(int i=0;i<ParticleCont;i++){
		printf("Particle[%d]=[%f,%f,%f]:\n",i,Particles[i]->PredictedPos[0],Particles[i]->PredictedPos[1],Particles[i]->PredictedPos[2]);
		for(unordered_set<Particle*>::iterator niter=Particles[i]->myNeighbor.begin();
			niter!=Particles[i]->myNeighbor.end();
			++niter){
				printf("%d\tNeighbor Particle=[%f,%f,%f]\n",(*niter)->id,(*niter)->PredictedPos[0],(*niter)->PredictedPos[1],(*niter)->PredictedPos[2]);
		}
	
	}
	cout<<"*****************************************************************************"<<endl;
#endif
	//solve lambda_i
	//solve delta_P_i & collision detection and handling
	//update predicted_position+=delta_P_i
	solveConstraints(MaxIteration);

	//update velocity
	//apply vorticity confinement and XSPH viscosity
	//update position:= predicted_position
	final_update(dt);
	//printf("Particles[0]=[%f,%f,%f]\n",Particles[0]->position[0],Particles[0]->position[1],Particles[0]->position[2]);
}

inline void PbfSolver::add_to_grid(GridElement *target_grid, Particle &particle) {
	int i = (int) (particle.PredictedPos[0] / core_radius);
	if(i<0) i=0;
	else if(i> grid_width-1) i=grid_width-1;
	int j = (int) (particle.PredictedPos[1] / core_radius);
	if(j<0) j=0;
	else if(j> grid_height-1) j=grid_height-1;
	int k = (int) (particle.PredictedPos[2] / core_radius);
	if(k<0) k=0;
	else if(k> grid_depth-1) k=grid_depth-1;

	list<Particle*>*tmp =&( target_grid[grid_index(i, j, k)].particles);
	tmp->push_back(&particle);
	//Particles[tmp->back()->id]=tmp->back();
}

inline int PbfSolver::grid_index(const int &i,const int &j, const int &k ){
	return grid_width * (k * grid_height + j) + i;
}

inline GridElement &PbfSolver::grid(int i, int j, int k) {
	return grid_elements[grid_index(i, j, k)];
}

void PbfSolver::findNeighbor(){
	update_grid();
	for(int iter=0;iter<ParticleCont;iter++){
		if(Particles[iter]->isOnSurface)
			continue;

		Particles[iter]->myNeighbor.clear();
		int i = (int) (Particles[iter]->PredictedPos[0] / core_radius);
		if(i<0) i=0;
		else if(i> grid_width-1) i=grid_width-1;
		int j = (int) (Particles[iter]->PredictedPos[1] / core_radius);
		if(j<0) j=0;
		else if(j> grid_height-1) j=grid_height-1;
		int k = (int) (Particles[iter]->PredictedPos[2] / core_radius);
		if(k<0) k=0;
		else if(k> grid_depth-1) k=grid_depth-1;


		for(int iiter=i-1;iiter<i+2;iiter++){
			if(iiter<0||iiter>grid_width-1) continue;
			for(int jiter=j-1;jiter<j+2;jiter++){
				if(jiter<0||jiter>grid_height-1) continue;
				for(int kiter=k-1;kiter<k+2;kiter++){
					if(kiter<0||kiter>grid_depth-1) continue;

					for(list<Particle*>::iterator pit=grid(iiter,jiter,kiter).particles.begin();
							pit!=grid(iiter,jiter,kiter).particles.end();
							++pit){
								if(Particles[iter]->myNeighbor.find(*pit)!=Particles[iter]->myNeighbor.end()) continue;
								if(Particles[iter]==*pit) continue;
								if((Particles[iter]->PredictedPos-(*pit)->PredictedPos).Length()<core_radius){
									Particles[iter]->myNeighbor.insert(*pit);
									(*pit)->myNeighbor.insert(Particles[iter]);
								}

					}

				}
			}
		}
	}
}