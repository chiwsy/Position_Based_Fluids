#ifndef GRIDSTRUCT_H
#define GRIDSTRUCT_H
//#include<vector>

struct particle{
	int LayerMask;
	unsigned int ID;
	float inv_mass;
	float invRestDensity;
	glm::vec4 position;
	glm::vec4 pred_position;
	glm::vec3 velocity;
	float lambda;
	glm::vec3 delta_pos;
	glm::vec3 external_forces;
	glm::vec3 curl;
};



#endif GRIDSTRUCT_H