
#define epsilon .1
#define c_viscocity .01
#define MaxIteration 3

#define jt (rand()%1000)/10000.0f
//#define gravity vec3(0,-9.8,0)

#define collision_restitution .0001f
enum ModeFlag {VORTICITY,ETA,VISCOSITY,RHO,GRADIENT,GRADIENTANTI,DELTAP};

#define ContainerFileName "bunny_fu.obj"

#define WIDTH		20
#define HEIGHT	80
#define DEPTH		20

const int particle_count =3000  ;
const float scale = 1.0f;

#define CORE_RADIUS 1.8f
#define REST_DENSITY 2000.0f
#define CONST_INV_REST_DENSITY .01f
#define TimeStep .05f
#define RELAXATION .01

#define PRESSURE_K 0.1
#define PRESSURE_N 6


/**************************************************************************
Selection of Obj files:
"bunny.obj"
"bunny_fu.obj"
"bunny_fu_low.obj"

"dragon.obj"

"cylinder_container.obj"

"hack_cube.obj"
"cont.obj"
***************************************************************************/