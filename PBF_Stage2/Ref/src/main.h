#ifndef MAIN_H
#define MAIN_H

#include<Windows.h>


#include <GL/glew.h>
#include <GL/glut.h>

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "glslUtility.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "kernel.h"
#include "utilities.h"


#if CUDA_VERSION >= 5000
    #include <helper_cuda.h>
    #include <helper_cuda_gl.h>
    #define compat_getMaxGflopsDeviceId() gpuGetMaxGflopsDeviceId() 
#else
    #include <cutil_inline.h>
    #include <cutil_gl_inline.h>
    #define compat_getMaxGflopsDeviceId() cutGetMaxGflopsDeviceId()
#endif

#ifdef WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include "Macros.h"
using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

GLuint positionLocation = 0;
unsigned int colorLocation;
unsigned int normalLocation;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Normal", "Color", "Position", "Texcoords" };
GLuint pbo = (GLuint)NULL;
GLuint planeVBO = (GLuint)NULL;
GLuint planeTBO = (GLuint)NULL;
GLuint planeIBO = (GLuint)NULL;
GLuint planetVBO = (GLuint)NULL;
GLuint planetIBO = (GLuint)NULL;
GLuint displayImage;
GLuint program[3];

GLuint sphereVBO = (GLuint)NULL;
GLuint sphereTBO = (GLuint)NULL;
GLuint sphereIBO = (GLuint)NULL;

GLuint depthTexture = 0;
GLuint normalTexture = 0;
GLuint positionTexture = 0;
GLuint colorTexture = 0;
GLuint FBO[1] = {0};	

const unsigned int HEIGHT_FIELD = 0;
const unsigned int PASS_THROUGH = 1;

const int field_width  = 20;
const int field_height = 50;

float fovy = 60.0f;
float zNear = 0.01;
float zFar = 1000.0f;

glm::mat4 projection;
glm::mat4 view;
glm::vec3 cameraPosition(-40.75,10,50);
glm::vec3 center(0,0,0);

int ParticleConts = 10000;
//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=1280; int height=720;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

void printUsage();
int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

void display();
void keyboard(unsigned char key, int x, int y);

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void init(int argc, char* argv[]);


void initPBO(GLuint* pbo);
void initCuda();
void initTextures();
void initVAO();
void initShaders(GLuint * program);
void updateCamera(GLuint* program,bool rotate);
void initFBO(int w, int h);
void bindFBO(int n);

//-------------------------------
//---------- GEOM STUFF ---------
//-------------------------------


//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

//camera stuff
float theta = 0; //r and l
float phi = 30; //up and down
float r = glm::length(cameraPosition-center);
float oldx;
float oldy;

bool Lpressed = false;
bool Rpressed = false;

void mouse(int button, int state, int x, int y);
void drag(int x, int y);
void update(int x, int y);

#endif
