#ifdef _WIN32
	#include<time.h>
	#include<Time.h>
	#include<stdio.h>
	#include<stdlib.h>
	#include "GLUT/glut.h"
#else
	#include <GL/glut.h>
#endif

#include "PBF.h"
#include "fps.h"

using namespace std;

bool zoom, trans;
bool paused = false;

vec3 gravity_direction;
//float origin_Gravity=-9.8;
vec3 origin_Gravity(-10.0,-10.0,-10.0);
vec3 gravity(0,-99.8,0);

int wndWidth = 700;
int wndHeight= 700;
int oldX, oldY, rotX = 0, rotY = 0, zoomZ = 0;
int oldTransX, oldTransY, transX = 0, transY = 0;
int simulation_steps = 2;

//extern vec3 gravity;

GLuint sphereId;
GLfloat rotation_matrix[16];

mmc::FpsTracker theFpsTracker;
FluidMaterial material(1000.0f, 0.1f,REST_DENSITY, 1.0f, 2.0f);
PbfSolver solver(WIDTH, HEIGHT, DEPTH, CORE_RADIUS, TimeStep, material,particle_count);
//PbfSolver solver;
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void reshape(int width, int height) {
	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glOrtho(-0.1 * WIDTH, 1.1 * WIDTH, +0.1 * HEIGHT, -1.1 * HEIGHT, 1.0, 1000.0);
	gluPerspective(55.0, 1.0, 1.0, 1000.0);

	wndWidth = width;
	wndHeight = height;
}


void mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON) {
		if (glutGetModifiers() & GLUT_ACTIVE_CTRL) {
			trans = true;
			oldTransX = x - transX;
			oldTransY = y - transY;
		} else {
			trans = false;
			oldX = x;
			oldY = y;
		}
	} else if (button == GLUT_RIGHT_BUTTON) {
		zoom = !zoom;
		oldY = y - zoomZ;
	}
}

void motion(int x, int y) {
	if (!zoom) {
		if (trans) {
			transX = x - oldTransX;
			transY = y - oldTransY;
		} else {
			rotY = x - oldX;
			oldX = x;
			rotX = y - oldY;
			oldY = y;
		}
	} else {
		zoomZ = y - oldY;
	}

	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
	/*case ' ':
		paused = !paused;
		break;*/
	case 'g':
	case 'G':
		solver.dropToGround=true;
		break;
	case 'q':
	case 'Q':
	case 0x1bU: /* ESC */
		exit(0);
	default:
		break;
	}
}

void idle() {
	glutPostRedisplay();
}

void init() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glEnable(GL_DEPTH_TEST);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glColorMaterial(GL_FRONT,GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	sphereId = glGenLists(1);
	glNewList(sphereId, GL_COMPILE);
	glutSolidSphere(0.5, 20, 20);
	glEndList();
}

void init_liquid() {
	Particle *particles = new Particle[particle_count];
	srand(time(NULL));
	int count = particle_count;
	Particle *particle_iter = particles;

	/*for (float k = 0; k < DEPTH; k+=1) {
	    for (float i = 0; i < WIDTH; i+=1) {
			count--;
			particle_iter->position[0] = i / scale;
			particle_iter->position[1] = 1.0;
			particle_iter->position[2] = k / scale;
			particle_iter->PredictedPos=particle_iter->position;
			particle_iter->isOnSurface=true;
			particle_iter->invRestDensity=CONST_INV_REST_DENSITY;
			particle_iter++;

		}
	}

	for (float j = 0; j < HEIGHT; j+=1) {
		for (float i = 0; i < WIDTH; i+=1) {
			count--;
			particle_iter->position[0] = i / scale;
			particle_iter->position[1] = j/scale;
			particle_iter->position[2] = 0;
			particle_iter->PredictedPos=particle_iter->position;
			particle_iter->isOnSurface=true;
			particle_iter->invRestDensity=CONST_INV_REST_DENSITY;
			particle_iter++;

			count--;
			particle_iter->position[0] = i / scale;
			particle_iter->position[1] = j/scale;
			particle_iter->position[2] = (DEPTH-1) / scale;
			particle_iter->PredictedPos=particle_iter->position;
			particle_iter->isOnSurface=true;
			particle_iter->invRestDensity=CONST_INV_REST_DENSITY;
			particle_iter++;
		}
	}

	for (float j = 0; j < HEIGHT; j+=1) {
		for (float k = 0; k < DEPTH; k+=1) {
			count--;
			particle_iter->position[0] = 0;
			particle_iter->position[1] = j/scale;
			particle_iter->position[2] = k/scale;
			particle_iter->PredictedPos=particle_iter->position;
			particle_iter->isOnSurface=true;
			particle_iter->invRestDensity=CONST_INV_REST_DENSITY;
			particle_iter++;

			count--;
			particle_iter->position[0] = (WIDTH-1) / scale;
			particle_iter->position[1] = j/scale;
			particle_iter->position[2] = k/ scale;
			particle_iter->PredictedPos=particle_iter->position;
			particle_iter->isOnSurface=true;
			particle_iter->invRestDensity=CONST_INV_REST_DENSITY;
			particle_iter++;
		}
	}*/

	count-=solver.objContainer.position.size();
	for(int i=0;i<solver.objContainer.position.size();i++){
		particle_iter->position=solver.objContainer.position[i]+vec3(0,2,0);
		particle_iter->PredictedPos=particle_iter->position;
		particle_iter->invRestDensity=CONST_INV_REST_DENSITY;
		particle_iter->isOnSurface=true;
		particle_iter++;
	}
	//solver.init_particles(particles, solver.objContainer.position.size());

	while (true) {
		for (int j = 1; j < HEIGHT-1; j++) {
			for (int k = 1; k < DEPTH-1; k++) {
				for (int i = 1; i < WIDTH-1; i++) {
					if (count-- == 0) {

						solver.init_particles(particles, particle_count);
						return;
					}

                    float rd=(rand()%2000-1000)/(10000.0f);

					//rd=0.0f;

					particle_iter->position[0] = i / scale+rd;
					rd=(rand()%2000-1000)/(10000.0f);
					particle_iter->position[1] = j / scale+.0f+rd;
					rd=(rand()%2000-1000)/(10000.0f);
					particle_iter->position[2] = k / scale+rd;
					particle_iter->PredictedPos=particle_iter->position;
					particle_iter++;
				}
			}
		}
	}
}

void drawOverlay()
{
  // Draw Overlay
  //glPopMatrix();
  glPushMatrix();
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glPushAttrib(GL_LIGHTING_BIT);
     glDisable(GL_LIGHTING);

     glMatrixMode(GL_PROJECTION);
     glLoadIdentity();
     gluOrtho2D(0.0, 1.0, 0.0, 1.0);

     glMatrixMode(GL_MODELVIEW);
     glLoadIdentity();
     glRasterPos2f(0.01, 0.01);
     
     char info[1024];
     sprintf(info, "Framerate: %1.1f %s %d %s ", 
         theFpsTracker.fpsAverage(),"  Particle Count:",particle_count,"   Click and drag to rotate, press g or G to drop the container");
 
     for (unsigned int i = 0; i < strlen(info); i++)
     {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, info[i]);
     }

	 glEnable(GL_LIGHTING);
  glPopAttrib();
  glPopMatrix();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(55.0, 1.0, 1.0, 1000.0);
}

void draw_particle(Particle &particle) {
	vec3 p = scale * particle.position;
	glTranslatef(+p[0], +p[1], +p[2]);
	if(!particle.isOnSurface)
		glColor3f(.125f,107/255.f,164/255.f);
	else
		glColor3f(122/255.f,186/255.f,122/255.f);
	glCallList(sphereId);
#if 0
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	glColor3ub(0, 0, 255);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glColor3ub(255, 0, 0);
	vec3 pos = particle.pressure_force / particle.density;
	glVertex3fv((GLfloat *) &pos);
	glEnd();
	glEnable(GL_LIGHTING);
#endif
	glTranslatef(-p[0], -p[1], -p[2]);
}

void extract_gravity_direction() {
	
	gravity_direction[0] = rotation_matrix[1];
	gravity_direction[1] = rotation_matrix[5];
	gravity_direction[2] = rotation_matrix[9];
	gravity_direction = gravity_direction.Normalize();
}


void display() {
	theFpsTracker.timestamp();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glGetFloatv(GL_MODELVIEW_MATRIX, rotation_matrix);

	/* Handle rotations and translations separately. */
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glRotatef(rotY, 0.0f, 1.0f, 0.0f);
	glRotatef(rotX, 1.0f, 0.0f, 0.0f);
	glMultMatrixf(rotation_matrix);

	/* Save the rotational component of the modelview matrix. */
	glPushMatrix();
	glLoadIdentity();

	gluLookAt(0.0, 0.0, 30.0,
	          0.0, 0.0, 0.0,
	          0.0, 1.0, 0.0);

	glTranslatef(0.0f, 0.0f, (float)zoomZ / 20.0f);
	glRotatef(rotY, 0.0f, 1.0f, 0.0f);
	glRotatef(rotX, 1.0f, 0.0f, 0.0f);
	glMultMatrixf(rotation_matrix);
	glTranslatef(-WIDTH / 2.0f, -HEIGHT / 2.0f, -DEPTH / 2.0f);

	rotX = rotY = 0;

	extract_gravity_direction();
	gravity=origin_Gravity*gravity_direction;
	//printf("gravity:[%f,%f,%f]\n",gravity[0],gravity[1],gravity[2]);
	if (!paused) {
		for (int i = 0; i < simulation_steps; ++i) {
			solver.update(TimeStep);
		}
	}


	solver.foreach_particle(draw_particle);

	
    glPopMatrix();
	drawOverlay();
	glutSwapBuffers();
	
	/* Restore the rotational component of the modelview matrix. */
	
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
	//print_usage();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(wndWidth, wndHeight);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("PB Fluids");

	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(display);
	glutIdleFunc(idle);

	init();
	init_liquid();

	glutMainLoop();

	return (0);
}