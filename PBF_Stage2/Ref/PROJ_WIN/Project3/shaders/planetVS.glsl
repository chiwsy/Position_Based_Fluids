#version 150

in vec4 Position;
out vec4 color;
void main(void)
{
	gl_Position = vec4(Position.xyz,1.0);
	//Color selection:
	color = vec4(0.3, 0.7, .2,1.)-(Position.w)*vec4(-.4, .3, -.6,.5);
}