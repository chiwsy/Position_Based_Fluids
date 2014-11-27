#version 150

in vec4 Position;
out vec3 color;
void main(void)
{
	gl_Position = vec4(Position.xyz,1.0);
	color = vec3(0.3, 0.7, .2)-(Position.w)*vec3(-.4, .3, -.6);
}