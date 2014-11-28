#version 150

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;

//in vec3 out_color;
//in vec3 clr;

in vec4 clr;
out vec4 FragColor;

out vec4 out_Position;
out vec4 out_Normal;



void main()
{

    vec2 coord = 2.01 * (TexCoord - vec2(0.5));
    float r = length(coord);
    if (r >= .25) { discard; }

    float dist = length(vec3(0,0,25)-WorldCoord);
    vec3 N = Right*-coord.x + Up*coord.y + ToCam*sqrt(1-r*r);
    vec3 L = normalize(vec3(0,30,80)-WorldCoord);
    float light = 0.2 + 0.8*clamp(dot(N,L),0.0, 1.0);
    //vec3 color = vec3(0.3, 0.7, .2);
	FragColor = vec4(clr.rgb*light,clr.a);
	out_Position=vec4(WorldCoord,1.0);
	out_Normal=vec4(N,1.0);
} 