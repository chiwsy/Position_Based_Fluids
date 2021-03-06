#version 150

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

//in vec3 color;
//out vec3 out_color;
layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;


out vec3 WorldCoord;
out vec3 ToCam;
out vec3 Up;
out vec3 Right;
out vec2 TexCoord;

in vec4 color[];
out vec4 clr;
//in vec3 color;

const float scale = 3.0;

void main()
{
	clr = color[0];
	//out_color = color;

    vec3 Position = gl_in[0].gl_Position.xyz;
    WorldCoord = Position;
	

    ToCam = normalize(u_cameraPos - Position);
    Up = vec3(0.0, 0.0, 1.0);

    Right = normalize(cross(ToCam, Up));

    Up = cross(Right, ToCam);

    vec3 Pos = Position + scale*Right - scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 0.0);
    EmitVertex();

    Pos = Position + scale*Right + scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 1.0);
    EmitVertex();

    Pos = Position - scale*Right - scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(1.0, 0.0);
    EmitVertex();

    Pos = Position - scale*Right + scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(1.0, 1.0);
    EmitVertex();

    EndPrimitive();
}