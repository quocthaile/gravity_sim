#version 440 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform int u_flipY;

out vec2 vTexCoord;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    float ty = (u_flipY != 0) ? (1.0 - aTexCoord.y) : aTexCoord.y;
    vTexCoord = vec2(aTexCoord.x, ty);
}

