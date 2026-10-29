#version 440 core

in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D u_screenTexture;

void main()
{
    FragColor = texture(u_screenTexture, vTexCoord);
}

