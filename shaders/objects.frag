#version 440 core

in vec4 vertexColor;
out vec4 fragmentColor;

void main()
{
    fragmentColor = vertexColor;
}
