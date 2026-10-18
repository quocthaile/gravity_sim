#version 440 core

layout(location = 0) in vec3 localPosition;

struct ObjectState
{
    vec4 position;
    vec4 velocity;
};

layout(std430, binding = 0) readonly buffer CurrentStateBuffer
{
    ObjectState currentState[];
};
layout(std430, binding = 12) readonly buffer ObjectDerivedBuffer
{
    vec4 radiusRs[];
};
layout(std430, binding = 14) readonly buffer ObjectRenderBuffer
{
    vec4 objectColors[];
};

uniform mat4 view;
uniform mat4 projection;

out vec4 vertexColor;

void main()
{
    uint objectIndex = uint(gl_InstanceID);
    vec3 worldPosition = currentState[objectIndex].position.xyz +
                         localPosition * radiusRs[objectIndex].x;
    vertexColor = objectColors[objectIndex];
    gl_Position = projection * view * vec4(worldPosition, 1.0);
}
