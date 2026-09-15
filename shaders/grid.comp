#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

struct SphereState
{
    vec4 position_mass;
    vec4 velocity_radius;
};

layout(std430, binding = 0) readonly buffer SphereStateBuffer { SphereState spheres[]; };

layout(std430, binding = 1) readonly buffer BaseGridBuffer { vec4 basePositions[]; };

layout(std430, binding = 2) writeonly buffer DeformedGridBuffer { vec4 deformedPositions[]; };

uniform uint u_objectCount;
uniform uint u_gridWidth;
uniform uint u_gridHeight;

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint z = gl_GlobalInvocationID.y;
    if (x >= u_gridWidth || z >= u_gridHeight)
    {
        return;
    }

    uint gridIndex = z * u_gridWidth + x;
    vec3 basePosition = basePositions[gridIndex].xyz;
    float totalDisplacement = 0.0;
    for (uint i = 0; i < u_objectCount; ++i)
    {
        vec3 objPos = spheres[i].position_mass.xyz;
        float objRs = spheres[i].velocity_radius.w;
        vec3 toObject = objPos - basePosition;
        float distance_m = length(toObject) * 1000.0;

        distance_m = max(distance_m, objRs * 1.0001);

        if (objRs > 0.0)
        {
            totalDisplacement -= (2.0e11 * objRs) / distance_m;
        }
    }

    deformedPositions[gridIndex] =
        vec4(basePosition.x, basePosition.y + totalDisplacement, basePosition.z, 1.0);
}
