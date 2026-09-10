#version 430 core
layout(location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

#define MAX_OBJS 200
struct sphreState
{
    vec4 position_mass;
    vec4 velocity_radius;
};

layout(std430, binding = 0) readonly buffer sphreStateSSBO
{
    sphreState sphres[MAX_OBJS];
};

uniform int u_numObjs;
uniform bool u_isGrid;

void main()
{
    vec3 vertexPos = aPos;
    if (u_isGrid)
    {
        float totalDisplacement = 0.0;
        for (int i = 0; i < u_numObjs; ++i)
        {
            vec3 objPos = sphres[i].position_mass.xyz;
            float objRs = sphres[i].velocity_radius.w;
            vec3 toObject = objPos - vertexPos;
            float distance_m = length(toObject) * 1000.0;

            distance_m = max(distance_m, objRs * 1.0001);

            if (objRs > 0.0)
            {
                totalDisplacement -= (2.0e11 * objRs) / distance_m;
            }
        }
        vertexPos.y = aPos.y + totalDisplacement;
    }
    gl_Position = projection * view * model * vec4(vertexPos, 1.0);
}
