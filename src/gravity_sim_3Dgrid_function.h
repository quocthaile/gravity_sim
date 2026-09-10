#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <array>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>

extern const double G;
extern const float c;

class Object;

GLFWwindow *StartGLU();
std::string LoadShaderSource(const std::string &filePath);
GLuint CreateShaderProgram(const char *vertexSource, const char *fragmentSource);
void CreateMeshBuffers(GLuint &VAO, GLuint &VBO, const float *vertices, size_t vertexCount, GLuint *EBO = nullptr,
                       const unsigned int *indices = nullptr, size_t indexCount = 0);
void InitializeRenderingPipeline();
void InitializeSimulationPipeline();
void Cleanup(GLuint shaderProgram);
void UpdateCam(GLuint shaderProgram, GLint viewLoc, glm::vec3 cameraPos);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void InitializeGlfwCallbacks(GLFWwindow *window);
glm::vec3 sphericalToCartesian(float r, float theta, float phi);
void DrawGrid(GLuint shaderProgram, GLuint gridVAO, size_t indexCount);
std::vector<float> CreateGridVertices(float size, int divisions);
std::vector<unsigned int> CreateGridIndices(int divisions);
std::vector<Object> CreateRandomOrbiters(int count, const glm::vec3 &center, float centralMass);

class Object
{
  public:
    GLuint VAO, VBO;
    glm::vec3 position = glm::vec3(400, 300, 0);
    glm::vec3 velocity = glm::vec3(0, 0, 0);
    size_t vertexCount;
    glm::vec4 color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

    bool Initalizing = false;
    bool Launched = false;
    bool target = false;

    float mass;
    float density;
    float radius;
    float rs;

    glm::vec3 LastPos = position;

    Object(glm::vec3 initPosition, glm::vec3 initVelocity, float mass, float density = 3344)
    {
        this->position = initPosition;
        this->velocity = initVelocity;
        this->mass = mass;
        this->density = density;
        this->radius = pow(((3 * this->mass / this->density) / (4 * 3.14159265359)), (1.0f / 3.0f)) / 100000;
        this->rs = (2 * G * this->mass) / (c * c);
        std::vector<float> vertices = DrawSphereMesh();
        vertexCount = vertices.size();
        CreateMeshBuffers(VAO, VBO, vertices.data(), vertexCount);
    }

    std::vector<float> DrawSphereMesh()
    {
        std::vector<float> vertices;
        int stacks = 10;
        int sectors = 10;
        for (float i = 0.0f; i <= stacks; ++i)
        {
            float theta1 = (i / stacks) * glm::pi<float>();
            float theta2 = (i + 1) / stacks * glm::pi<float>();
            for (float j = 0.0f; j < sectors; ++j)
            {
                float phi1 = j / sectors * 2 * glm::pi<float>();
                float phi2 = (j + 1) / sectors * 2 * glm::pi<float>();
                glm::vec3 v1 = sphericalToCartesian(radius, theta1, phi1);
                glm::vec3 v2 = sphericalToCartesian(radius, theta1, phi2);
                glm::vec3 v3 = sphericalToCartesian(radius, theta2, phi1);
                glm::vec3 v4 = sphericalToCartesian(radius, theta2, phi2);
                vertices.insert(vertices.end(), {v1.x, v1.y, v1.z});
                vertices.insert(vertices.end(), {v2.x, v2.y, v2.z});
                vertices.insert(vertices.end(), {v3.x, v3.y, v3.z});
                vertices.insert(vertices.end(), {v2.x, v2.y, v2.z});
                vertices.insert(vertices.end(), {v4.x, v4.y, v4.z});
                vertices.insert(vertices.end(), {v3.x, v3.y, v3.z});
            }
        }
        return vertices;
    }

    void UpdatePos()
    {
        this->position[0] += this->velocity[0] / 94;
        this->position[1] += this->velocity[1] / 94;
        this->position[2] += this->velocity[2] / 94;
        this->radius = pow(((3 * this->mass / this->density) / (4 * 3.14159265359)), (1.0f / 3.0f)) / 100000;
    }

    void UpdateVertices()
    {
        std::vector<float> vertices = DrawSphereMesh();
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    }

    glm::vec3 GetPos() const { return this->position; }

    void accelerate(float x, float y, float z)
    {
        this->velocity[0] += x / 96;
        this->velocity[1] += y / 96;
        this->velocity[2] += z / 96;
    }

    float CheckCollision(const Object &other)
    {
        float dx = other.position[0] - this->position[0];
        float dy = other.position[1] - this->position[1];
        float dz = other.position[2] - this->position[2];
        float distance = std::pow(dx * dx + dy * dy + dz * dz, (1.0f / 2.0f));
        if (other.radius + this->radius > distance)
        {
            return -0.2f;
        }
        return 1.0f;
    }
};

struct sphreStateCPU
{
    glm::vec4 position_mass;
    glm::vec4 velocity_radius;
};

static_assert(sizeof(sphreStateCPU) == 2 * sizeof(glm::vec4),
              "sphreStateCPU must be a vec4 pair for std430 compatibility.");
static_assert(alignof(sphreStateCPU) == alignof(glm::vec4), "sphreStateCPU must respect vec4 alignment.");

inline constexpr int maxObjects = 200;

extern bool running;
extern bool pause;
extern glm::vec3 cameraPos;
extern glm::vec3 cameraFront;
extern glm::vec3 cameraUp;
extern float lastX;
extern float lastY;
extern float yaw;
extern float pitch;
extern float deltaTime;
extern float lastFrame;
extern float initMass;
extern int numRandomObjects;
extern float gridSize;
extern int gridDivisions;
extern std::vector<Object> objs;
extern GLuint gridVAO;
extern GLuint gridVBO;
extern GLuint gridEBO;
extern GLuint sphreStateSSBO;
extern size_t gridIndexCount;
extern std::array<sphreStateCPU, maxObjects> sphreStateData;
