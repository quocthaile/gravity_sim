#pragma once

#include "gpu_memory_manager.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

extern const double kGravitationalConstant;
extern const float kSpeedOfLight; // đổi từ float qua double

// numerical error
extern double dt;
void SavePositions(const char *filename, const std::vector<std::vector<glm::vec3>> &positions,
                   double dt);

class Object;

// Stable logical object identity used for validation/result matching. It is independent of the
// object's vector index or GPU slot. 0 is reserved as "unassigned"; allocated IDs start at 1.
constexpr std::uint32_t kInvalidObjectId = 0;
std::uint32_t AllocateObjectId();

struct ShaderPrograms
{
    GLuint gridRendering = 0;
    GLuint gridCompute = 0;
    GLuint objectRendering = 0;
    GLuint nbodyCompute = 0;
    GLuint integrateCompute = 0;
};

struct ShaderUniformLocations
{
    GLint gridModel = -1;
    GLint gridObjectColor = -1;
    GLint gridView = -1;
    GLint gridProjection = -1;
    GLint objectView = -1;
    GLint objectProjection = -1;
};

struct ComputeUniformLocations
{
    GLint nbodyObjectCount = -1;
    GLint nbodyGravitationalConstant = -1;
    GLint nbodyEpsilon = -1;

    GLint integrateObjectCount = -1;
    GLint integratePaused = -1;

    GLint gridObjectCount = -1;
    GLint gridWidth = -1;
    GLint gridHeight = -1;
};

GLFWwindow *StartGLU();
std::string LoadShaderSource(const std::string &filePath);
void CreateShaderProgram(ShaderPrograms &programs);
void InitializeComputeUniformLocations(const ShaderPrograms &programs,
                                       ComputeUniformLocations &locations);
// inline const glm::vec3 kInitialCameraPosition(0.0f, 1000.0f, 5000.0f);
glm::mat4 CreateScenePerspectiveProjection();
void InitializeGridRenderingUniforms(GLuint gridProgram, ShaderUniformLocations &uniformLocations);
void InitializeObjectRenderingUniforms(GLuint objectProgram,
                                       ShaderUniformLocations &uniformLocations);
void CreateUnitSphere();
void DestroyObjectRenderingResources();
GLuint CreateGraphicsProgram(const std::string &vertexFilePath,
                             const std::string &fragmentFilePath);
GLuint CreateComputeProgram(const std::string &filePath);
void CreateMeshBuffers(GLuint &VAO, GLuint &vbo, const float *vertices, size_t vertexCount,
                       GLuint *ebo = nullptr, const unsigned int *indices = nullptr,
                       size_t indexCount = 0);
std::vector<glm::vec4> CreateBaseGridGPU();
void InitializeGridRenderingResources(const std::vector<glm::vec4> &gridVertices,
                                      const std::vector<unsigned int> &gridIndices);
void ComputePipeline(const std::vector<glm::vec4> &basePositions);
void RunNBodyCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                     size_t objectCount, float epsilon);
void RunIntegration(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                    size_t objectCount, bool paused);
void RunObjectStateSync(GLFWwindow *window, size_t &objectCount);
void RunGpuStateTransition(GLuint nbodyShaderProgram, GLuint integrateShaderProgram,
                           const ComputeUniformLocations &locations, size_t objectCount,
                           float epsilon, bool paused);
void RunGridCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                    size_t objectCount);
void Cleanup(const ShaderPrograms &shaderPrograms);
void BeginFrame();
void UpdateInitializingObject(GLFWwindow *window);
void UpdateCamera(const ShaderPrograms &shaderPrograms,
                  const ShaderUniformLocations &uniformLocations, glm::vec3 cameraPosition);
void KeyCallback(GLFWwindow *window, int key, int scanCode, int action, int mods);
void MouseCallback(GLFWwindow *window, double xPosition, double yPosition);
void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
void ScrollCallback(GLFWwindow *window, double xOffset, double yOffset);
void InitializeGlfwCallbacks(GLFWwindow *window);
glm::vec3 SphericalToCartesian(float radius, float theta, float phi);
void DrawGrid(GLuint renderingShaderProgram, GLuint gridVAO, size_t indexCount,
              const ShaderUniformLocations &uniformLocations);
void DrawObjects(GLuint objectRenderingShaderProgram, const std::vector<Object> &objects);
std::vector<float> CreateGridVertices(float size, int divisions);
std::vector<unsigned int> CreateGridIndices(int divisions);
std::vector<Object> CreateRandomOrbiters(int count, const glm::vec3 &center, float centralMass);
std::vector<Object> CreateObjects(const std::string &objectFilePath = {},
                                  int randomObjectCount = 0);
class Object
{
  public:
    // Identity (stable logical ID, not the vector index / GPU slot)
    std::uint32_t id = kInvalidObjectId;
    // Set when this object is materialized in the GPU state buffers.
    size_t gpuStateIndex = std::numeric_limits<size_t>::max();
    // Rendering metadata
    glm::vec4 color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    // Simulation state
    glm::vec3 position = glm::vec3(400, 300, 0);
    glm::vec3 velocity = glm::vec3(0, 0, 0);
    // Interaction state
    bool initializing = false;
    bool launched = false;
    bool target = false;
    // Physical parameter
    float mass;
    float density;
    // Derived state
    float radius; // đổi từ float qua double
    float rs;
    // Auxiliary state
    glm::vec3 lastPos = position;
    // Constructor
    Object(glm::vec3 initPosition, glm::vec3 initVelocity, float mass, float density = 3344)
    {
        this->id = AllocateObjectId();
        this->position = initPosition;
        this->velocity = initVelocity;
        this->mass = mass;
        this->density = density;
        this->radius =
            pow(((3 * this->mass / this->density) / (4 * 3.14159265359)), (1.0f / 3.0f)) / 100000;
        this->rs = (2 * kGravitationalConstant * this->mass) / (kSpeedOfLight * kSpeedOfLight);
    }
    void UpdatePosition(double dt)
    {
        // this->position[0] += this->velocity[0] / 94;
        // this->position[1] += this->velocity[1] / 94;
        // this->position[2] += this->velocity[2] / 94;
        this->position[0] += this->velocity[0] * dt;
        this->position[1] += this->velocity[1] * dt;
        this->position[2] += this->velocity[2] * dt;
        this->radius =
            pow(((3 * this->mass / this->density) / (4 * 3.14159265359)), (1.0f / 3.0f)) / 100000;
    }
    glm::vec3 GetPosition() const { return this->position; }

    void Accelerate(float x, float y, float z, double dt)
    {
        // this->velocity[0] += x / 96;
        // this->velocity[1] += y / 96;
        // this->velocity[2] += z / 96;
        this->velocity[0] += x * dt;
        this->velocity[1] += y * dt;
        this->velocity[2] += z * dt;
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

struct GpuObjectState
{
    glm::vec4 position;
    glm::vec4 velocity;
};

struct GpuObjectPhysical
{
    glm::vec4 massDensity;
};

struct GpuObjectDerived
{
    glm::vec4 radiusRs;
};

struct GpuObjectControl
{
    std::uint32_t initializing;
    std::uint32_t launched;
    std::uint32_t target;
    // Stable logical object ID (mirrors Object::id). Reuses the former padding slot so the
    // struct stays 16 bytes and the std430 layout at binding 3 is unchanged.
    std::uint32_t id;
};

struct GpuObjectAcceleration
{
    glm::vec4 acceleration;
};

struct GpuObjectCollision
{
    std::uint32_t overlapCount;
};

struct GpuObjectRender
{
    glm::vec4 color;
};

static_assert(sizeof(GpuObjectState) == 2 * sizeof(glm::vec4));
static_assert(sizeof(GpuObjectPhysical) == sizeof(glm::vec4));
static_assert(sizeof(GpuObjectDerived) == sizeof(glm::vec4));
static_assert(sizeof(GpuObjectControl) == sizeof(glm::vec4));
static_assert(offsetof(GpuObjectControl, id) == 3 * sizeof(std::uint32_t));
static_assert(sizeof(GpuObjectAcceleration) == 16);
static_assert(sizeof(GpuObjectCollision) == 4);
static_assert(sizeof(GpuObjectRender) == sizeof(glm::vec4));

size_t CalculateObjectStateBufferCapacity(size_t requiredCount);
void ManageObjectStateBufferCapacity(size_t objectCount);
void objectGpuState(size_t firstObject, size_t objectCount);
void SynchronizeObjectStateBuffers(size_t &objectCount, size_t newObjectCount);
void SynchronizeObjectGpuProperties(size_t objectIndex);
void UploadInitializingObjectState(size_t objectIndex);

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
extern float initMass; // đổi từ float qua double
extern int numRandomObjects;
extern float gridSize;
extern int gridDivisions;
extern const GLuint kGridLocalSizeX;
extern const GLuint kGridLocalSizeY;
extern std::vector<Object> objs;
extern GLuint gridVAO;
extern GLuint gridEBO;
extern GpuMemoryManager gpuMemoryManager;
extern size_t gridNodeCount;
extern size_t gridIndexCount;
extern size_t objectStateCapacity;

// mass aggregation
extern float CEll_SIZE;
extern float worldMin;
extern float worldMax;