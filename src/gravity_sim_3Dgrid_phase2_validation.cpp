#include "gravity_sim_3Dgrid_function.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

const double kGravitationalConstant = 6.6743e-11;
const float kSpeedOfLight = 299792458.0f;
extern constexpr GLuint kGridLocalSizeX = 16;
extern constexpr GLuint kGridLocalSizeY = 16;

bool running = true;
bool pause = false;
glm::vec3 cameraPos(0.0f);
glm::vec3 cameraFront(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
float lastX = 0.0f;
float lastY = 0.0f;
float yaw = -90.0f;
float pitch = 0.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
float initMass = 1.0e20f;
int numRandomObjects = 0;
float gridSize = 1.0f;
int gridDivisions = 1;
std::vector<Object> objs;
GLuint gridVAO = 0;
GLuint gridEBO = 0;
GpuMemoryManager gpuMemoryManager;
size_t gridNodeCount = 0;
size_t gridIndexCount = 0;
size_t objectStateCapacity = 0;

namespace
{
struct ReferenceObject
{
    std::uint32_t id;
    glm::vec3 position;
    glm::vec3 velocity;
    float mass;
    float radius;
    bool initializing;
};

struct AccelerationSample
{
    std::uint32_t id;
    glm::vec3 acceleration;
};

void Require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

std::vector<AccelerationSample> AdvanceReference(std::vector<ReferenceObject> &objects,
                                                 float epsilon)
{
    const std::vector<ReferenceObject> currentState = objects;
    std::vector<ReferenceObject> nextState = currentState;
    std::vector<AccelerationSample> accelerations;
    accelerations.reserve(currentState.size());

    for (size_t targetIndex = 0; targetIndex < currentState.size(); ++targetIndex)
    {
        const ReferenceObject &target = currentState[targetIndex];
        glm::vec3 velocity = target.velocity;
        glm::vec3 totalAcceleration(0.0f);

        if (!target.initializing)
        {
            for (size_t sourceIndex = 0; sourceIndex < currentState.size(); ++sourceIndex)
            {
                const ReferenceObject &source = currentState[sourceIndex];
                if (sourceIndex == targetIndex || source.initializing)
                {
                    continue;
                }

                const float dx = source.position.x - target.position.x;
                const float dy = source.position.y - target.position.y;
                const float dz = source.position.z - target.position.z;
                const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (distance <= 0.0f)
                {
                    continue;
                }

                const glm::vec3 direction(dx / distance, dy / distance, dz / distance);
                const float distanceMeters = distance * 1000.0f;
                const double gravitationalForce =
                    (kGravitationalConstant * target.mass * source.mass) /
                    (distanceMeters * distanceMeters + epsilon * epsilon);
                const float accelerationMagnitude = gravitationalForce / target.mass;
                const glm::vec3 pairAcceleration = direction * accelerationMagnitude;
                totalAcceleration += pairAcceleration;
                velocity.x += pairAcceleration.x / 96.0f;
                velocity.y += pairAcceleration.y / 96.0f;
                velocity.z += pairAcceleration.z / 96.0f;
                if (source.radius + target.radius > distance)
                {
                    velocity *= -0.2f;
                }
            }
        }

        nextState[targetIndex].velocity = velocity;
        nextState[targetIndex].position = target.position + velocity / 94.0f;
        accelerations.push_back({target.id, totalAcceleration});
    }

    objects = std::move(nextState);
    return accelerations;
}

void ValidateGpuState(const std::vector<ReferenceObject> &referenceState,
                      const std::vector<AccelerationSample> &referenceAccelerations, size_t step)
{
    const auto *gpuState =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    const auto *gpuAccelerations =
        gpuMemoryManager.Get(BufferRole::ObjectAcceleration).MappedPtr<GpuObjectAcceleration>();
    const auto *gpuControls =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    Require(gpuState != nullptr && gpuAccelerations != nullptr && gpuControls != nullptr,
            "Expected persistent mappings for GPU validation");

    std::unordered_map<std::uint32_t, size_t> gpuSlotById;
    for (size_t slot = 0; slot < objs.size(); ++slot)
    {
        Require(gpuSlotById.emplace(gpuControls[slot].id, slot).second,
                "Duplicate GPU object ID in control buffer");
    }

    std::unordered_map<std::uint32_t, const ReferenceObject *> referenceById;
    for (const ReferenceObject &object : referenceState)
    {
        referenceById.emplace(object.id, &object);
    }

    std::unordered_map<std::uint32_t, glm::vec3> accelerationById;
    for (const AccelerationSample &sample : referenceAccelerations)
    {
        accelerationById.emplace(sample.id, sample.acceleration);
    }

    float maxAccelerationError = 0.0f;
    float maxRelativeAccelerationError = 0.0f;
    float maxPositionError = 0.0f;
    float maxVelocityError = 0.0f;
    for (const auto &[objectId, referenceObject] : referenceById)
    {
        const auto gpuSlot = gpuSlotById.find(objectId);
        Require(gpuSlot != gpuSlotById.end(), "GPU state is missing a reference object ID");
        const size_t slot = gpuSlot->second;
        const glm::vec3 actualPosition = glm::vec3(gpuState[slot].position);
        const glm::vec3 actualVelocity = glm::vec3(gpuState[slot].velocity);
        const glm::vec3 actualAcceleration = glm::vec3(gpuAccelerations[slot].acceleration);
        const glm::vec3 expectedAcceleration = accelerationById.at(objectId);

        const float accelerationError = glm::length(actualAcceleration - expectedAcceleration);
        const float expectedAccelerationLength = glm::length(expectedAcceleration);
        maxAccelerationError = std::max(maxAccelerationError, accelerationError);
        maxRelativeAccelerationError =
            std::max(maxRelativeAccelerationError,
                     accelerationError / std::max(expectedAccelerationLength, 1.0e-6f));
        maxPositionError =
            std::max(maxPositionError, glm::length(actualPosition - referenceObject->position));
        maxVelocityError =
            std::max(maxVelocityError, glm::length(actualVelocity - referenceObject->velocity));
    }

    std::cout << "GPU/CPU oracle checkpoint=" << step
              << ", max acceleration error=" << maxAccelerationError
              << ", max relative acceleration error=" << maxRelativeAccelerationError
              << ", max position error=" << maxPositionError
              << ", max velocity error=" << maxVelocityError << std::endl;
    Require(maxAccelerationError <= 0.05f, "GPU acceleration exceeded validation tolerance");
    Require(maxPositionError <= 0.002f, "GPU position exceeded validation tolerance");
    Require(maxVelocityError <= 0.02f, "GPU velocity exceeded validation tolerance");
}

void ShutdownValidation(GLFWwindow *window, GLuint nbodyProgram, GLuint integrateProgram)
{
    for (Object &object : objs)
    {
        glDeleteVertexArrays(1, &object.VAO);
        glDeleteBuffers(1, &object.VBO);
    }
    gpuMemoryManager.Shutdown();
    glDeleteProgram(nbodyProgram);
    glDeleteProgram(integrateProgram);
    if (window != nullptr)
    {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
    objs.clear();
}
} // namespace

int main()
{
    GLFWwindow *window = nullptr;
    GLuint nbodyProgram = 0;
    GLuint integrateProgram = 0;
    try
    {
        Require(glfwInit() == GLFW_TRUE, "Unable to initialize GLFW for GPU validation");
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(16, 16, "Phase 2 GPU validation", nullptr, nullptr);
        Require(window != nullptr, "Unable to create a hidden OpenGL 4.4 context");
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        Require(glewInit() == GLEW_OK, "Unable to initialize GLEW for GPU validation");
        (void)glGetError();
        Require(GLEW_VERSION_4_4 == GL_TRUE, "GPU validation requires OpenGL 4.4");

        objs.emplace_back(glm::vec3(-2.0f, 0.0f, 0.0f), glm::vec3(0.0f), 1.0e20f, 1000.0f);
        objs.emplace_back(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(0.0f), 1.0e20f, 1000.0f);
        std::vector<ReferenceObject> referenceState;
        referenceState.reserve(objs.size());
        for (const Object &object : objs)
        {
            referenceState.push_back({object.id, object.position, object.velocity, object.mass,
                                      object.radius, object.initializing});
        }

        ComputePipeline({glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)});
        objectGpuState(0, objs.size());
        nbodyProgram = CreateComputeProgramFromFile("shaders/nbody.comp");
        integrateProgram = CreateComputeProgramFromFile("shaders/integrate.comp");

        constexpr float epsilon = 10.0f;
        constexpr size_t validationSteps = 8;
        for (size_t step = 1; step <= validationSteps; ++step)
        {
            const std::vector<AccelerationSample> expectedAccelerations =
                AdvanceReference(referenceState, epsilon);
            const GLuint currentHandleBefore =
                gpuMemoryManager.Get(BufferRole::CurrentState).Handle();
            const GLuint nextHandleBefore = gpuMemoryManager.Get(BufferRole::NextState).Handle();

            RunGpuStateTransition(nbodyProgram, integrateProgram, objs.size(), epsilon, false);
            gpuMemoryManager.FenceGpuCompletion();
            gpuMemoryManager.WaitForCpuWrite();

            Require(gpuMemoryManager.Get(BufferRole::CurrentState).Handle() == nextHandleBefore &&
                        gpuMemoryManager.Get(BufferRole::NextState).Handle() == currentHandleBefore,
                    "CurrentState/NextState did not exchange physical buffers");
            GLint currentBinding = 0;
            GLint nextBinding = 0;
            glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 0, &currentBinding);
            glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 1, &nextBinding);
            Require(static_cast<GLuint>(currentBinding) == nextHandleBefore &&
                        static_cast<GLuint>(nextBinding) == currentHandleBefore,
                    "State swap did not rebind SSBO slots 0 and 1");

            ValidateGpuState(referenceState, expectedAccelerations, step);
        }

        const GLenum glError = glGetError();
        Require(glError == GL_NO_ERROR, "OpenGL reported an error during GPU validation");
        ShutdownValidation(window, nbodyProgram, integrateProgram);
        std::cout << "Phase 2 GPU validation passed for acceleration and 8 state transitions."
                  << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Phase 2 GPU validation failed: " << error.what() << std::endl;
        ShutdownValidation(window, nbodyProgram, integrateProgram);
        return 1;
    }
}