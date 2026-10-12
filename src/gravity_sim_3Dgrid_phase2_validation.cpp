#include "gravity_sim_3Dgrid_function.hpp"

#include <algorithm>
#include <chrono>
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

void ValidateDrawnPosition(GLuint renderingProgram, const Object &object,
                           const glm::vec3 &cpuPosition, const glm::vec3 &expectedPosition)
{
    const GLint modelLocation = glGetUniformLocation(renderingProgram, "model");
    const GLint objectColorLocation = glGetUniformLocation(renderingProgram, "objectColor");
    Require(modelLocation >= 0 && objectColorLocation >= 0,
            "Rendering shader is missing an object uniform");

    Object drawObject = object;
    drawObject.position = cpuPosition;
    glUseProgram(renderingProgram);
    DrawObjects(std::vector<Object>{drawObject}, modelLocation, objectColorLocation);

    GLfloat modelValues[16] = {};
    GLfloat colorValues[4] = {};
    glGetUniformfv(renderingProgram, modelLocation, modelValues);
    glGetUniformfv(renderingProgram, objectColorLocation, colorValues);
    const glm::vec3 drawnPosition(modelValues[12], modelValues[13], modelValues[14]);
    const glm::vec4 drawnColor(colorValues[0], colorValues[1], colorValues[2], colorValues[3]);
    Require(glm::length(drawnPosition - expectedPosition) <= 0.002f,
            "DrawObjects did not use the expected object position");
    Require(glm::length(drawnColor - object.color) <= 1.0e-6f,
            "DrawObjects did not preserve CPU-side object color");
}

void ValidateDrawnPositions(GLuint renderingProgram,
                            const std::vector<ReferenceObject> &referenceState)
{
    for (const Object &object : objs)
    {
        const auto reference = std::find_if(referenceState.begin(), referenceState.end(),
                                            [&object](const ReferenceObject &candidate)
                                            { return candidate.id == object.id; });
        Require(reference != referenceState.end(), "Missing object in render-position oracle");
        ValidateDrawnPosition(renderingProgram, object,
                              glm::vec3(10000.0f + object.id, -20000.0f, 30000.0f),
                              reference->position);
    }
}

float MeasureFenceAndMappedPositionRead(size_t objectCount, std::chrono::nanoseconds &waitTotal,
                                        std::chrono::nanoseconds &readTotal)
{
    const auto waitStart = std::chrono::steady_clock::now();
    gpuMemoryManager.WaitForGpuCompletion();
    const auto readStart = std::chrono::steady_clock::now();
    waitTotal += readStart - waitStart;

    const auto *state = gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    Require(state != nullptr, "CurrentState mapping is unavailable during timing observation");
    float positionChecksum = 0.0f;
    for (size_t index = 0; index < objectCount; ++index)
    {
        positionChecksum +=
            state[index].position.x + state[index].position.y + state[index].position.z;
    }
    readTotal += std::chrono::steady_clock::now() - readStart;
    return positionChecksum;
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
    Require(gpuState != nullptr && gpuAccelerations != nullptr,
            "Expected persistent mappings for GPU validation");

    std::unordered_map<std::uint32_t, size_t> gpuSlotById;
    const size_t stateCount = gpuMemoryManager.Get(BufferRole::CurrentState).LogicalCount();
    for (const Object &object : objs)
    {
        Require(object.gpuStateIndex < stateCount,
                "Object-to-GPU-state mapping exceeds the logical state count");
        Require(gpuSlotById.emplace(object.id, object.gpuStateIndex).second,
                "Duplicate object ID in GPU state mapping");
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

void ShutdownValidation(GLFWwindow *window, GLuint renderingProgram, GLuint nbodyProgram,
                        GLuint integrateProgram)
{
    for (Object &object : objs)
    {
        glDeleteVertexArrays(1, &object.VAO);
        glDeleteBuffers(1, &object.VBO);
    }
    gpuMemoryManager.Shutdown();
    glDeleteProgram(renderingProgram);
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
    GLuint renderingProgram = 0;
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

        const std::string vertexSource = LoadShaderSource("shaders/grid.vert");
        const std::string fragmentSource = LoadShaderSource("shaders/grid.frag");
        renderingProgram = CreateGraphicsProgram(vertexSource.c_str(), fragmentSource.c_str());
        const GLint modelLocation = glGetUniformLocation(renderingProgram, "model");
        const GLint objectColorLocation = glGetUniformLocation(renderingProgram, "objectColor");
        Object cpuOnlyObject = objs.front();
        const glm::vec3 fallbackPosition(17.0f, -23.0f, 41.0f);
        ValidateDrawnPosition(renderingProgram, cpuOnlyObject, fallbackPosition, fallbackPosition);
        DrawObjects(std::vector<Object>{}, modelLocation, objectColorLocation);

        ComputePipeline({glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)});
        objectGpuState(0, objs.size());
        Object invalidStateObject = objs.front();
        invalidStateObject.gpuStateIndex =
            gpuMemoryManager.Get(BufferRole::CurrentState).LogicalCount();
        bool invalidStateRejected = false;
        try
        {
            DrawObjects(std::vector<Object>{invalidStateObject}, modelLocation,
                        objectColorLocation);
        }
        catch (const std::out_of_range &)
        {
            invalidStateRejected = true;
        }
        Require(invalidStateRejected, "DrawObjects accepted an out-of-range GPU state slot");
        nbodyProgram = CreateComputeProgramFromFile("shaders/nbody.comp");
        integrateProgram = CreateComputeProgramFromFile("shaders/integrate.comp");

        constexpr float epsilon = 10.0f;
        constexpr size_t validationSteps = 8;
        size_t objectCount = objs.size();
        std::chrono::nanoseconds totalFenceWait{};
        std::chrono::nanoseconds totalMappedRead{};
        size_t timingSamples = 0;
        float positionChecksum = 0.0f;
        for (size_t step = 1; step <= validationSteps; ++step)
        {
            const std::vector<AccelerationSample> expectedAccelerations =
                AdvanceReference(referenceState, epsilon);
            const GLuint currentHandleBefore =
                gpuMemoryManager.Get(BufferRole::CurrentState).Handle();
            const GLuint nextHandleBefore = gpuMemoryManager.Get(BufferRole::NextState).Handle();

            RunGpuStateTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false);
            gpuMemoryManager.FenceGpuCompletion();

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

            if (step > 1)
            {
                positionChecksum +=
                    MeasureFenceAndMappedPositionRead(objectCount, totalFenceWait, totalMappedRead);
                ++timingSamples;
            }
            ValidateDrawnPositions(renderingProgram, referenceState);
            ValidateGpuState(referenceState, expectedAccelerations, step);
        }

        objs.emplace_back(glm::vec3(5.0f, 1.0f, -3.0f), glm::vec3(0.0f), 1.0e20f, 1000.0f);
        objs.back().initializing = true;
        SynchronizeObjectStateCount(objectCount, objs.size());
        Require(objs.back().gpuStateIndex == objectCount - 1,
                "Late-created object was not assigned a GPU state slot");
        referenceState.push_back({objs.back().id, objs.back().position, objs.back().velocity,
                                  objs.back().mass, objs.back().radius, objs.back().initializing});
        const std::vector<AccelerationSample> expectedAccelerations =
            AdvanceReference(referenceState, epsilon);
        const GLuint currentHandleBefore = gpuMemoryManager.Get(BufferRole::CurrentState).Handle();
        const GLuint nextHandleBefore = gpuMemoryManager.Get(BufferRole::NextState).Handle();
        RunGpuStateTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false);
        gpuMemoryManager.FenceGpuCompletion();
        Require(gpuMemoryManager.Get(BufferRole::CurrentState).Handle() == nextHandleBefore &&
                    gpuMemoryManager.Get(BufferRole::NextState).Handle() == currentHandleBefore,
                "CurrentState/NextState did not exchange after adding an object");
        positionChecksum +=
            MeasureFenceAndMappedPositionRead(objectCount, totalFenceWait, totalMappedRead);
        ++timingSamples;
        ValidateDrawnPositions(renderingProgram, referenceState);
        ValidateGpuState(referenceState, expectedAccelerations, validationSteps + 1);

        const double averageWaitMicroseconds =
            std::chrono::duration<double, std::micro>(totalFenceWait).count() / timingSamples;
        const double averageReadMicroseconds =
            std::chrono::duration<double, std::micro>(totalMappedRead).count() / timingSamples;
        std::cout << "Synchronization observation (" << timingSamples
                  << " hidden-context samples): average fence wait=" << averageWaitMicroseconds
                  << " us, mapped position read=" << averageReadMicroseconds
                  << " us per state array, checksum=" << positionChecksum << std::endl;

        const GLenum glError = glGetError();
        Require(glError == GL_NO_ERROR, "OpenGL reported an error during GPU validation");
        ShutdownValidation(window, renderingProgram, nbodyProgram, integrateProgram);
        std::cout
            << "GPU validation passed for rendering positions, lifecycle, and 9 state transitions."
            << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Phase 2 GPU validation failed: " << error.what() << std::endl;
        ShutdownValidation(window, renderingProgram, nbodyProgram, integrateProgram);
        return 1;
    }
}