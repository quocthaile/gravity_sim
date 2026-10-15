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

struct RenderMetrics
{
    std::chrono::nanoseconds cpuSubmitTime{};
    std::uint64_t gpuRenderTimeNanoseconds = 0;
    size_t drawCalls = 0;
};

void ValidateInstancedRendering(GLuint objectRenderingProgram, GLuint framebuffer,
                                const std::vector<ReferenceObject> &referenceState,
                                size_t instanceCount, RenderMetrics &metrics)
{
    constexpr GLsizei targetWidth = 128;
    constexpr GLsizei targetHeight = 128;
    const GLint viewLocation = glGetUniformLocation(objectRenderingProgram, "view");
    const GLint projectionLocation = glGetUniformLocation(objectRenderingProgram, "projection");
    Require(viewLocation >= 0 && projectionLocation >= 0,
            "Object vertex shader is missing a camera uniform");

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glViewport(0, 0, targetWidth, targetHeight);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(objectRenderingProgram);
    const glm::mat4 identity(1.0f);
    glUniformMatrix4fv(viewLocation, 1, GL_FALSE, glm::value_ptr(identity));
    glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, glm::value_ptr(identity));

    Require(instanceCount <= objs.size(), "Requested instance count exceeds live objects");
    std::vector<Object> staleCpuObjects(objs.begin(), objs.begin() + instanceCount);
    for (Object &object : staleCpuObjects)
    {
        object.position = glm::vec3(10000.0f + object.id, -20000.0f, 30000.0f);
    }

    GLuint timerQuery = 0;
    glGenQueries(1, &timerQuery);
    glBeginQuery(GL_TIME_ELAPSED, timerQuery);
    const auto cpuStart = std::chrono::steady_clock::now();
    DrawObjects(objectRenderingProgram, staleCpuObjects);
    metrics.cpuSubmitTime += std::chrono::steady_clock::now() - cpuStart;
    glEndQuery(GL_TIME_ELAPSED);
    GLuint64 gpuTimeNanoseconds = 0;
    glGetQueryObjectui64v(timerQuery, GL_QUERY_RESULT, &gpuTimeNanoseconds);
    glDeleteQueries(1, &timerQuery);
    metrics.gpuRenderTimeNanoseconds += gpuTimeNanoseconds;
    ++metrics.drawCalls;

    const auto readPixel = [](GLint x, GLint y)
    {
        GLubyte color[4] = {};
        glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, color);
        return glm::u8vec4(color[0], color[1], color[2], color[3]);
    };
    const auto worldToPixel = [](float coordinate, GLsizei extent)
    { return static_cast<GLint>((coordinate + 1.0f) * 0.5f * static_cast<float>(extent)); };

    for (size_t index = 0; index < instanceCount; ++index)
    {
        const Object &object = objs[index];
        const auto reference = std::find_if(referenceState.begin(), referenceState.end(),
                                            [&object](const ReferenceObject &candidate)
                                            { return candidate.id == object.id; });
        Require(reference != referenceState.end(), "Missing object in render-position oracle");
        const glm::vec3 position = reference->position;
        const GLint centerX = worldToPixel(position.x, targetWidth);
        const GLint centerY = worldToPixel(position.y, targetHeight);
        const GLint innerX = worldToPixel(position.x + reference->radius * 0.5f, targetWidth);
        const GLint outsideX = worldToPixel(position.x + reference->radius * 1.4f, targetWidth);
        const glm::u8vec4 centerColor = readPixel(centerX, centerY);
        const glm::u8vec4 innerColor = readPixel(innerX, centerY);
        const glm::u8vec4 outsideColor = readPixel(outsideX, centerY);
        const auto matchesObjectColor = [&object](const glm::u8vec4 &pixel)
        {
            const glm::vec3 actual(pixel.r, pixel.g, pixel.b);
            const glm::vec3 expected(object.color);
            return glm::length(actual / 255.0f - expected) < 0.15f;
        };
        Require(matchesObjectColor(centerColor),
                "Instanced object color or GPU position did not match the expected object");
        Require(matchesObjectColor(innerColor),
                "Instanced object visual scale is smaller than its GPU radius");
        Require(outsideColor.r < 20 && outsideColor.g < 20 && outsideColor.b < 20,
                "Instanced object visual scale is larger than its GPU radius");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 16, 16);
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
    gpuMemoryManager.WaitForGpuCompletion();
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
                        GLuint integrateProgram, GLuint framebuffer, GLuint colorTexture)
{
    DestroyObjectRenderingResources();
    glDeleteFramebuffers(1, &framebuffer);
    glDeleteTextures(1, &colorTexture);
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
    GLuint validationFramebuffer = 0;
    GLuint validationColorTexture = 0;
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

        objs.emplace_back(glm::vec3(-0.55f, 0.0f, 0.0f), glm::vec3(0.0f), 1.0e16f, 1000.0f);
        objs.emplace_back(glm::vec3(0.55f, 0.0f, 0.0f), glm::vec3(0.0f), 1.0e16f, 1000.0f);
        objs[0].color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        objs[1].color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        std::vector<ReferenceObject> referenceState;
        referenceState.reserve(objs.size());
        for (const Object &object : objs)
        {
            referenceState.push_back({object.id, object.position, object.velocity, object.mass,
                                      object.radius, object.initializing});
        }

        const std::string vertexSource = LoadShaderSource("shaders/objects.vert");
        const std::string fragmentSource = LoadShaderSource("shaders/objects.frag");
        renderingProgram = CreateGraphicsProgram(vertexSource.c_str(), fragmentSource.c_str());
        GLint linkStatus = GL_FALSE;
        glGetProgramiv(renderingProgram, GL_LINK_STATUS, &linkStatus);
        Require(linkStatus == GL_TRUE, "Object rendering shader failed to link");

        glGenTextures(1, &validationColorTexture);
        glBindTexture(GL_TEXTURE_2D, validationColorTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 128, 128, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glGenFramebuffers(1, &validationFramebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, validationFramebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               validationColorTexture, 0);
        Require(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
                "Unable to create the object-render validation framebuffer");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        InitializeObjectRenderingResources();

        ComputePipeline({glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)});
        objectGpuState(0, objs.size());
        RenderMetrics renderMetrics;
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 1,
                                   renderMetrics);
        Object invalidStateObject = objs.front();
        invalidStateObject.gpuStateIndex =
            gpuMemoryManager.Get(BufferRole::CurrentState).LogicalCount();
        bool invalidStateRejected = false;
        try
        {
            DrawObjects(renderingProgram, std::vector<Object>{invalidStateObject});
        }
        catch (const std::logic_error &)
        {
            invalidStateRejected = true;
        }
        Require(invalidStateRejected, "DrawObjects accepted an out-of-range GPU state slot");
        DrawObjects(renderingProgram, {});
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
            ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState,
                                       objs.size(), renderMetrics);
            ValidateGpuState(referenceState, expectedAccelerations, step);
        }

        objs.emplace_back(glm::vec3(5.0f, 1.0f, -3.0f), glm::vec3(0.0f), 1.0e20f, 1000.0f);
        objs.back().position = glm::vec3(0.0f);
        objs.back().mass = 1.0e16f;
        objs.back().density = 1000.0f;
        objs.back().radius =
            pow(((3 * objs.back().mass / objs.back().density) / (4 * 3.14159265359)),
                (1.0f / 3.0f)) /
            100000.0f;
        objs.back().color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
        objs.back().initializing = true;
        SynchronizeObjectStateBuffers(objectCount, objs.size());
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
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState,
                                   objs.size(), renderMetrics);
        ValidateGpuState(referenceState, expectedAccelerations, validationSteps + 1);

        const std::vector<glm::vec3> pausedPositions = [&referenceState]
        {
            std::vector<glm::vec3> positions;
            positions.reserve(referenceState.size());
            for (const ReferenceObject &object : referenceState)
            {
                positions.push_back(object.position);
            }
            return positions;
        }();
        RunGpuStateTransition(nbodyProgram, integrateProgram, objectCount, epsilon, true);
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState,
                                   objs.size(), renderMetrics);
        gpuMemoryManager.WaitForGpuCompletion();
        const auto *pausedState =
            gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
        Require(pausedState != nullptr, "CurrentState mapping unavailable after paused step");
        for (size_t index = 0; index < pausedPositions.size(); ++index)
        {
            Require(glm::length(glm::vec3(pausedState[index].position) - pausedPositions[index]) <=
                        0.002f,
                    "Paused simulation changed an object's rendered position");
        }

        const std::vector<AccelerationSample> resumedAccelerations =
            AdvanceReference(referenceState, epsilon);
        RunGpuStateTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false);
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState,
                                   objs.size(), renderMetrics);
        ValidateGpuState(referenceState, resumedAccelerations, validationSteps + 2);

        const double averageWaitMicroseconds =
            std::chrono::duration<double, std::micro>(totalFenceWait).count() / timingSamples;
        const double averageReadMicroseconds =
            std::chrono::duration<double, std::micro>(totalMappedRead).count() / timingSamples;
        std::cout << "Synchronization observation (" << timingSamples
                  << " hidden-context samples): average fence wait=" << averageWaitMicroseconds
                  << " us, mapped position read=" << averageReadMicroseconds
                  << " us per state array, checksum=" << positionChecksum << std::endl;
        std::cout
            << "Instanced rendering observation (" << renderMetrics.drawCalls
            << " draws): average CPU submission="
            << std::chrono::duration<double, std::micro>(renderMetrics.cpuSubmitTime).count() /
                   renderMetrics.drawCalls
            << " us, average GPU draw="
            << static_cast<double>(renderMetrics.gpuRenderTimeNanoseconds) /
                   renderMetrics.drawCalls / 1000.0
            << " us" << std::endl;

        const GLenum glError = glGetError();
        Require(glError == GL_NO_ERROR, "OpenGL reported an error during GPU validation");
        ShutdownValidation(window, renderingProgram, nbodyProgram, integrateProgram,
                           validationFramebuffer, validationColorTexture);
        std::cout << "GPU validation passed for rendering positions, lifecycle, pause/resume, and "
                     "11 state transitions."
                  << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Phase 2 GPU validation failed: " << error.what() << std::endl;
        ShutdownValidation(window, renderingProgram, nbodyProgram, integrateProgram,
                           validationFramebuffer, validationColorTexture);
        return 1;
    }
}