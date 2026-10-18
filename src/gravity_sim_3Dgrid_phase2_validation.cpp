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
    bool launched;
};

struct AccelerationSample
{
    std::uint32_t id;
    glm::vec3 acceleration;
    size_t collisionCount;
};

struct ValidationMetrics
{
    size_t checkpoints = 0;
    size_t maxCollisionCountError = 0;
    float maxAccelerationError = 0.0f;
    float maxRelativeAccelerationError = 0.0f;
    float maxPositionError = 0.0f;
    float maxVelocityError = 0.0f;
};

void Require(bool condition, const std::string &message);

template <typename T> std::vector<T> CopyMappedBuffer(BufferRole role, size_t count)
{
    const T *data = gpuMemoryManager.Get(role).MappedPtr<T>();
    Require(data != nullptr, "Expected persistent mapping while preserving object state");
    return std::vector<T>(data, data + count);
}

void ValidateObjectBufferLayout(size_t expectedCount)
{
    constexpr std::array objectRoles = {BufferRole::CurrentState,   BufferRole::NextState,
                                        BufferRole::ObjectPhysical, BufferRole::ObjectDerived,
                                        BufferRole::ObjectControl,  BufferRole::ObjectAcceleration,
                                        BufferRole::ObjectRender,   BufferRole::ObjectCollision};
    for (const BufferRole role : objectRoles)
    {
        const GpuBuffer &buffer = gpuMemoryManager.Get(role);
        Require(buffer.LogicalCount() == expectedCount,
                "Per-object logical counts diverged after append or resize");
        Require(buffer.Capacity() == objectStateCapacity,
                "Per-object buffer capacities diverged after append or resize");
        GLint boundHandle = 0;
        glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, BindingFor(role), &boundHandle);
        Require(static_cast<GLuint>(boundHandle) == buffer.Handle(),
                "Per-object buffer is not bound at its semantic SSBO binding");
    }
}

void RunCountedTransition(GLuint nbodyProgram, GLuint integrateProgram, size_t objectCount,
                          float epsilon, bool paused, size_t &transitionCount)
{
    RunGpuStateTransition(nbodyProgram, integrateProgram, objectCount, epsilon, paused);
    ++transitionCount;
}

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
                                                 float epsilon, bool paused)
{
    const std::vector<ReferenceObject> currentState = objects;
    std::vector<ReferenceObject> nextState = currentState;
    std::vector<glm::vec3> totalAccelerations(currentState.size(), glm::vec3(0.0f));
    std::vector<size_t> collisionCounts(currentState.size(), 0);
    std::vector<AccelerationSample> accelerations;
    accelerations.reserve(currentState.size());

    // Compute snapshot gravity and overlap metadata separately; integration applies the response.
    for (size_t targetIndex = 0; targetIndex < currentState.size(); ++targetIndex)
    {
        const ReferenceObject &target = currentState[targetIndex];
        if (target.initializing)
        {
            continue;
        }

        for (size_t sourceIndex = 0; sourceIndex < currentState.size(); ++sourceIndex)
        {
            const ReferenceObject &source = currentState[sourceIndex];
            if (sourceIndex == targetIndex || source.initializing)
            {
                continue;
            }

            const glm::vec3 displacement = source.position - target.position;
            const float distance = glm::length(displacement);
            if (distance <= 0.0f)
            {
                continue;
            }

            const float distanceMeters = distance * 1000.0f;
            const float softenedDistanceSquared =
                distanceMeters * distanceMeters + epsilon * epsilon;
            const double gravitationalForce =
                (kGravitationalConstant * static_cast<double>(target.mass) * source.mass) /
                static_cast<double>(softenedDistanceSquared);
            const float accelerationMagnitude =
                static_cast<float>(gravitationalForce / static_cast<double>(target.mass));
            const glm::vec3 pairAcceleration = displacement / distance * accelerationMagnitude;
            const bool colliding = source.radius + target.radius > distance;
            totalAccelerations[targetIndex] += pairAcceleration;
            collisionCounts[targetIndex] += static_cast<size_t>(colliding);
        }
    }

    for (size_t targetIndex = 0; targetIndex < currentState.size(); ++targetIndex)
    {
        const ReferenceObject &target = currentState[targetIndex];
        glm::vec3 velocity = target.velocity;
        glm::vec3 position = target.position;
        if (!target.initializing && !paused)
        {
            velocity += totalAccelerations[targetIndex] / 96.0f;
            float collisionScale = 1.0f;
            for (size_t overlapIndex = 0; overlapIndex < collisionCounts[targetIndex];
                 ++overlapIndex)
            {
                collisionScale *= -0.2f;
            }
            velocity *= collisionScale;
            position += velocity / 94.0f;
        }

        nextState[targetIndex].velocity = velocity;
        nextState[targetIndex].position = position;
        accelerations.push_back(
            {target.id, totalAccelerations[targetIndex], collisionCounts[targetIndex]});
    }

    objects = std::move(nextState);
    return accelerations;
}

void ValidateGpuState(const std::vector<ReferenceObject> &referenceState,
                      const std::vector<AccelerationSample> &referenceAccelerations, size_t step,
                      ValidationMetrics &metrics)
{
    constexpr float accelerationAbsoluteTolerance = 1.0e-4f; // m/s^2
    constexpr float accelerationRelativeTolerance = 1.0e-3f;
    constexpr float positionTolerance = 2.0e-3f; // simulation position units
    constexpr float velocityTolerance = 2.0e-2f; // velocity units
    gpuMemoryManager.WaitForGpuCompletion();
    const auto *gpuState =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    const auto *gpuAccelerations =
        gpuMemoryManager.Get(BufferRole::ObjectAcceleration).MappedPtr<GpuObjectAcceleration>();
    const auto *gpuCollisions =
        gpuMemoryManager.Get(BufferRole::ObjectCollision).MappedPtr<GpuObjectCollision>();
    Require(gpuState != nullptr && gpuAccelerations != nullptr && gpuCollisions != nullptr,
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

    std::unordered_map<std::uint32_t, AccelerationSample> accelerationById;
    for (const AccelerationSample &sample : referenceAccelerations)
    {
        accelerationById.emplace(sample.id, sample);
    }

    float maxAccelerationError = 0.0f;
    float maxRelativeAccelerationError = 0.0f;
    float maxPositionError = 0.0f;
    float maxVelocityError = 0.0f;
    size_t maxCollisionCountError = 0;
    size_t relativeAccelerationSamples = 0;
    for (const auto &[objectId, referenceObject] : referenceById)
    {
        const auto gpuSlot = gpuSlotById.find(objectId);
        Require(gpuSlot != gpuSlotById.end(), "GPU state is missing a reference object ID");
        const size_t slot = gpuSlot->second;
        const glm::vec3 actualPosition = glm::vec3(gpuState[slot].position);
        const glm::vec3 actualVelocity = glm::vec3(gpuState[slot].velocity);
        const GpuObjectAcceleration &actualAcceleration = gpuAccelerations[slot];
        const GpuObjectCollision &actualCollision = gpuCollisions[slot];
        const AccelerationSample &expected = accelerationById.at(objectId);

        const glm::vec3 actualAccelerationVector(actualAcceleration.acceleration);
        const float accelerationError =
            glm::length(actualAccelerationVector - expected.acceleration);
        const float expectedAccelerationLength = glm::length(expected.acceleration);
        maxAccelerationError = std::max(maxAccelerationError, accelerationError);
        if (expectedAccelerationLength >= 1.0e-6f)
        {
            maxRelativeAccelerationError = std::max(maxRelativeAccelerationError,
                                                    accelerationError / expectedAccelerationLength);
            ++relativeAccelerationSamples;
        }
        const size_t collisionCountError =
            actualCollision.overlapCount > expected.collisionCount
                ? actualCollision.overlapCount - expected.collisionCount
                : expected.collisionCount - actualCollision.overlapCount;
        maxCollisionCountError = std::max(maxCollisionCountError, collisionCountError);
        maxPositionError =
            std::max(maxPositionError, glm::length(actualPosition - referenceObject->position));
        maxVelocityError =
            std::max(maxVelocityError, glm::length(actualVelocity - referenceObject->velocity));
    }

    std::cout << "GPU/snapshot-oracle checkpoint=" << step
              << ", max acceleration error=" << maxAccelerationError << " m/s^2"
              << ", max relative acceleration error=" << maxRelativeAccelerationError
              << " (samples=" << relativeAccelerationSamples << ')'
              << ", max collision-count error=" << maxCollisionCountError
              << ", max position error=" << maxPositionError << " units"
              << ", max velocity error=" << maxVelocityError << " units" << std::endl;
    Require(maxAccelerationError <= accelerationAbsoluteTolerance,
            "GPU acceleration exceeded absolute validation tolerance");
    if (relativeAccelerationSamples > 0)
    {
        Require(maxRelativeAccelerationError <= accelerationRelativeTolerance,
                "GPU acceleration exceeded relative validation tolerance");
    }
    Require(maxCollisionCountError == 0,
            "GPU collision overlap count did not match the snapshot oracle");
    Require(maxPositionError <= positionTolerance, "GPU position exceeded validation tolerance");
    Require(maxVelocityError <= velocityTolerance, "GPU velocity exceeded validation tolerance");
    ++metrics.checkpoints;
    metrics.maxCollisionCountError =
        std::max(metrics.maxCollisionCountError, maxCollisionCountError);
    metrics.maxAccelerationError = std::max(metrics.maxAccelerationError, maxAccelerationError);
    metrics.maxRelativeAccelerationError =
        std::max(metrics.maxRelativeAccelerationError, maxRelativeAccelerationError);
    metrics.maxPositionError = std::max(metrics.maxPositionError, maxPositionError);
    metrics.maxVelocityError = std::max(metrics.maxVelocityError, maxVelocityError);
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
#if defined(NDEBUG)
        constexpr const char *buildConfiguration = "Release";
#else
        constexpr const char *buildConfiguration = "Debug";
#endif
        std::cout << "Validation build configuration: " << buildConfiguration << std::endl;
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
        std::cout << "OpenGL context: requested=4.4, version="
                  << reinterpret_cast<const char *>(glGetString(GL_VERSION))
                  << ", renderer=" << reinterpret_cast<const char *>(glGetString(GL_RENDERER))
                  << ", vendor=" << reinterpret_cast<const char *>(glGetString(GL_VENDOR))
                  << std::endl;

        objs.emplace_back(glm::vec3(-0.55f, 0.0f, 0.0f), glm::vec3(0.0f), 1.0e16f, 1000.0f);
        objs.emplace_back(glm::vec3(0.55f, 0.0f, 0.0f), glm::vec3(0.0f), 1.0e16f, 1000.0f);
        objs[0].color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        objs[1].color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        std::vector<ReferenceObject> referenceState;
        referenceState.reserve(objs.size());
        for (const Object &object : objs)
        {
            referenceState.push_back({object.id, object.position, object.velocity, object.mass,
                                      object.radius, object.initializing, object.launched});
        }

        renderingProgram = CreateGraphicsProgram("shaders/objects.vert", "shaders/objects.frag");
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
        CreateUnitSphere();

        ComputePipeline({glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)});
        objectGpuState(0, objs.size());
        ValidateObjectBufferLayout(objs.size());
        RenderMetrics renderMetrics;
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 1,
                                   renderMetrics);
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 2,
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
        nbodyProgram = CreateComputeProgram("shaders/nbody.comp");
        integrateProgram = CreateComputeProgram("shaders/integrate.comp");

        constexpr float epsilon = 10.0f;
        const GLuint currentHandleBeforeZeroDispatch =
            gpuMemoryManager.Get(BufferRole::CurrentState).Handle();
        const GLuint nextHandleBeforeZeroDispatch =
            gpuMemoryManager.Get(BufferRole::NextState).Handle();
        RunGpuStateTransition(nbodyProgram, integrateProgram, 0, epsilon, false);
        Require(gpuMemoryManager.Get(BufferRole::CurrentState).Handle() ==
                        currentHandleBeforeZeroDispatch &&
                    gpuMemoryManager.Get(BufferRole::NextState).Handle() ==
                        nextHandleBeforeZeroDispatch,
                "Zero-object dispatch changed the ping-pong state roles");
        constexpr size_t validationSteps = 8;
        size_t objectCount = objs.size();
        size_t transitionCount = 0;
        ValidationMetrics validationMetrics;
        std::chrono::nanoseconds totalFenceWait{};
        std::chrono::nanoseconds totalMappedRead{};
        size_t timingSamples = 0;
        float positionChecksum = 0.0f;

        for (size_t objectIndex = 0; objectIndex < objs.size(); ++objectIndex)
        {
            objs[objectIndex].initializing = true;
            referenceState[objectIndex].initializing = true;
            SynchronizeObjectGpuProperties(objectIndex);
        }
        const std::vector<AccelerationSample> zeroInteractionAccelerations =
            AdvanceReference(referenceState, epsilon, false);
        RunCountedTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false,
                             transitionCount);
        ValidateGpuState(referenceState, zeroInteractionAccelerations, transitionCount,
                         validationMetrics);
        const auto *zeroInteractionCollision =
            gpuMemoryManager.Get(BufferRole::ObjectCollision).MappedPtr<GpuObjectCollision>();
        for (size_t index = 0; index < objectCount; ++index)
        {
            Require(glm::length(zeroInteractionAccelerations[index].acceleration) == 0.0f &&
                        zeroInteractionCollision[index].overlapCount == 0u,
                    "Initializing objects produced gravity or collision output");
            objs[index].initializing = false;
            referenceState[index].initializing = false;
            SynchronizeObjectGpuProperties(index);
        }

        for (size_t step = 1; step <= validationSteps; ++step)
        {
            const glm::vec3 initialVelocity = referenceState.front().velocity;
            const std::vector<AccelerationSample> expectedAccelerations =
                AdvanceReference(referenceState, epsilon, false);
            if (step == 1)
            {
                Require(expectedAccelerations.front().collisionCount == 0 &&
                            expectedAccelerations.back().collisionCount == 0,
                        "Separated two-body gravity scenario unexpectedly overlaps");
                Require(glm::length(expectedAccelerations.front().acceleration) > 0.0f,
                        "Two-body gravity scenario produced no acceleration");
            }
            const GLuint currentHandleBefore =
                gpuMemoryManager.Get(BufferRole::CurrentState).Handle();
            const GLuint nextHandleBefore = gpuMemoryManager.Get(BufferRole::NextState).Handle();

            RunCountedTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false,
                                 transitionCount);
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
            ValidateObjectBufferLayout(objectCount);
            ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 1,
                                       renderMetrics);
            ValidateGpuState(referenceState, expectedAccelerations, step, validationMetrics);
            if (step == 1)
            {
                const auto *state =
                    gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
                const auto *accelerations = gpuMemoryManager.Get(BufferRole::ObjectAcceleration)
                                                .MappedPtr<GpuObjectAcceleration>();
                const auto *collisions = gpuMemoryManager.Get(BufferRole::ObjectCollision)
                                             .MappedPtr<GpuObjectCollision>();
                const glm::vec3 velocityDelta = glm::vec3(state[0].velocity) - initialVelocity;
                Require(glm::dot(velocityDelta, expectedAccelerations[0].acceleration) > 0.0f,
                        "Two-body gravity did not accelerate along the force direction");
                Require(collisions[0].overlapCount == 0u && collisions[1].overlapCount == 0u,
                        "Separated bodies unexpectedly produced collision output");
                Require(sizeof(accelerations[0]) == 16,
                        "Acceleration record is not the required 16-byte layout");
            }
        }

        const size_t countBeforeAppend = objectCount;
        objs.emplace_back(glm::vec3(0.0f, 0.25f, 0.0f), glm::vec3(12.0f, -3.0f, 1.0f), 1.0e18f,
                          1000.0f);
        objs.back().color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
        objs.back().initializing = true;
        SynchronizeObjectStateBuffers(objectCount, objs.size());
        Require(objectStateCapacity == 256,
                "Appending one object unexpectedly resized the minimum-capacity buffers");
        Require(objs.back().gpuStateIndex == objectCount - 1,
                "Late-created object was not assigned a GPU state slot");
        ValidateObjectBufferLayout(objectCount);
        referenceState.push_back({objs.back().id, objs.back().position, objs.back().velocity,
                                  objs.back().mass, objs.back().radius, objs.back().initializing,
                                  objs.back().launched});
        std::vector<ReferenceObject> referenceWithoutInitializingObject(referenceState.begin(),
                                                                        referenceState.end() - 1);
        const std::vector<AccelerationSample> expectedWithoutInitializingObject =
            AdvanceReference(referenceWithoutInitializingObject, epsilon, false);
        const std::vector<AccelerationSample> expectedAccelerations =
            AdvanceReference(referenceState, epsilon, false);
        for (size_t index = 0; index < countBeforeAppend; ++index)
        {
            Require(glm::length(expectedAccelerations[index].acceleration -
                                expectedWithoutInitializingObject[index].acceleration) < 1.0e-7f,
                    "Initializing object affected the reference physics of an existing body");
        }
        RunCountedTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false,
                             transitionCount);
        gpuMemoryManager.FenceGpuCompletion();
        positionChecksum +=
            MeasureFenceAndMappedPositionRead(objectCount, totalFenceWait, totalMappedRead);
        ++timingSamples;
        ValidateObjectBufferLayout(objectCount);
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 1,
                                   renderMetrics);
        ValidateGpuState(referenceState, expectedAccelerations, transitionCount, validationMetrics);
        const auto *initializingAcceleration =
            gpuMemoryManager.Get(BufferRole::ObjectAcceleration).MappedPtr<GpuObjectAcceleration>();
        const auto *initializingCollision =
            gpuMemoryManager.Get(BufferRole::ObjectCollision).MappedPtr<GpuObjectCollision>();
        Require(glm::length(glm::vec3(
                    initializingAcceleration[objs.back().gpuStateIndex].acceleration)) < 1.0e-7f,
                "Initializing object received normal-physics acceleration");
        Require(initializingCollision[objs.back().gpuStateIndex].overlapCount == 0u,
                "Initializing object retained a stale collision count");
        const auto *initializingState =
            gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
        Require(
            glm::all(glm::equal(glm::vec3(initializingState[objs.back().gpuStateIndex].position),
                                objs.back().position)) &&
                glm::all(
                    glm::equal(glm::vec3(initializingState[objs.back().gpuStateIndex].velocity),
                               objs.back().velocity)),
            "Initializing object did not preserve its exact state");

        objs.back().initializing = false;
        objs.back().launched = true;
        SynchronizeObjectGpuProperties(objs.size() - 1);
        referenceState.back().initializing = false;
        referenceState.back().launched = true;
        const float firstLaunchDistance =
            glm::length(referenceState[0].position - referenceState.back().position);
        const float secondLaunchDistance =
            glm::length(referenceState[1].position - referenceState.back().position);
        Require(firstLaunchDistance > 0.0f &&
                    firstLaunchDistance < referenceState[0].radius + referenceState.back().radius &&
                    secondLaunchDistance > 0.0f &&
                    secondLaunchDistance < referenceState[1].radius + referenceState.back().radius,
                "Launched collision geometry does not overlap using constructor-computed radii");
        const glm::vec3 firstInitialVelocity = referenceState[0].velocity;
        const glm::vec3 secondInitialVelocity = referenceState[1].velocity;
        const glm::vec3 launchedInitialVelocity = referenceState.back().velocity;
        const std::vector<AccelerationSample> launchedAccelerations =
            AdvanceReference(referenceState, epsilon, false);
        Require(glm::length(launchedAccelerations.back().acceleration) > 1.0e-6f,
                "Launched object did not participate in the reference physics");
        Require(launchedAccelerations[0].collisionCount == 1 &&
                    launchedAccelerations[1].collisionCount == 1 &&
                    launchedAccelerations.back().collisionCount == 2,
                "Actual-radius geometry did not produce the expected one/two overlaps");
        RunCountedTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false,
                             transitionCount);
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 1,
                                   renderMetrics);
        ValidateGpuState(referenceState, launchedAccelerations, transitionCount, validationMetrics);
        const auto *launchedState =
            gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
        const glm::vec3 expectedEvenOverlapVelocity =
            (launchedInitialVelocity + launchedAccelerations.back().acceleration / 96.0f) *
            (-0.2f * -0.2f);
        Require(glm::length(glm::vec3(launchedState[objs.back().gpuStateIndex].velocity) -
                            expectedEvenOverlapVelocity) <= 1.0e-6f,
                "Two overlaps did not apply the positive even-parity collision scale");
        const glm::vec3 expectedOddOverlapVelocity =
            (firstInitialVelocity + launchedAccelerations[0].acceleration / 96.0f) * -0.2f;
        Require(glm::length(glm::vec3(launchedState[objs[0].gpuStateIndex].velocity) -
                            expectedOddOverlapVelocity) <= 1.0e-6f,
                "One overlap did not apply the negative odd-parity collision scale");
        (void)secondInitialVelocity;
        const auto *controls =
            gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
        Require(controls[objs.back().gpuStateIndex].initializing == 0u &&
                    controls[objs.back().gpuStateIndex].launched == 1u,
                "Initializing-to-launched control metadata was not synchronized");

        std::vector<glm::vec3> pausedPositions;
        std::vector<glm::vec3> pausedVelocities;
        pausedPositions.reserve(referenceState.size());
        pausedVelocities.reserve(referenceState.size());
        gpuMemoryManager.WaitForGpuCompletion();
        const auto *stateBeforePause =
            gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
        Require(stateBeforePause != nullptr, "CurrentState mapping unavailable before pause");
        for (const Object &object : objs)
        {
            pausedPositions.emplace_back(stateBeforePause[object.gpuStateIndex].position);
            pausedVelocities.emplace_back(stateBeforePause[object.gpuStateIndex].velocity);
        }
        const std::vector<AccelerationSample> pausedAccelerations =
            AdvanceReference(referenceState, epsilon, true);
        Require(glm::length(pausedAccelerations.back().acceleration) > 1.0e-6f,
                "Paused acceleration diagnostic was unexpectedly empty");
        RunCountedTransition(nbodyProgram, integrateProgram, objectCount, epsilon, true,
                             transitionCount);
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 1,
                                   renderMetrics);
        gpuMemoryManager.WaitForGpuCompletion();
        const auto *pausedState =
            gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
        Require(pausedState != nullptr, "CurrentState mapping unavailable after paused step");
        for (size_t index = 0; index < pausedPositions.size(); ++index)
        {
            const size_t slot = objs[index].gpuStateIndex;
            Require(
                glm::all(glm::equal(glm::vec3(pausedState[slot].position), pausedPositions[index])),
                "Paused simulation changed an object's position");
            Require(glm::all(
                        glm::equal(glm::vec3(pausedState[slot].velocity), pausedVelocities[index])),
                    "Paused simulation changed an object's velocity");
        }
        ValidateGpuState(referenceState, pausedAccelerations, transitionCount, validationMetrics);

        const std::vector<AccelerationSample> resumedAccelerations =
            AdvanceReference(referenceState, epsilon, false);
        RunCountedTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false,
                             transitionCount);
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 1,
                                   renderMetrics);
        ValidateGpuState(referenceState, resumedAccelerations, transitionCount, validationMetrics);

        const size_t countBeforeGrowth = objectCount;
        const size_t capacityBeforeGrowth = objectStateCapacity;
        const auto currentBeforeGrowth =
            CopyMappedBuffer<GpuObjectState>(BufferRole::CurrentState, countBeforeGrowth);
        const auto nextBeforeGrowth =
            CopyMappedBuffer<GpuObjectState>(BufferRole::NextState, countBeforeGrowth);
        const auto physicalBeforeGrowth =
            CopyMappedBuffer<GpuObjectPhysical>(BufferRole::ObjectPhysical, countBeforeGrowth);
        const auto derivedBeforeGrowth =
            CopyMappedBuffer<GpuObjectDerived>(BufferRole::ObjectDerived, countBeforeGrowth);
        const auto controlBeforeGrowth =
            CopyMappedBuffer<GpuObjectControl>(BufferRole::ObjectControl, countBeforeGrowth);
        const auto accelerationBeforeGrowth = CopyMappedBuffer<GpuObjectAcceleration>(
            BufferRole::ObjectAcceleration, countBeforeGrowth);
        const auto collisionBeforeGrowth =
            CopyMappedBuffer<GpuObjectCollision>(BufferRole::ObjectCollision, countBeforeGrowth);
        const auto renderBeforeGrowth =
            CopyMappedBuffer<GpuObjectRender>(BufferRole::ObjectRender, countBeforeGrowth);

        constexpr size_t growthObjectCount = 257;
        while (objs.size() < growthObjectCount)
        {
            const float offset = static_cast<float>(objs.size());
            objs.emplace_back(glm::vec3(20.0f + offset * 2.0f, 50.0f, 0.0f), glm::vec3(0.0f),
                              1.0e16f, 1000.0f);
            objs.back().initializing = true;
        }
        Require(objs.size() > capacityBeforeGrowth,
                "Capacity-growth scenario did not exceed the old capacity");
        SynchronizeObjectStateBuffers(objectCount, objs.size());
        Require(capacityBeforeGrowth == 256 && objectStateCapacity == 512,
                "Object state buffers did not grow from 256 to 512 slots");
        ValidateObjectBufferLayout(objectCount);

        const auto *currentAfterGrowth =
            gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
        const auto *nextAfterGrowth =
            gpuMemoryManager.Get(BufferRole::NextState).MappedPtr<GpuObjectState>();
        const auto *physicalAfterGrowth =
            gpuMemoryManager.Get(BufferRole::ObjectPhysical).MappedPtr<GpuObjectPhysical>();
        const auto *derivedAfterGrowth =
            gpuMemoryManager.Get(BufferRole::ObjectDerived).MappedPtr<GpuObjectDerived>();
        const auto *controlAfterGrowth =
            gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
        const auto *accelerationAfterGrowth =
            gpuMemoryManager.Get(BufferRole::ObjectAcceleration).MappedPtr<GpuObjectAcceleration>();
        const auto *collisionAfterGrowth =
            gpuMemoryManager.Get(BufferRole::ObjectCollision).MappedPtr<GpuObjectCollision>();
        const auto *renderAfterGrowth =
            gpuMemoryManager.Get(BufferRole::ObjectRender).MappedPtr<GpuObjectRender>();
        for (size_t index = 0; index < countBeforeGrowth; ++index)
        {
            Require(glm::length(currentAfterGrowth[index].position -
                                currentBeforeGrowth[index].position) <= 1.0e-6f &&
                        glm::length(currentAfterGrowth[index].velocity -
                                    currentBeforeGrowth[index].velocity) <= 1.0e-6f,
                    "Capacity growth did not preserve CurrentState");
            Require(glm::length(nextAfterGrowth[index].position -
                                nextBeforeGrowth[index].position) <= 1.0e-6f &&
                        glm::length(nextAfterGrowth[index].velocity -
                                    nextBeforeGrowth[index].velocity) <= 1.0e-6f,
                    "Capacity growth did not preserve NextState");
            Require(glm::length(physicalAfterGrowth[index].massDensity -
                                physicalBeforeGrowth[index].massDensity) <= 1.0e-6f,
                    "Capacity growth did not preserve ObjectPhysical");
            Require(glm::length(derivedAfterGrowth[index].radiusRs -
                                derivedBeforeGrowth[index].radiusRs) <= 1.0e-6f,
                    "Capacity growth did not preserve ObjectDerived");
            Require(controlAfterGrowth[index].initializing ==
                            controlBeforeGrowth[index].initializing &&
                        controlAfterGrowth[index].launched == controlBeforeGrowth[index].launched &&
                        controlAfterGrowth[index].target == controlBeforeGrowth[index].target &&
                        controlAfterGrowth[index].id == controlBeforeGrowth[index].id,
                    "Capacity growth did not preserve ObjectControl");
            Require(glm::length(accelerationAfterGrowth[index].acceleration -
                                accelerationBeforeGrowth[index].acceleration) <= 1.0e-6f,
                    "Capacity growth did not preserve ObjectAcceleration");
            Require(collisionAfterGrowth[index].overlapCount ==
                        collisionBeforeGrowth[index].overlapCount,
                    "Capacity growth did not preserve ObjectCollision");
            Require(glm::length(renderAfterGrowth[index].color - renderBeforeGrowth[index].color) <=
                        1.0e-6f,
                    "Capacity growth did not preserve ObjectRender");
        }
        for (size_t index = referenceState.size(); index < objs.size(); ++index)
        {
            const Object &object = objs[index];
            referenceState.push_back({object.id, object.position, object.velocity, object.mass,
                                      object.radius, object.initializing, object.launched});
        }

        const std::vector<AccelerationSample> growthAccelerations =
            AdvanceReference(referenceState, epsilon, false);
        RunCountedTransition(nbodyProgram, integrateProgram, objectCount, epsilon, false,
                             transitionCount);
        gpuMemoryManager.FenceGpuCompletion();
        ValidateObjectBufferLayout(objectCount);
        ValidateInstancedRendering(renderingProgram, validationFramebuffer, referenceState, 1,
                                   renderMetrics);
        ValidateGpuState(referenceState, growthAccelerations, transitionCount, validationMetrics);

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
        std::cout << "Validation scenarios: two-body acceleration, zero interaction, "
                  << "single and multiple overlap parity, zero-object dispatch, 8-step ping-pong, "
                  << "pause/resume, initializing exclusion and launch, "
                  << "append without resize, 256-to-512 capacity growth, FBO position/scale/color, "
                  << "invalid slot, and 0/1/2 instances." << std::endl;
        std::cout << "Validation summary: state transitions=" << transitionCount
                  << ", oracle checkpoints=" << validationMetrics.checkpoints
                  << ", max acceleration error=" << validationMetrics.maxAccelerationError
                  << " m/s^2, max relative acceleration error="
                  << validationMetrics.maxRelativeAccelerationError
                  << ", max collision-count error=" << validationMetrics.maxCollisionCountError
                  << ", max position error=" << validationMetrics.maxPositionError
                  << " units, max velocity error=" << validationMetrics.maxVelocityError
                  << " units, final logical count=" << objectCount
                  << ", final capacity=" << objectStateCapacity << std::endl;
        std::cout << "GPU/snapshot-oracle validation passed." << std::endl;
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