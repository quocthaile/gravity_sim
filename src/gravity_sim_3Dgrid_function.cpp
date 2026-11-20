#include "gravity_sim_3Dgrid_function.hpp"
#include "error_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

void SavePositions(const char *filename, const std::vector<std::vector<glm::vec3>> &positions,
                   double dt)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << filename << '\n';
        return;
    }

    file << "time,step,object_id,x,y,z\n";
    file << std::setprecision(10);

    for (size_t step = 0; step < positions.size(); ++step)
    {
        double simulationTime = static_cast<double>(step) * dt;

        for (size_t objectId = 0; objectId < positions[step].size(); ++objectId)
        {
            const auto &p = positions[step][objectId];

            file << simulationTime << ',' << step << ',' << objectId << ',' << p.x << ',' << p.y
                 << ',' << p.z << '\n';
        }
    }

    if (!file.good())
    {
        std::cerr << "Error writing CSV: " << filename << '\n';
    }
    else
    {
        std::cout << "Saved positions to " << filename << '\n';
    }
}

struct ObjectMesh
{
    GLuint objectVAO = 0;
    GLuint objectVBO = 0;
    GLsizei vertexCount = 0;
};

ObjectMesh objectMesh;

namespace
{
void ValidateOctreeStorageCapacity(std::size_t objectCapacity, std::size_t nodeCapacity)
{
    GLint64 maxStorageBlockSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxStorageBlockSize);
    if (maxStorageBlockSize <= 0)
    {
        throw std::runtime_error("OpenGL reported an invalid maximum SSBO block size");
    }

    const auto validateBuffer = [maxStorageBlockSize](const char *name, std::size_t elementSize,
                                                       std::size_t capacity)
    {
        if (elementSize == 0 ||
            capacity > std::numeric_limits<std::size_t>::max() / elementSize)
        {
            throw std::length_error(std::string(name) + " capacity overflows addressable memory");
        }
        const std::size_t byteSize = capacity * elementSize;
        if (byteSize > static_cast<std::uint64_t>(maxStorageBlockSize) ||
            byteSize > static_cast<std::size_t>(std::numeric_limits<GLsizeiptr>::max()))
        {
            throw std::length_error(std::string(name) +
                                    " exceeds the OpenGL shader storage block limit");
        }
    };

    validateBuffer("MassField", sizeof(GpuMassAggregate), nodeCapacity);
    validateBuffer("OctreeNodes", sizeof(GpuOctreeNode), nodeCapacity);
    validateBuffer("ObjectIndexList", sizeof(GpuObjectIndex), objectCapacity);
    validateBuffer("ObjectLeafIndex", sizeof(GpuObjectIndex), objectCapacity);
    validateBuffer("OctreeMetadata", sizeof(GpuOctreeMetadata), 1);
}

void ResetOctreeMetadata()
{
    const GpuOctreeMetadata metadata =
        MakeUninitializedOctreeMetadata(gpuMemoryManager.Get(BufferRole::OctreeNodes).Capacity());
    gpuMemoryManager.Upload(BufferRole::OctreeMetadata, &metadata, sizeof(metadata), 1);
}

void ValidateOctreeBindings()
{
    constexpr BufferRole octreeRoles[] = {
        BufferRole::MassField,       BufferRole::OctreeNodes,     BufferRole::ObjectIndexList,
        BufferRole::OctreeMetadata,  BufferRole::ObjectLeafIndex,
    };

    for (const BufferRole role : octreeRoles)
    {
        GLint boundBuffer = 0;
        glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, BindingFor(role), &boundBuffer);
        if (boundBuffer != static_cast<GLint>(gpuMemoryManager.Get(role).Handle()))
        {
            throw std::runtime_error("Octree SSBO is not bound to its assigned binding point");
        }
    }
}
} // namespace

// Tolerances for float comparison: absolute for values near zero, relative for large magnitudes
// (e.g. mass ~1e22) where a fixed absolute epsilon would be meaningless.
constexpr float kFloatAbsEpsilon = 1e-6f;
constexpr float kFloatRelEpsilon = 1e-5f;

bool IsNearlyEqual(float left, float right)
{
    if (left == right) // Fast path; also handles matching infinities.
    {
        return true;
    }
    const float diff = std::fabs(left - right);
    const float scale = std::max(std::fabs(left), std::fabs(right));
    return diff <= std::max(kFloatAbsEpsilon, kFloatRelEpsilon * scale);
}

bool IsCenterOfMassNearlyEqual(float gpuValue, float referenceValue, float nodeDiameter)
{
    const float difference = std::fabs(gpuValue - referenceValue);
    const float scale = std::max({std::fabs(gpuValue), std::fabs(referenceValue),
                                  std::fabs(nodeDiameter)});
    return difference <= std::max(kFloatAbsEpsilon, kFloatRelEpsilon * scale);
}

bool IsSameVec4(const glm::vec4 &left, const glm::vec4 &right)
{
    return IsNearlyEqual(left.x, right.x) && IsNearlyEqual(left.y, right.y) &&
           IsNearlyEqual(left.z, right.z) && IsNearlyEqual(left.w, right.w);
}

std::uint32_t AllocateObjectId()
{
    // Monotonic, deterministic within one run: IDs follow Object construction order and are never
    // reused, so they stay unique among simultaneously existing objects.
    static std::uint32_t nextObjectId = kInvalidObjectId + 1;
    if (nextObjectId == std::numeric_limits<std::uint32_t>::max())
    {
        throw std::overflow_error("Object ID space exhausted");
    }
    return nextObjectId++;
}

std::string LoadShaderSource(const std::string &filePath)
{
    std::ifstream shaderFile(filePath);
    if (!shaderFile)
    {
        throw std::runtime_error("Unable to open shader file: " + filePath);
    }
    return std::string(std::istreambuf_iterator<char>(shaderFile),
                       std::istreambuf_iterator<char>());
}

void CreateShaderProgram(ShaderPrograms &shaderPrograms)
{
    shaderPrograms.gridRendering = CreateGraphicsProgram("shaders/grid.vert", "shaders/grid.frag");
    shaderPrograms.objectRendering =
        CreateGraphicsProgram("shaders/objects.vert", "shaders/objects.frag");
    shaderPrograms.gridCompute = CreateComputeProgram("shaders/grid.comp");
    shaderPrograms.boundingBoxCompute = CreateComputeProgram("shaders/bounding_box.comp");
    shaderPrograms.octreeBuildCompute = CreateComputeProgram("shaders/build_octree.comp");
    shaderPrograms.massAggregationCompute = CreateComputeProgram("shaders/mass_aggregation.comp");
    shaderPrograms.barnesHutCompute = CreateComputeProgram("shaders/barnes_hut.comp");
    shaderPrograms.collisionCompute = CreateComputeProgram("shaders/collision.comp");
    shaderPrograms.nbodyCompute = CreateComputeProgram("shaders/nbody.comp");
    shaderPrograms.integrateCompute = CreateComputeProgram("shaders/integrate.comp");
}

void InitializeComputeUniformLocations(const ShaderPrograms &programs,
                                       ComputeUniformLocations &locations)
{
    const auto getRequiredLocation = [](GLuint program, const char *programName,
                                        const char *uniformName) -> GLint
    {
        const GLint location = glGetUniformLocation(program, uniformName);
        if (location == -1)
        {
            throw std::runtime_error("Required uniform '" + std::string(uniformName) +
                                     "' not found in compute program '" + programName + "'");
        }
        return location;
    };

    locations.boundingBoxObjectCount =
        getRequiredLocation(programs.boundingBoxCompute, "boundingBoxCompute", "u_objectCount");
    locations.boundingBoxObjectCapacity = getRequiredLocation(
        programs.boundingBoxCompute, "boundingBoxCompute", "u_objectCapacity");
    locations.octreeBuildObjectCount =
        getRequiredLocation(programs.octreeBuildCompute, "octreeBuildCompute", "u_objectCount");
    locations.octreeBuildObjectCapacity = getRequiredLocation(
        programs.octreeBuildCompute, "octreeBuildCompute", "u_objectCapacity");

    locations.barnesHutObjectCount =
        getRequiredLocation(programs.barnesHutCompute, "barnesHutCompute", "u_objectCount");
    locations.barnesHutGravitationalConstant = getRequiredLocation(
        programs.barnesHutCompute, "barnesHutCompute", "u_gravitationalConstant");
    locations.barnesHutEpsilon =
        getRequiredLocation(programs.barnesHutCompute, "barnesHutCompute", "u_epsilon");
    locations.barnesHutTheta =
        getRequiredLocation(programs.barnesHutCompute, "barnesHutCompute", "u_theta");
    locations.collisionObjectCount =
        getRequiredLocation(programs.collisionCompute, "collisionCompute", "u_objectCount");

    locations.nbodyObjectCount =
        getRequiredLocation(programs.nbodyCompute, "nbodyCompute", "u_objectCount");
    locations.nbodyGravitationalConstant = getRequiredLocation(
        programs.nbodyCompute, "nbodyCompute", "u_gravitationalConstant");
    locations.nbodyEpsilon = getRequiredLocation(programs.nbodyCompute, "nbodyCompute", "u_epsilon");

    locations.integrateObjectCount =
        getRequiredLocation(programs.integrateCompute, "integrateCompute", "u_objectCount");
    locations.integratePaused =
        getRequiredLocation(programs.integrateCompute, "integrateCompute", "u_paused");
    locations.integrateDt =
        getRequiredLocation(programs.integrateCompute, "integrateCompute", "u_dt");

    locations.gridObjectCount =
        getRequiredLocation(programs.gridCompute, "gridCompute", "u_objectCount");
    locations.gridWidth = getRequiredLocation(programs.gridCompute, "gridCompute", "u_gridWidth");
    locations.gridHeight =
        getRequiredLocation(programs.gridCompute, "gridCompute", "u_gridHeight");
}

glm::mat4 CreateScenePerspectiveProjection()
{
    return glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 750000.0f);
}

void InitializeGridRenderingUniforms(GLuint gridProgram, ShaderUniformLocations &uniformLocations)
{
    glUseProgram(gridProgram);
    uniformLocations.gridModel = glGetUniformLocation(gridProgram, "model");
    uniformLocations.gridObjectColor = glGetUniformLocation(gridProgram, "objectColor");
    uniformLocations.gridView = glGetUniformLocation(gridProgram, "view");
    uniformLocations.gridProjection = glGetUniformLocation(gridProgram, "projection");

    const glm::mat4 projection = CreateScenePerspectiveProjection();
    glUniformMatrix4fv(uniformLocations.gridProjection, 1, GL_FALSE, glm::value_ptr(projection));
}

void InitializeObjectRenderingUniforms(GLuint objectProgram,
                                       ShaderUniformLocations &uniformLocations)
{
    glUseProgram(objectProgram);
    uniformLocations.objectView = glGetUniformLocation(objectProgram, "view");
    uniformLocations.objectProjection = glGetUniformLocation(objectProgram, "projection");

    const glm::mat4 projection = CreateScenePerspectiveProjection();
    glUniformMatrix4fv(uniformLocations.objectProjection, 1, GL_FALSE, glm::value_ptr(projection));
}

void InitializeGlfwCallbacks(GLFWwindow *window)
{
    // Mouse callbacks for camera rotation and zooming.
    glfwSetCursorPosCallback(window, MouseCallback);
    glfwSetScrollCallback(window, ScrollCallback);
    glfwSetKeyCallback(window, KeyCallback);
    // Mouse button callback for creating and launching objects.
    glfwSetMouseButtonCallback(window, MouseButtonCallback);
    // GLFW_CURSOR_NORMAL allows the mouse to move freely and be visible, enabling camera rotation
    // with the right mouse button.
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void PrintComputeLimits()
{
    GLint maxSsboBindings = 0;
    GLint64 maxSsboBlockSize = 0;
    GLint maxComputeStorageBlocks = 0;
    GLint maxWorkGroupInvocations = 0;
    GLint maxSharedMemorySize = 0;
    GLint maxWorkGroupSize[3] = {};
    GLint maxWorkGroupCount[3] = {};

    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxSsboBindings);
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxSsboBlockSize);
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxComputeStorageBlocks);
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxWorkGroupInvocations);
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &maxSharedMemorySize);

    for (GLuint dimension = 0; dimension < 3; ++dimension)
    {
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, dimension, &maxWorkGroupSize[dimension]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, dimension, &maxWorkGroupCount[dimension]);
    }

    std::cout << "OpenGL compute limits" << std::endl;
    std::cout << "  SSBO binding points: " << maxSsboBindings << " (valid indices: 0 - "
              << maxSsboBindings - 1 << ")" << std::endl;
    std::cout << "  Max SSBO block size: " << maxSsboBlockSize << " bytes" << std::endl;
    std::cout << "  Compute shader SSBO blocks: " << maxComputeStorageBlocks << std::endl;
    std::cout << "  Max work-group size: " << maxWorkGroupSize[0] << " x " << maxWorkGroupSize[1]
              << " x " << maxWorkGroupSize[2] << std::endl;
    std::cout << "  Max invocations per work-group: " << maxWorkGroupInvocations << std::endl;
    std::cout << "  Max work-group count: " << maxWorkGroupCount[0] << " x " << maxWorkGroupCount[1]
              << " x " << maxWorkGroupCount[2] << std::endl;
    std::cout << "  Max shared memory per work-group: " << maxSharedMemorySize << " bytes"
              << std::endl;
}

GLFWwindow *StartGLU(bool hidden)
{
    // Initialize GLFW and create a window with an OpenGL context.
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW." << std::endl;
        return nullptr;
    }
    // Request OpenGL 4.4 core for immutable buffer storage and persistent mapping.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (hidden)
    {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }
    // Create a window with the specified dimensions and title.
    GLFWwindow *window = glfwCreateWindow(1920, 1080, "GRAVITY SIMULATION - 3D GRID", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    // Initialize GLEW to load OpenGL function pointers.
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        glfwTerminate();
        return nullptr;
    }
    if (!GLEW_VERSION_4_4)
    {
        std::cerr << "OpenGL 4.4 is required for persistent mapped buffer storage." << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    // Print compute limits for debugging purposes.
    PrintComputeLimits();
    // Enable depth testing and set the viewport dimensions.
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, 1920, 1080);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return window;
}

GLuint CreateGraphicsProgram(const std::string &vertexFilePath, const std::string &fragmentFilePath)
{
    const std::string vertexSource = LoadShaderSource(vertexFilePath);
    const std::string fragmentSource = LoadShaderSource(fragmentFilePath);
    const char *vertexSourceData = vertexSource.c_str();
    const char *fragmentSourceData = fragmentSource.c_str();
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSourceData, nullptr);
    glCompileShader(vertexShader);

    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSourceData, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
    }

    GLuint renderingShaderProgram = glCreateProgram();
    glAttachShader(renderingShaderProgram, vertexShader);
    glAttachShader(renderingShaderProgram, fragmentShader);
    glLinkProgram(renderingShaderProgram);

    glGetProgramiv(renderingShaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(renderingShaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return renderingShaderProgram;
}

GLuint CreateComputeProgram(const std::string &filePath)
{
    const std::string computeSource = LoadShaderSource(filePath);
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    const char *source = computeSource.c_str();
    glShaderSource(computeShader, 1, &source, nullptr);
    glCompileShader(computeShader);

    GLint success = GL_FALSE;
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        GLint infoLogLength = 0;
        glGetShaderiv(computeShader, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::string infoLog(static_cast<size_t>(infoLogLength), '\0');
        glGetShaderInfoLog(computeShader, infoLogLength, nullptr, infoLog.data());
        glDeleteShader(computeShader);
        throw std::runtime_error("Compute shader compilation failed: " + infoLog);
    }

    GLuint computeShaderProgram = glCreateProgram();
    glAttachShader(computeShaderProgram, computeShader);
    glLinkProgram(computeShaderProgram);

    glGetProgramiv(computeShaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        GLint infoLogLength = 0;
        glGetProgramiv(computeShaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::string infoLog(static_cast<size_t>(infoLogLength), '\0');
        glGetProgramInfoLog(computeShaderProgram, infoLogLength, nullptr, infoLog.data());
        glDeleteShader(computeShader);
        glDeleteProgram(computeShaderProgram);
        throw std::runtime_error("Compute program linking failed: " + infoLog);
    }

    glDeleteShader(computeShader);
    return computeShaderProgram;
}

void CreateMeshBuffers(GLuint &VAO, GLuint &VBO, const float *vertices, size_t vertexCount,
                       GLuint *EBO, const unsigned int *indices, size_t indexCount)
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(float), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    if (EBO != nullptr)
    {
        glGenBuffers(1, EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(unsigned int), indices,
                     GL_STATIC_DRAW);
    }

    glBindVertexArray(0);
}

void CreateUnitSphere()
{
    if (objectMesh.objectVAO != 0)
    {
        return;
    }

    std::vector<float> meshVertices;
    constexpr int stacks = 10;
    constexpr int sectors = 10;
    for (int stack = 0; stack < stacks; ++stack)
    {
        const float theta1 = (static_cast<float>(stack) / stacks) * glm::pi<float>();
        const float theta2 = (static_cast<float>(stack + 1) / stacks) * glm::pi<float>();
        for (int sector = 0; sector < sectors; ++sector)
        {
            const float phi1 = (static_cast<float>(sector) / sectors) * 2.0f * glm::pi<float>();
            const float phi2 = (static_cast<float>(sector + 1) / sectors) * 2.0f * glm::pi<float>();
            const glm::vec3 v1 = SphericalToCartesian(1.0f, theta1, phi1);
            const glm::vec3 v2 = SphericalToCartesian(1.0f, theta1, phi2);
            const glm::vec3 v3 = SphericalToCartesian(1.0f, theta2, phi1);
            const glm::vec3 v4 = SphericalToCartesian(1.0f, theta2, phi2);
            meshVertices.insert(meshVertices.end(),
                                {v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, v2.x, v2.y,
                                 v2.z, v4.x, v4.y, v4.z, v3.x, v3.y, v3.z});
        }
    }

    objectMesh.vertexCount = static_cast<GLsizei>(meshVertices.size() / 3);
    CreateMeshBuffers(objectMesh.objectVAO, objectMesh.objectVBO, meshVertices.data(),
                      meshVertices.size());
}

void DestroyObjectRenderingResources()
{
    if (objectMesh.objectVAO != 0)
    {
        glDeleteVertexArrays(1, &objectMesh.objectVAO);
        objectMesh.objectVAO = 0;
    }
    if (objectMesh.objectVBO != 0)
    {
        glDeleteBuffers(1, &objectMesh.objectVBO);
        objectMesh.objectVBO = 0;
    }
    objectMesh.vertexCount = 0;
}

std::vector<glm::vec4> CreateBaseGridGPU()
{
    std::vector<float> gridVertices = CreateGridVertices(gridSize, gridDivisions);
    const size_t nodeCount = gridVertices.size() / 3;
    std::vector<glm::vec4> gpuGridVertices;
    gpuGridVertices.reserve(nodeCount);
    for (size_t i = 0; i < nodeCount; ++i)
    {
        gpuGridVertices.emplace_back(gridVertices[i * 3], gridVertices[i * 3 + 1],
                                     gridVertices[i * 3 + 2], 1.0f);
    }
    return gpuGridVertices;
}

size_t CalculateObjectStateBufferCapacity(size_t requiredCount)
{
    if (requiredCount == 0)
    {
        return 0;
    }
    // Start with a minimum capacity of 256 and double it until it meets or exceeds the required
    // count.
    size_t capacity = (objectStateCapacity == 0) ? 256 : objectStateCapacity;
    // Ensure the capacity is at least as large as the required count.
    while (capacity < requiredCount)
    {
        if (capacity > std::numeric_limits<size_t>::max() / 2)
        {
            throw std::length_error("Object state capacity exceeds addressable memory");
        }
        capacity *= 2;
    }
    return capacity;
}

void ManageObjectStateBufferCapacity(size_t objectCount)
{
    constexpr size_t kMinimumObjectStateCapacity = 256;
    size_t newCapacity = objectStateCapacity;

    if (objectCount > objectStateCapacity)
    {
        newCapacity = CalculateObjectStateBufferCapacity(objectCount);
    }
    else if (objectStateCapacity > kMinimumObjectStateCapacity &&
             objectCount <= objectStateCapacity / 4)
    {
        newCapacity = objectStateCapacity / 2;
        if (newCapacity < kMinimumObjectStateCapacity)
        {
            newCapacity = kMinimumObjectStateCapacity;
        }
    }

    if (newCapacity == objectStateCapacity)
    {
        return;
    }

    const size_t newNodeCapacity = CalculateOctreeNodeBufferCapacity(newCapacity);
    ValidateOctreeStorageCapacity(newCapacity, newNodeCapacity);
    gpuMemoryManager.WaitForGpuCompletion();
    gpuMemoryManager.Resize(BufferRole::CurrentState, newCapacity);
    gpuMemoryManager.Resize(BufferRole::NextState, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectPhysical, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectDerived, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectControl, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectAcceleration, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectRender, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectCollision, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectIndexList, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectLeafIndex, newCapacity);
    gpuMemoryManager.Resize(BufferRole::OctreeNodes, newNodeCapacity);
    gpuMemoryManager.Resize(BufferRole::MassField, newNodeCapacity);

    gpuMemoryManager.Get(BufferRole::ObjectIndexList).SetLogicalCount(0);
    gpuMemoryManager.Get(BufferRole::ObjectLeafIndex).SetLogicalCount(0);
    gpuMemoryManager.Get(BufferRole::OctreeNodes).SetLogicalCount(0);
    gpuMemoryManager.Get(BufferRole::MassField).SetLogicalCount(0);
    ResetOctreeMetadata();
    ValidateOctreeBindings();

    objectStateCapacity = newCapacity;
}

void objectGpuState(size_t firstObject, size_t objectCount)
{
    if (firstObject > objectCount || objectCount > objs.size() || objectCount > objectStateCapacity)
    {
        throw std::out_of_range("Object materialization range exceeds available objects/capacity");
    }

    gpuMemoryManager.WaitForGpuCompletion();
    auto *state = gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    auto *physical =
        gpuMemoryManager.Get(BufferRole::ObjectPhysical).MappedPtr<GpuObjectPhysical>();
    auto *derived = gpuMemoryManager.Get(BufferRole::ObjectDerived).MappedPtr<GpuObjectDerived>();
    auto *control = gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    auto *render = gpuMemoryManager.Get(BufferRole::ObjectRender).MappedPtr<GpuObjectRender>();
    auto *collisions =
        gpuMemoryManager.Get(BufferRole::ObjectCollision).MappedPtr<GpuObjectCollision>();
    if (state == nullptr || physical == nullptr || derived == nullptr || control == nullptr ||
        render == nullptr || collisions == nullptr)
    {
        throw std::runtime_error("CPU-facing object buffers must be persistently mapped");
    }

    std::vector<GpuObjectState> nextStates;
    std::vector<GpuObjectAcceleration> zeroAccelerations;
    std::vector<GpuObjectCollision> zeroCollisions;
    nextStates.reserve(objectCount - firstObject);
    zeroAccelerations.assign(objectCount - firstObject, GpuObjectAcceleration{glm::vec4(0.0f)});
    zeroCollisions.assign(objectCount - firstObject, GpuObjectCollision{0});
    for (size_t i = firstObject; i < objectCount; ++i)
    {
        Object &object = objs[i];
        object.gpuStateIndex = i;
        const GpuObjectState initialState = {glm::vec4(object.GetPosition(), 0.0f),
                                             glm::vec4(object.velocity, 0.0f)};
        state[i] = initialState;
        physical[i] = {glm::vec4(object.mass, object.density, 0.0f, 0.0f)};
        derived[i] = {glm::vec4(object.radius, object.rs, 0.0f, 0.0f)};
        control[i] = {static_cast<std::uint32_t>(object.initializing),
                      static_cast<std::uint32_t>(object.launched),
                      static_cast<std::uint32_t>(object.target), object.id};
        render[i] = {object.color};
        nextStates.push_back(initialState);
    }

    if (!nextStates.empty())
    {
        const std::size_t byteOffset = firstObject * sizeof(GpuObjectState);
        gpuMemoryManager.Upload(BufferRole::NextState, nextStates.data(),
                                nextStates.size() * sizeof(GpuObjectState), objectCount,
                                byteOffset);
        gpuMemoryManager.Upload(BufferRole::ObjectAcceleration, zeroAccelerations.data(),
                                zeroAccelerations.size() * sizeof(GpuObjectAcceleration),
                                objectCount, firstObject * sizeof(GpuObjectAcceleration));
        gpuMemoryManager.Upload(BufferRole::ObjectCollision, zeroCollisions.data(),
                                zeroCollisions.size() * sizeof(GpuObjectCollision), objectCount,
                                firstObject * sizeof(GpuObjectCollision));
    }

    for (BufferRole role :
         {BufferRole::CurrentState, BufferRole::ObjectPhysical, BufferRole::ObjectDerived,
          BufferRole::ObjectControl, BufferRole::ObjectAcceleration, BufferRole::ObjectRender})
    {
        gpuMemoryManager.Get(role).SetLogicalCount(objectCount);
    }
    gpuMemoryManager.Get(BufferRole::ObjectCollision).SetLogicalCount(objectCount);
    gpuMemoryManager.Get(BufferRole::NextState).SetLogicalCount(objectCount);
}

void SynchronizeObjectStateBuffers(size_t &objectCount, size_t newObjectCount)
{
    const size_t firstChangedObject = newObjectCount > objectCount ? objectCount : newObjectCount;
    objectCount = newObjectCount;
    ManageObjectStateBufferCapacity(objectCount);
    objectGpuState(firstChangedObject, objectCount);
}

void SynchronizeObjectGpuProperties(size_t objectIndex)
{
    if (objectIndex >= objs.size() || !gpuMemoryManager.Contains(BufferRole::ObjectControl) ||
        objectIndex >= gpuMemoryManager.Get(BufferRole::ObjectControl).LogicalCount())
    {
        return;
    }

    auto *physical =
        gpuMemoryManager.Get(BufferRole::ObjectPhysical).MappedPtr<GpuObjectPhysical>();
    auto *derived = gpuMemoryManager.Get(BufferRole::ObjectDerived).MappedPtr<GpuObjectDerived>();
    auto *control = gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    auto *render = gpuMemoryManager.Get(BufferRole::ObjectRender).MappedPtr<GpuObjectRender>();
    if (physical == nullptr || derived == nullptr || control == nullptr || render == nullptr)
    {
        throw std::runtime_error("CPU-facing object buffers must be persistently mapped");
    }

    const Object &object = objs[objectIndex];
    const glm::vec4 massDensity(object.mass, object.density, 0.0f, 0.0f);
    const glm::vec4 radiusRs(object.radius, object.rs, 0.0f, 0.0f);
    const GpuObjectControl objectControl = {static_cast<std::uint32_t>(object.initializing),
                                            static_cast<std::uint32_t>(object.launched),
                                            static_cast<std::uint32_t>(object.target), object.id};
    const bool physicalChanged = !IsSameVec4(physical[objectIndex].massDensity, massDensity);
    const bool derivedChanged = !IsSameVec4(derived[objectIndex].radiusRs, radiusRs);
    const bool controlChanged = control[objectIndex].initializing != objectControl.initializing ||
                                control[objectIndex].launched != objectControl.launched ||
                                control[objectIndex].target != objectControl.target ||
                                control[objectIndex].id != objectControl.id;
    const bool renderChanged = !IsSameVec4(render[objectIndex].color, object.color);
    if (!physicalChanged && !derivedChanged && !controlChanged && !renderChanged)
    {
        return;
    }

    gpuMemoryManager.WaitForGpuCompletion();
    if (physicalChanged)
    {
        physical[objectIndex].massDensity = massDensity;
    }
    if (derivedChanged)
    {
        derived[objectIndex].radiusRs = radiusRs;
    }
    if (controlChanged)
    {
        control[objectIndex] = objectControl;
    }
    if (renderChanged)
    {
        render[objectIndex].color = object.color;
    }
}

void UploadInitializingObjectState(size_t objectIndex)
{
    if (objectIndex >= objs.size() || !objs[objectIndex].initializing ||
        !gpuMemoryManager.Contains(BufferRole::CurrentState) ||
        objectIndex >= gpuMemoryManager.Get(BufferRole::CurrentState).LogicalCount())
    {
        return;
    }

    gpuMemoryManager.WaitForGpuCompletion();
    const Object &object = objs[objectIndex];
    const GpuObjectState state = {glm::vec4(object.GetPosition(), 0.0f),
                                  glm::vec4(object.velocity, 0.0f)};
    const size_t logicalCount = objs.size();
    const size_t byteOffset = objectIndex * sizeof(GpuObjectState);
    gpuMemoryManager.Upload(BufferRole::CurrentState, &state, sizeof(state), logicalCount,
                            byteOffset);
    gpuMemoryManager.Upload(BufferRole::NextState, &state, sizeof(state), logicalCount, byteOffset);
}

void InitializeGridRenderingResources(const std::vector<glm::vec4> &gridVertices,
                                      const std::vector<unsigned int> &gridIndices)
{
    // The grid VAO sources its vertices directly from the DeformedGrid buffer written by
    // grid.comp, so ComputePipeline() must have created DeformedGrid before this is called.
    const GpuBuffer &deformedGrid = gpuMemoryManager.Get(BufferRole::DeformedGrid);
    if (deformedGrid.Capacity() < gridVertices.size())
    {
        throw std::length_error("DeformedGrid capacity is smaller than the grid node count");
    }

    gridNodeCount = gridVertices.size();
    gridIndexCount = gridIndices.size();

    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridEBO);
    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, deformedGrid.Handle());
    // location 0, 3 positions, type float, not normalized, stride is size of glm::vec4, offset is 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);
    glEnableVertexAttribArray(0);
    // Bind the EBO and upload the index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndices.size() * sizeof(unsigned int),
                 gridIndices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ComputePipeline(const std::vector<glm::vec4> &gridVertices)
{
    GLint maxStorageBufferBindings = 0;
    GLint maxComputeStorageBlocks = 0;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxStorageBufferBindings);
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxComputeStorageBlocks);
    if (maxStorageBufferBindings <= static_cast<GLint>(BindingFor(BufferRole::ObjectCollision)))
    {
        throw std::runtime_error("OpenGL does not provide SSBO binding 15 for ObjectCollision");
    }
    if (maxComputeStorageBlocks < 8)
    {
        throw std::runtime_error(
            "OpenGL does not provide enough compute SSBO blocks for Barnes-Hut traversal");
    }

    objectStateCapacity = CalculateObjectStateBufferCapacity(objs.size());
    if (objectStateCapacity == 0)
    {
        objectStateCapacity = 256;
    }
    const size_t octreeNodeCapacity = CalculateOctreeNodeBufferCapacity(objectStateCapacity);
    ValidateOctreeStorageCapacity(objectStateCapacity, octreeNodeCapacity);

    gpuMemoryManager.Create({BufferRole::CurrentState, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::ReadWrite,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectState),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::NextState, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::ReadWrite,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectState),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectPhysical, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::ReadWrite,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectPhysical),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectDerived, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::ReadWrite,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectDerived),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectControl, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::ReadWrite,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectControl),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectAcceleration, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::Read,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectAcceleration),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectRender, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::Write,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectRender),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectCollision, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::ReadWrite,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectCollision),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::MassField, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(GpuMassAggregate), octreeNodeCapacity});
    gpuMemoryManager.Create({BufferRole::OctreeNodes, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(GpuOctreeNode), octreeNodeCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectIndexList, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(GpuObjectIndex), objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::OctreeMetadata, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(GpuOctreeMetadata), 1});
    gpuMemoryManager.Create({BufferRole::ObjectLeafIndex, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(GpuObjectIndex), objectStateCapacity});
    ResetOctreeMetadata();
    ValidateOctreeBindings();
    gpuMemoryManager.Create({BufferRole::BaseGrid, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(glm::vec4), gridVertices.size()});
    gpuMemoryManager.Upload(BufferRole::BaseGrid, gridVertices.data(),
                            gridVertices.size() * sizeof(glm::vec4), gridVertices.size());
    gpuMemoryManager.Create({BufferRole::DeformedGrid, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(glm::vec4), gridVertices.size()});
}

void RunNBodyCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                     size_t objectCount, float epsilon)
{
    if (objectCount == 0)
    {
        return;
    }

    constexpr GLuint localSizeX = 64;
    const GLuint groupCountX = static_cast<GLuint>((objectCount + localSizeX - 1) / localSizeX);

    glUseProgram(computeShaderProgram);

    glUniform1ui(locations.nbodyObjectCount, static_cast<GLuint>(objectCount));
    glUniform1d(locations.nbodyGravitationalConstant, kGravitationalConstant);
    glUniform1f(locations.nbodyEpsilon, epsilon);

    glDispatchCompute(groupCountX, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    static size_t lastReportedObjectCount = std::numeric_limits<size_t>::max();
    if (objectCount != lastReportedObjectCount)
    {
        std::cout << "GPU N-body dispatch: objects=" << objectCount
                  << ", work groups=" << groupCountX << " x 1 x 1" << std::endl;
        lastReportedObjectCount = objectCount;
    }
}

namespace
{
template <typename T>
std::vector<T> ReadOctreeBuffer(BufferRole role, std::size_t count)
{
    const GpuBuffer &buffer = gpuMemoryManager.Get(role);
    if (count > buffer.Capacity() || count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        throw std::runtime_error("Octree debug readback exceeds buffer capacity");
    }

    std::vector<T> result(count);
    if (!result.empty())
    {
        glBindBuffer(GL_COPY_READ_BUFFER, buffer.Handle());
        glGetBufferSubData(GL_COPY_READ_BUFFER, 0,
                           static_cast<GLsizeiptr>(result.size() * sizeof(T)), result.data());
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
    }
    return result;
}

#if defined(_DEBUG) || !defined(NDEBUG)
void ValidateOctreeDebug(std::size_t objectCount)
{
    const GpuBuffer &metadataBuffer = gpuMemoryManager.Get(BufferRole::OctreeMetadata);
    GpuOctreeMetadata metadata{};
    glBindBuffer(GL_COPY_READ_BUFFER, metadataBuffer.Handle());
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(metadata), &metadata);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);

    const auto *controls =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    const auto *states =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    if (controls == nullptr || states == nullptr)
    {
        throw std::runtime_error("Object buffers are unavailable for Octree validation");
    }

    std::vector<std::uint8_t> activeObjects(objectCount, 0u);
    std::size_t expectedActiveCount = 0;
    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
    {
        if (controls[objectIndex].initializing == 0u)
        {
            activeObjects[objectIndex] = 1u;
            ++expectedActiveCount;
        }
    }

    const auto status = static_cast<GpuOctreeBuildStatus>(metadata.configuration.w);
    if (metadata.counts.x != expectedActiveCount)
    {
        throw std::runtime_error("Octree active-object count " +
                                 std::to_string(metadata.counts.x) + " does not match " +
                                 std::to_string(expectedActiveCount) +
                                 " active objects in ObjectControl; build status=" +
                                 std::to_string(metadata.configuration.w) + ", rootMin=(" +
                                 std::to_string(metadata.rootBoundsMinDiameter.x) + "," +
                                 std::to_string(metadata.rootBoundsMinDiameter.y) + "," +
                                 std::to_string(metadata.rootBoundsMinDiameter.z) + "), side=" +
                                 std::to_string(metadata.rootBoundsMinDiameter.w) + ", rootMax=(" +
                                 std::to_string(metadata.rootBoundsMax.x) + "," +
                                 std::to_string(metadata.rootBoundsMax.y) + "," +
                                 std::to_string(metadata.rootBoundsMax.z) + ")");
    }
    if (expectedActiveCount == 0)
    {
        if (status != GpuOctreeBuildStatus::Empty || metadata.counts.y != 0u ||
            metadata.counts.z != 0u || metadata.counts.w != kInvalidOctreeIndex)
        {
            throw std::runtime_error("Empty Octree metadata is inconsistent");
        }
        return;
    }
    if (status != GpuOctreeBuildStatus::Valid || metadata.counts.w != 0u ||
        metadata.counts.y == 0u || metadata.counts.y > metadata.configuration.z ||
        metadata.configuration.z !=
            gpuMemoryManager.Get(BufferRole::OctreeNodes).Capacity())
    {
        throw std::runtime_error("GPU Octree construction did not produce a valid tree");
    }

    const std::size_t nodeCount = metadata.counts.y;
    const auto nodes = ReadOctreeBuffer<GpuOctreeNode>(BufferRole::OctreeNodes, nodeCount);
    const auto objectIndices =
        ReadOctreeBuffer<GpuObjectIndex>(BufferRole::ObjectIndexList, expectedActiveCount);
    const auto objectLeafIndices =
        ReadOctreeBuffer<GpuObjectIndex>(BufferRole::ObjectLeafIndex,
                                         gpuMemoryManager.Get(BufferRole::ObjectLeafIndex)
                                             .Capacity());
    std::vector<std::uint8_t> objectOccurrences(objectCount, 0u);
    std::vector<std::uint8_t> occupiedIndexSlots(expectedActiveCount, 0u);
    std::size_t leafCount = 0;

    for (std::size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
    {
        const GpuOctreeNode &node = nodes[nodeIndex];
        if ((node.links.w & kOctreeNodeFlagOccupied) == 0u ||
            node.range.z > metadata.configuration.x ||
            !(node.boundsMinDiameter.w > 0.0f) ||
            !std::isfinite(node.boundsMinDiameter.w))
        {
            throw std::runtime_error("Octree contains an uninitialized or invalid node");
        }

        if ((node.links.w & kOctreeNodeFlagLeaf) != 0u)
        {
            ++leafCount;
            if (node.links.x != 0u || node.links.y != 0u || node.links.z != 0u ||
                node.range.x > expectedActiveCount ||
                node.range.y > expectedActiveCount - node.range.x)
            {
                throw std::runtime_error("Octree leaf has invalid child links or object range");
            }
            for (std::uint32_t offset = 0; offset < node.range.y; ++offset)
            {
                const std::size_t listIndex = node.range.x + offset;
                const GpuObjectIndex objectIndex = objectIndices[listIndex];
                if (listIndex >= occupiedIndexSlots.size() ||
                    occupiedIndexSlots[listIndex] != 0u || objectIndex >= objectCount ||
                    activeObjects[objectIndex] == 0u ||
                    objectOccurrences[objectIndex] != 0u ||
                    objectLeafIndices[objectIndex] != nodeIndex)
                {
                    throw std::runtime_error("Octree leaf contains an invalid or duplicate object");
                }
                const glm::vec3 position = states[objectIndex].position;
                const glm::vec3 boundsMin = node.boundsMinDiameter;
                const glm::vec3 boundsMax = node.boundsMax;
                const float boundsTolerance =
                    std::max(node.boundsMinDiameter.w * 1.0e-5f, 1.0e-4f);
                if (glm::any(glm::lessThan(position, boundsMin - boundsTolerance)) ||
                    glm::any(glm::greaterThan(position, boundsMax + boundsTolerance)))
                {
                    throw std::runtime_error("Octree leaf bounds do not contain their objects");
                }
                occupiedIndexSlots[listIndex] = 1u;
                objectOccurrences[objectIndex] = 1u;
            }
            continue;
        }

        std::uint32_t mask = node.links.z;
        std::uint32_t maskChildCount = 0u;
        while (mask != 0u)
        {
            maskChildCount += mask & 1u;
            mask >>= 1u;
        }
        if ((node.links.z & 0xFFFFFF00u) != 0u || node.links.y != maskChildCount ||
            node.links.y == 0u ||
            node.links.x > nodeCount || node.links.y > nodeCount - node.links.x)
        {
            throw std::runtime_error("Octree internal node has invalid child links");
        }

        std::uint32_t childRank = 0u;
        for (std::uint32_t octant = 0; octant < kOctreeBranchFactor; ++octant)
        {
            if ((node.links.z & (1u << octant)) == 0u)
            {
                continue;
            }
            const GpuOctreeNode &child = nodes[node.links.x + childRank];
            if ((child.links.w & kOctreeNodeFlagOccupied) == 0u ||
                child.range.z != node.range.z + 1u)
            {
                throw std::runtime_error("Octree child range is not contiguous or depth is invalid");
            }

            const float childSide = node.boundsMinDiameter.w * 0.5f;
            glm::vec3 expectedChildMin = node.boundsMinDiameter;
            if ((octant & 1u) != 0u)
            {
                expectedChildMin.x += childSide;
            }
            if ((octant & 2u) != 0u)
            {
                expectedChildMin.y += childSide;
            }
            if ((octant & 4u) != 0u)
            {
                expectedChildMin.z += childSide;
            }
            const glm::vec3 childMin = child.boundsMinDiameter;
            const float boundsTolerance = std::max(childSide * 1.0e-5f, 1.0e-4f);
            if (std::abs(child.boundsMinDiameter.w - childSide) > boundsTolerance ||
                glm::any(glm::greaterThan(glm::abs(childMin - expectedChildMin),
                                          glm::vec3(boundsTolerance))))
            {
                throw std::runtime_error("Octree child bounds do not match its octant");
            }
            ++childRank;
        }
    }

    if (leafCount != metadata.counts.z ||
        std::find(occupiedIndexSlots.begin(), occupiedIndexSlots.end(), 0u) !=
            occupiedIndexSlots.end())
    {
        throw std::runtime_error("Octree leaf ranges do not cover the active object index list");
    }
    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
    {
        if (objectOccurrences[objectIndex] != activeObjects[objectIndex] ||
            (activeObjects[objectIndex] == 0u &&
             objectLeafIndices[objectIndex] != kInvalidOctreeIndex))
        {
            throw std::runtime_error("Octree lost or duplicated an active object");
        }
    }
}
#endif

GpuOctreeMetadata ReadOctreeMetadata()
{
    const GpuBuffer &metadataBuffer = gpuMemoryManager.Get(BufferRole::OctreeMetadata);
    GpuOctreeMetadata metadata{};
    glBindBuffer(GL_COPY_READ_BUFFER, metadataBuffer.Handle());
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(metadata), &metadata);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    return metadata;
}

struct MassAggregationNodeResult
{
    std::size_t nodeIndex = 0;
    std::size_t parentIndex = std::numeric_limits<std::size_t>::max();
    bool leaf = false;
    std::size_t childCount = 0;
    std::vector<std::size_t> childIndices;
    std::size_t objectCount = 0;
    double massGpu = 0.0;
    double massReference = 0.0;
    double massAbsoluteError = 0.0;
    glm::dvec3 comGpu{0.0};
    glm::dvec3 comReference{0.0};
    double comAbsoluteError = 0.0;
};

struct MassAggregationValidation
{
    std::vector<MassAggregationNodeResult> nodes;
    std::size_t activeObjectCount = 0;
    std::size_t leafCount = 0;
    double rootMassError = 0.0;
    double rootComError = 0.0;
};

MassAggregationValidation ValidateMassAggregation(std::size_t objectCount)
{
    const GpuOctreeMetadata metadata = ReadOctreeMetadata();
    const auto status = static_cast<GpuOctreeBuildStatus>(metadata.configuration.w);
    const auto *controls =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    const auto *states =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    const auto *physical =
        gpuMemoryManager.Get(BufferRole::ObjectPhysical).MappedPtr<GpuObjectPhysical>();
    if (controls == nullptr || states == nullptr || physical == nullptr)
    {
        throw std::runtime_error("Object buffers are unavailable for mass validation");
    }

    std::size_t expectedActiveCount = 0;
    double expectedTotalMass = 0.0;
    std::unordered_map<std::uint32_t, std::size_t> objectIndexById;
    std::vector<std::uint32_t> activeObjectIds;
    objectIndexById.reserve(objectCount);
    activeObjectIds.reserve(objectCount);
    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
    {
        const std::uint32_t objectId = controls[objectIndex].id;
        if (objectId == kInvalidObjectId ||
            !objectIndexById.emplace(objectId, objectIndex).second)
        {
            throw std::runtime_error("Mass validation found an invalid or duplicate object ID");
        }
        if (controls[objectIndex].initializing != 0u)
        {
            continue;
        }
        activeObjectIds.push_back(objectId);
        ++expectedActiveCount;
    }
    std::sort(activeObjectIds.begin(), activeObjectIds.end());
    for (const std::uint32_t objectId : activeObjectIds)
    {
        const std::size_t objectIndex = objectIndexById.at(objectId);
        const double mass = physical[objectIndex].massDensity.x;
        const glm::vec3 position = states[objectIndex].position;
        if (!std::isfinite(mass) || !std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z))
        {
            throw std::runtime_error("Mass validation found non-finite active-object input");
        }
        expectedTotalMass += mass;
    }

    if (metadata.counts.x != expectedActiveCount)
    {
        throw std::runtime_error(
            "Mass aggregation active-object count differs from ObjectControl");
    }
    if (expectedActiveCount == 0)
    {
        if (status != GpuOctreeBuildStatus::Empty || metadata.counts.y != 0u)
        {
            throw std::runtime_error("Mass aggregation did not preserve the empty-tree status");
        }
        const GpuMassAggregate emptyAggregate =
            ReadOctreeBuffer<GpuMassAggregate>(BufferRole::MassField, 1).front();
        if (emptyAggregate.centerMass.x != 0.0f || emptyAggregate.centerMass.y != 0.0f ||
            emptyAggregate.centerMass.z != 0.0f || emptyAggregate.centerMass.w != 0.0f)
        {
            throw std::runtime_error("Empty Octree produced a non-zero mass aggregate");
        }
        return {};
    }
    if (status != GpuOctreeBuildStatus::Valid || metadata.counts.y == 0u ||
        metadata.counts.w >= metadata.counts.y)
    {
        throw std::runtime_error("Mass aggregation did not produce a valid root aggregate");
    }

    const std::size_t nodeCount = metadata.counts.y;
    const std::size_t rootIndex = metadata.counts.w;
    const auto nodes = ReadOctreeBuffer<GpuOctreeNode>(BufferRole::OctreeNodes, nodeCount);
    const auto objectIndices =
        ReadOctreeBuffer<GpuObjectIndex>(BufferRole::ObjectIndexList, expectedActiveCount);
    const auto aggregates =
        ReadOctreeBuffer<GpuMassAggregate>(BufferRole::MassField, nodeCount);
    std::unordered_set<std::uint32_t> visitedObjectIds;
    visitedObjectIds.reserve(expectedActiveCount);
    std::vector<std::size_t> parents(nodeCount, std::numeric_limits<std::size_t>::max());
    std::vector<double> referenceMasses(nodeCount, 0.0);
    std::vector<glm::dvec3> referenceCenters(nodeCount, glm::dvec3(0.0));
    MassAggregationValidation result;
    result.activeObjectCount = expectedActiveCount;
    result.nodes.resize(nodeCount);

    for (std::size_t reverseIndex = 0; reverseIndex < nodeCount; ++reverseIndex)
    {
        const std::size_t nodeIndex = nodeCount - 1 - reverseIndex;
        const GpuOctreeNode &node = nodes[nodeIndex];
        const bool leaf = (node.links.w & kOctreeNodeFlagLeaf) != 0u;
        double massSum = 0.0;
        glm::dvec3 weightedPosition(0.0);
        std::size_t childCount = 0;
        std::size_t nodeObjectCount = 0;

        if (leaf)
        {
            const std::size_t firstObject = node.range.x;
            nodeObjectCount = node.range.y;
            if (firstObject > expectedActiveCount ||
                nodeObjectCount > expectedActiveCount - firstObject)
            {
                throw std::runtime_error("Mass aggregation leaf has an invalid object range");
            }
            ++result.leafCount;
            for (std::size_t offset = 0; offset < nodeObjectCount; ++offset)
            {
                const std::size_t gpuObjectIndex = objectIndices[firstObject + offset];
                if (gpuObjectIndex >= objectCount)
                {
                    throw std::runtime_error(
                        "Mass aggregation lost or duplicated an active object");
                }
                const std::uint32_t objectId = controls[gpuObjectIndex].id;
                const auto referenceObject = objectIndexById.find(objectId);
                if (referenceObject == objectIndexById.end() ||
                    controls[referenceObject->second].initializing != 0u ||
                    !visitedObjectIds.insert(objectId).second)
                {
                    throw std::runtime_error(
                        "Mass aggregation lost or duplicated an active object ID");
                }
                const std::size_t objectIndex = referenceObject->second;
                const double mass = physical[objectIndex].massDensity.x;
                const glm::dvec3 position(states[objectIndex].position);
                massSum += mass;
                weightedPosition += position * mass;
            }
        }
        else
        {
            const std::size_t firstChild = node.links.x;
            childCount = node.links.y;
            if (childCount == 0 || firstChild > nodeCount ||
                childCount > nodeCount - firstChild || firstChild <= nodeIndex)
            {
                throw std::runtime_error("Mass aggregation internal node has invalid child links");
            }
            for (std::size_t offset = 0; offset < childCount; ++offset)
            {
                const std::size_t childIndex = firstChild + offset;
                if (parents[childIndex] != std::numeric_limits<std::size_t>::max())
                {
                    throw std::runtime_error("Octree node is referenced by multiple parents");
                }
                parents[childIndex] = nodeIndex;
                result.nodes[nodeIndex].childIndices.push_back(childIndex);
                massSum += referenceMasses[childIndex];
                weightedPosition += referenceCenters[childIndex] * referenceMasses[childIndex];
            }
        }

        const glm::dvec3 centerOfMass =
            massSum == 0.0 ? glm::dvec3(0.0) : weightedPosition / massSum;
        referenceMasses[nodeIndex] = massSum;
        referenceCenters[nodeIndex] = centerOfMass;

        const glm::vec4 gpuAggregate = aggregates[nodeIndex].centerMass;
        if (!std::isfinite(gpuAggregate.x) || !std::isfinite(gpuAggregate.y) ||
            !std::isfinite(gpuAggregate.z) || !std::isfinite(gpuAggregate.w))
        {
            throw std::runtime_error("Mass aggregation produced a non-finite node aggregate");
        }

        const glm::vec3 referenceCenterFloat(centerOfMass);
        const float nodeDiameter = node.boundsMinDiameter.w;
        if (!IsNearlyEqual(gpuAggregate.w, static_cast<float>(massSum)) ||
            !IsCenterOfMassNearlyEqual(gpuAggregate.x, referenceCenterFloat.x, nodeDiameter) ||
            !IsCenterOfMassNearlyEqual(gpuAggregate.y, referenceCenterFloat.y, nodeDiameter) ||
            !IsCenterOfMassNearlyEqual(gpuAggregate.z, referenceCenterFloat.z, nodeDiameter))
        {
            std::ostringstream message;
            message << "GPU aggregate differs at Octree node " << nodeIndex << ": mass GPU="
                    << gpuAggregate.w << ", reference=" << massSum << ", center GPU=("
                    << gpuAggregate.x << ", " << gpuAggregate.y << ", " << gpuAggregate.z
                    << "), reference=(" << referenceCenterFloat.x << ", "
                    << referenceCenterFloat.y << ", " << referenceCenterFloat.z << ')';
            throw std::runtime_error(message.str());
        }

        MassAggregationNodeResult &nodeResult = result.nodes[nodeIndex];
        nodeResult.nodeIndex = nodeIndex;
        nodeResult.parentIndex = parents[nodeIndex];
        nodeResult.leaf = leaf;
        nodeResult.childCount = childCount;
        nodeResult.objectCount = nodeObjectCount;
        nodeResult.massGpu = gpuAggregate.w;
        nodeResult.massReference = massSum;
        nodeResult.massAbsoluteError = std::abs(nodeResult.massGpu - massSum);
        nodeResult.comGpu = glm::dvec3(gpuAggregate);
        nodeResult.comReference = centerOfMass;
        nodeResult.comAbsoluteError = glm::length(nodeResult.comGpu - centerOfMass);
    }

    if (parents[rootIndex] != std::numeric_limits<std::size_t>::max())
    {
        throw std::runtime_error("Octree root is referenced as a child");
    }
    for (std::size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
    {
        if (nodeIndex != rootIndex &&
            parents[nodeIndex] == std::numeric_limits<std::size_t>::max())
        {
            throw std::runtime_error("Octree contains a node unreachable from its root");
        }
        result.nodes[nodeIndex].parentIndex = parents[nodeIndex];
    }
    if (visitedObjectIds.size() != activeObjectIds.size())
    {
        throw std::runtime_error("Octree object partition does not match active object IDs");
    }
    if (!IsNearlyEqual(static_cast<float>(referenceMasses[rootIndex]),
                       static_cast<float>(expectedTotalMass)))
    {
        throw std::runtime_error("Octree root reference mass differs from total object mass");
    }

    const MassAggregationNodeResult &root = result.nodes[rootIndex];
    result.rootMassError = root.massAbsoluteError;
    result.rootComError = root.comAbsoluteError;
    return result;
}

#if defined(_DEBUG) || !defined(NDEBUG)

void ValidateBarnesHutDebug(std::size_t objectCount)
{
    const GpuOctreeMetadata metadata = ReadOctreeMetadata();
    const auto status = static_cast<GpuOctreeBuildStatus>(metadata.configuration.w);
    if (status != GpuOctreeBuildStatus::Valid && status != GpuOctreeBuildStatus::Empty)
    {
        throw std::runtime_error("Barnes-Hut traversal failed; metadata status=" +
                                 std::to_string(metadata.configuration.w));
    }

    const auto *controls =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    const auto *accelerations =
        gpuMemoryManager.Get(BufferRole::ObjectAcceleration).MappedPtr<GpuObjectAcceleration>();
    if (controls == nullptr || accelerations == nullptr)
    {
        throw std::runtime_error("Object buffers are unavailable for Barnes-Hut validation");
    }

    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
    {
        const glm::vec4 acceleration = accelerations[objectIndex].acceleration;
        if (!std::isfinite(acceleration.x) || !std::isfinite(acceleration.y) ||
            !std::isfinite(acceleration.z))
        {
            throw std::runtime_error("Barnes-Hut produced a non-finite object acceleration");
        }
        if (controls[objectIndex].initializing != 0u &&
            (acceleration.x != 0.0f || acceleration.y != 0.0f || acceleration.z != 0.0f))
        {
            throw std::runtime_error("Barnes-Hut assigned acceleration to an initializing object");
        }
    }
}

void ValidateCollisionDebug(std::size_t objectCount)
{
    const auto *controls =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    const auto *states =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    const auto *derived =
        gpuMemoryManager.Get(BufferRole::ObjectDerived).MappedPtr<GpuObjectDerived>();
    const auto *collisions =
        gpuMemoryManager.Get(BufferRole::ObjectCollision).MappedPtr<GpuObjectCollision>();
    if (controls == nullptr || states == nullptr || derived == nullptr || collisions == nullptr)
    {
        throw std::runtime_error("Object buffers are unavailable for collision validation");
    }

    for (std::size_t targetIndex = 0; targetIndex < objectCount; ++targetIndex)
    {
        std::uint32_t expectedOverlapCount = 0;
        if (controls[targetIndex].initializing == 0u)
        {
            const glm::vec3 targetPosition = states[targetIndex].position;
            for (std::size_t sourceIndex = 0; sourceIndex < objectCount; ++sourceIndex)
            {
                if (sourceIndex == targetIndex || controls[sourceIndex].initializing != 0u)
                {
                    continue;
                }
                const float distance =
                    glm::length(glm::vec3(states[sourceIndex].position) - targetPosition);
                if (distance > 0.0f &&
                    derived[sourceIndex].radiusRs.x + derived[targetIndex].radiusRs.x > distance)
                {
                    ++expectedOverlapCount;
                }
            }
        }
        if (collisions[targetIndex].overlapCount != expectedOverlapCount)
        {
            throw std::runtime_error("Collision pass differs from the direct reference count");
        }
    }
}

void ValidateGridDeformationDebug(std::size_t objectCount, std::size_t gridVertexCount)
{
    const auto *states =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    const auto *derived =
        gpuMemoryManager.Get(BufferRole::ObjectDerived).MappedPtr<GpuObjectDerived>();
    if (states == nullptr || derived == nullptr)
    {
        throw std::runtime_error("Object buffers are unavailable for grid validation");
    }

    const auto basePositions =
        ReadOctreeBuffer<glm::vec4>(BufferRole::BaseGrid, gridVertexCount);
    const auto deformedPositions =
        ReadOctreeBuffer<glm::vec4>(BufferRole::DeformedGrid, gridVertexCount);
    for (std::size_t vertexIndex = 0; vertexIndex < gridVertexCount; ++vertexIndex)
    {
        const glm::vec3 basePosition = basePositions[vertexIndex];
        float totalDisplacement = 0.0f;
        for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
        {
            const glm::vec3 objectPosition = states[objectIndex].position;
            const float objectSchwarzschildRadius = derived[objectIndex].radiusRs.y;
            const glm::vec3 toObject = objectPosition - basePosition;
            float distanceMeters = glm::length(toObject) * 1000.0f;
            distanceMeters = std::max(distanceMeters, objectSchwarzschildRadius * 1.0001f);
            if (objectSchwarzschildRadius > 0.0f)
            {
                totalDisplacement -=
                    (2.0e11f * objectSchwarzschildRadius) / distanceMeters;
            }
        }

        const glm::vec4 expected(basePosition.x, basePosition.y + totalDisplacement,
                                 basePosition.z, 1.0f);
        const glm::vec4 actual = deformedPositions[vertexIndex];
        if (!IsSameVec4(actual, expected))
        {
            throw std::runtime_error(
                "GPU grid deformation differs from the CPU Schwarzschild reference at vertex " +
                std::to_string(vertexIndex));
        }
    }
}
} // namespace
#endif

namespace
{
constexpr std::uint32_t kBenchmarkBaseSeed = 20261118u;

std::uint32_t BenchmarkSeed(std::size_t distributionIndex, std::size_t objectCount)
{
    return kBenchmarkBaseSeed +
           static_cast<std::uint32_t>(distributionIndex * 1000 + objectCount);
}

std::vector<Object> CreateBenchmarkObjects(const char *distribution, std::size_t objectCount,
                                           std::uint32_t seed)
{
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> coordinate(-12000.0f, 12000.0f);
    std::uniform_real_distribution<float> mass(1.0e20f, 3.0e23f);
    std::uniform_real_distribution<float> diskRadius(1000.0f, 12000.0f);
    std::uniform_real_distribution<float> angle(0.0f, 2.0f * glm::pi<float>());
    std::normal_distribution<float> clusterOffset(0.0f, 900.0f);
    std::normal_distribution<float> diskThickness(0.0f, 80.0f);
    const std::array<glm::vec3, 3> clusterCenters = {
        glm::vec3(-5000.0f, -2500.0f, -3000.0f), glm::vec3(4500.0f, 3500.0f, -2000.0f),
        glm::vec3(500.0f, -4000.0f, 5000.0f)};

    std::vector<Object> result;
    result.reserve(objectCount);
    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
    {
        glm::vec3 position;
        if (std::strcmp(distribution, "uniform") == 0)
        {
            position = glm::vec3(coordinate(random), coordinate(random), coordinate(random));
        }
        else if (std::strcmp(distribution, "clustered") == 0)
        {
            const glm::vec3 center = clusterCenters[objectIndex % clusterCenters.size()];
            position = center + glm::vec3(clusterOffset(random), clusterOffset(random),
                                          clusterOffset(random));
        }
        else if (std::strcmp(distribution, "disk") == 0)
        {
            const float radius = diskRadius(random);
            const float theta = angle(random);
            position = glm::vec3(radius * std::cos(theta), diskThickness(random),
                                 radius * std::sin(theta));
        }
        else
        {
            throw std::invalid_argument("Unknown benchmark distribution: " +
                                        std::string(distribution));
        }
        result.emplace_back(position, glm::vec3(0.0f), mass(random));
    }
    return result;
}

struct TrajectoryObjectState
{
    std::uint32_t id = kInvalidObjectId;
    double mass = 0.0;
    double radius = 0.0;
    glm::dvec3 position = glm::dvec3(0.0);
    glm::dvec3 velocity = glm::dvec3(0.0);
};

std::vector<TrajectoryObjectState> CreateTrajectoryObjectStates(
    const std::vector<Object> &objects)
{
    std::vector<TrajectoryObjectState> states;
    states.reserve(objects.size());
    for (const Object &object : objects)
    {
        const glm::dvec3 position(object.position);
        const glm::dvec3 velocity(object.velocity);
        if (object.id == kInvalidObjectId || !std::isfinite(object.mass) || object.mass < 0.0f ||
            !std::isfinite(object.radius) || object.radius < 0.0f ||
            !std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
            !std::isfinite(velocity.y) || !std::isfinite(velocity.z))
        {
            throw std::invalid_argument("Benchmark input has invalid object data");
        }
        states.push_back({object.id, static_cast<double>(object.mass),
                          static_cast<double>(object.radius), position, velocity});
    }

    // Keep comparisons in stable ID order instead of relying on array positions.
    std::sort(states.begin(), states.end(),
              [](const TrajectoryObjectState &left, const TrajectoryObjectState &right) {
                  return left.id < right.id;
              });
    for (std::size_t index = 1; index < states.size(); ++index)
    {
        if (states[index - 1].id == states[index].id)
        {
            throw std::invalid_argument("Benchmark input contains duplicate object IDs");
        }
    }
    return states;
}

std::vector<glm::dvec3> CalculateDirectAccelerations(
    const std::vector<TrajectoryObjectState> &objects, double gravitationalConstant,
    double epsilonMeters)
{
    if (!std::isfinite(gravitationalConstant) || gravitationalConstant < 0.0 ||
        !std::isfinite(epsilonMeters) || epsilonMeters < 0.0)
    {
        throw std::invalid_argument("Direct N-body constants must be finite and non-negative");
    }

    std::vector<glm::dvec3> accelerations(objects.size(), glm::dvec3(0.0));
    for (std::size_t targetIndex = 0; targetIndex < objects.size(); ++targetIndex)
    {
        glm::dvec3 acceleration(0.0);
        for (std::size_t sourceIndex = 0; sourceIndex < objects.size(); ++sourceIndex)
        {
            if (objects[sourceIndex].id == objects[targetIndex].id)
            {
                continue;
            }

            const glm::dvec3 displacement =
                objects[sourceIndex].position - objects[targetIndex].position;
            const double distance = VectorMagnitude(displacement);
            if (distance == 0.0)
            {
                continue;
            }

            const double distanceMeters = distance * 1000.0;
            const double softenedDistanceSquared =
                distanceMeters * distanceMeters + epsilonMeters * epsilonMeters;
            const double accelerationMagnitude =
                gravitationalConstant * objects[sourceIndex].mass / softenedDistanceSquared;

            // a = G * sourceMass * direction / (distanceMeters^2 + epsilon^2).
            acceleration += displacement / distance * accelerationMagnitude;
        }
        if (!std::isfinite(acceleration.x) || !std::isfinite(acceleration.y) ||
            !std::isfinite(acceleration.z))
        {
            throw std::overflow_error("Direct N-body produced a non-finite acceleration");
        }
        accelerations[targetIndex] = acceleration;
    }
    return accelerations;
}

std::filesystem::path CreateBenchmarkRunDirectory()
{
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;
    const std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
#ifdef _WIN32
    localtime_s(&localTime, &currentTime);
#else
    localtime_r(&currentTime, &localTime);
#endif
    std::ostringstream timestamp;
    timestamp << std::put_time(&localTime, "%Y%m%d_%H%M%S") << '_' << std::setw(3)
              << std::setfill('0') << milliseconds;

    const std::filesystem::path resultRoot("numerical_error");
    std::filesystem::create_directories(resultRoot);
    for (std::size_t suffix = 0;; ++suffix)
    {
        const std::string runName = "run_" + timestamp.str() +
                                    (suffix == 0 ? "" : "_" + std::to_string(suffix));
        const std::filesystem::path runDirectory = resultRoot / runName;
        std::error_code error;
        if (std::filesystem::create_directory(runDirectory, error))
        {
            return runDirectory;
        }
        if (error && error != std::errc::file_exists)
        {
            throw std::filesystem::filesystem_error(
                "Unable to create benchmark result directory", runDirectory, error);
        }
    }
}
} // namespace

void RunOctreeConstruction(GLuint boundingBoxShaderProgram, GLuint octreeBuildShaderProgram,
                           const ComputeUniformLocations &locations, size_t objectCount,
                           bool validateDebug)
{
    const size_t objectCapacity = gpuMemoryManager.Get(BufferRole::CurrentState).Capacity();
    if (objectCount > objectCapacity ||
        objectCount > std::numeric_limits<std::uint32_t>::max() ||
        objectCapacity > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error("Octree object count exceeds its 32-bit GPU index range");
    }
    if (gpuMemoryManager.Get(BufferRole::CurrentState).LogicalCount() < objectCount ||
        gpuMemoryManager.Get(BufferRole::ObjectControl).LogicalCount() < objectCount)
    {
        throw std::runtime_error("Octree input buffers do not contain the requested object range");
    }

    glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
    glUseProgram(boundingBoxShaderProgram);
    glUniform1ui(locations.boundingBoxObjectCount, static_cast<GLuint>(objectCount));
    glUniform1ui(locations.boundingBoxObjectCapacity, static_cast<GLuint>(objectCapacity));
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(octreeBuildShaderProgram);
    glUniform1ui(locations.octreeBuildObjectCount, static_cast<GLuint>(objectCount));
    glUniform1ui(locations.octreeBuildObjectCapacity, static_cast<GLuint>(objectCapacity));
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

#if defined(_DEBUG) || !defined(NDEBUG)
    if (validateDebug)
    {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
        gpuMemoryManager.WaitForGpuCompletion();
        try
        {
            ValidateOctreeDebug(objectCount);
        }
        catch (const std::exception &error)
        {
            std::cerr << "Octree debug validation failed: " << error.what() << std::endl;
            throw;
        }
    }
#else
    (void)validateDebug;
#endif
}

std::string RunMassAggregationBenchmark(GLuint boundingBoxShaderProgram,
                                        GLuint octreeBuildShaderProgram,
                                        GLuint massAggregationShaderProgram)
{
    constexpr std::size_t repetitions = 3;
    const std::array<std::size_t, 3> objectCounts = {32, 128, 202};
    const std::array<const char *, 3> distributions = {"uniform", "clustered", "disk"};
    const std::filesystem::path runDirectory = CreateBenchmarkRunDirectory();
    const std::filesystem::path massDirectory = runDirectory / "mass_aggregation";
    std::filesystem::create_directories(massDirectory);

    std::ofstream metadataFile(runDirectory / "run_metadata.json");
    std::ofstream inputFile(massDirectory / "mass_aggregation_input.csv");
    std::ofstream nodeFile(massDirectory / "mass_aggregation.csv");
    std::ofstream summaryFile(massDirectory / "mass_aggregation_summary.csv");
    if (!metadataFile || !inputFile || !nodeFile || !summaryFile)
    {
        throw std::runtime_error("Unable to create mass aggregation benchmark output files");
    }

    metadataFile << "{\n"
                 << "  \"phase\": 1,\n"
                 << "  \"name\": \"mass_aggregation\",\n"
                 << "  \"base_seed\": " << kBenchmarkBaseSeed << ",\n"
                 << "  \"repetitions\": " << repetitions << ",\n"
                 << "  \"object_counts\": [32, 128, 202],\n"
                 << "  \"distributions\": [\"uniform\", \"clustered\", \"disk\"],\n"
                 << "  \"mass_range\": [1.0e20, 3.0e23],\n"
                 << "  \"uniform_position_range\": [-12000.0, 12000.0],\n"
                 << "  \"cluster_center_positions\": [[-5000.0, -2500.0, -3000.0], "
                    "[4500.0, 3500.0, -2000.0], [500.0, -4000.0, 5000.0]],\n"
                 << "  \"cluster_offset_standard_deviation\": 900.0,\n"
                 << "  \"disk_radius_range\": [1000.0, 12000.0],\n"
                 << "  \"disk_thickness_standard_deviation\": 80.0,\n"
                 << "  \"repetitions_use_identical_saved_input\": true,\n"
                 << "  \"comparison_tolerance_absolute\": 1.0e-6,\n"
                 << "  \"comparison_tolerance_relative\": 1.0e-5,\n"
                 << "  \"center_of_mass_relative_tolerance_scale\": "
                    "\"max(component_magnitude, node_diameter)\",\n"
                 << "  \"mass_unit\": \"simulation mass unit\",\n"
                 << "  \"position_unit\": \"simulation distance unit\",\n"
                 << "  \"cases\": [\n";
    bool firstCase = true;
    for (std::size_t distributionIndex = 0; distributionIndex < distributions.size();
         ++distributionIndex)
    {
        for (const std::size_t objectCount : objectCounts)
        {
            const std::uint32_t seed = BenchmarkSeed(distributionIndex, objectCount);
            if (!firstCase)
            {
                metadataFile << ",\n";
            }
            metadataFile << "    {\"distribution\": \"" << distributions[distributionIndex]
                         << "\", \"object_count\": " << objectCount << ", \"seed\": " << seed
                         << "}";
            firstCase = false;
        }
    }
    metadataFile << "\n  ]\n}\n";
    inputFile << "distribution,object_count,seed,object_id,mass,position_x,position_y,position_z\n";
    nodeFile << "distribution,object_count,seed,repeat,node_id,parent_node_id,is_leaf,child_count,"
                "child_node_ids,object_count_in_node,mass_gpu,mass_reference,mass_absolute_error,"
                "com_gpu_x,com_gpu_y,com_gpu_z,com_reference_x,com_reference_y,com_reference_z,"
                "com_absolute_error\n";
    summaryFile << "distribution,object_count,seed,repeat,node_count,leaf_count,"
                   "root_mass_absolute_error,root_com_absolute_error,max_mass_absolute_error,"
                   "max_com_absolute_error,status\n";
    nodeFile << std::setprecision(17);
    summaryFile << std::setprecision(17);

    ComputeUniformLocations locations;
    locations.boundingBoxObjectCount =
        glGetUniformLocation(boundingBoxShaderProgram, "u_objectCount");
    locations.boundingBoxObjectCapacity =
        glGetUniformLocation(boundingBoxShaderProgram, "u_objectCapacity");
    locations.octreeBuildObjectCount =
        glGetUniformLocation(octreeBuildShaderProgram, "u_objectCount");
    locations.octreeBuildObjectCapacity =
        glGetUniformLocation(octreeBuildShaderProgram, "u_objectCapacity");
    if (locations.boundingBoxObjectCount < 0 || locations.boundingBoxObjectCapacity < 0 ||
        locations.octreeBuildObjectCount < 0 || locations.octreeBuildObjectCapacity < 0)
    {
        throw std::runtime_error("Required Octree benchmark uniforms are unavailable");
    }

    for (std::size_t distributionIndex = 0; distributionIndex < distributions.size();
         ++distributionIndex)
    {
        const char *distribution = distributions[distributionIndex];
        for (const std::size_t objectCount : objectCounts)
        {
            const std::uint32_t seed = BenchmarkSeed(distributionIndex, objectCount);
            const std::vector<Object> caseObjects =
                CreateBenchmarkObjects(distribution, objectCount, seed);
            for (const Object &object : caseObjects)
            {
                inputFile << distribution << ',' << objectCount << ',' << seed << ',' << object.id
                          << ',' << std::setprecision(9) << object.mass << ',' << object.position.x
                          << ',' << object.position.y << ',' << object.position.z << '\n';
            }

            for (std::size_t repeat = 1; repeat <= repetitions; ++repeat)
            {
                objs = caseObjects;
                objectGpuState(0, objectCount);
                RunOctreeConstruction(boundingBoxShaderProgram, octreeBuildShaderProgram,
                                      locations, objectCount);
                RunMassAggregation(massAggregationShaderProgram, objectCount, false);
                glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
                gpuMemoryManager.WaitForGpuCompletion();

                const MassAggregationValidation validation =
                    ValidateMassAggregation(objectCount);
                double maxMassError = 0.0;
                double maxComError = 0.0;
                std::size_t rootNodeIndex = 0;
                for (const MassAggregationNodeResult &node : validation.nodes)
                {
                    maxMassError = std::max(maxMassError, node.massAbsoluteError);
                    maxComError = std::max(maxComError, node.comAbsoluteError);
                    if (node.parentIndex == std::numeric_limits<std::size_t>::max())
                    {
                        rootNodeIndex = node.nodeIndex;
                    }
                }
                for (const MassAggregationNodeResult &node : validation.nodes)
                {
                    nodeFile << distribution << ',' << objectCount << ',' << seed << ',' << repeat
                             << ',' << node.nodeIndex << ',';
                    if (node.parentIndex == std::numeric_limits<std::size_t>::max())
                    {
                        nodeFile << -1;
                    }
                    else
                    {
                        nodeFile << node.parentIndex;
                    }
                    nodeFile << ',' << (node.leaf ? 1 : 0) << ',' << node.childCount << ',';
                    for (std::size_t childOffset = 0; childOffset < node.childIndices.size();
                         ++childOffset)
                    {
                        if (childOffset != 0)
                        {
                            nodeFile << ';';
                        }
                        nodeFile << node.childIndices[childOffset];
                    }
                    nodeFile << ',' << node.objectCount << ',' << node.massGpu << ','
                             << node.massReference << ',' << node.massAbsoluteError << ','
                             << node.comGpu.x << ',' << node.comGpu.y << ',' << node.comGpu.z
                             << ',' << node.comReference.x << ',' << node.comReference.y << ','
                             << node.comReference.z << ',' << node.comAbsoluteError << '\n';
                }
                summaryFile << distribution << ',' << objectCount << ',' << seed << ',' << repeat
                            << ',' << validation.nodes.size() << ',' << validation.leafCount << ','
                            << validation.rootMassError << ',' << validation.rootComError << ','
                            << maxMassError << ',' << maxComError << ",PASS\n";
                std::cout << "Mass aggregation PASS: distribution=" << distribution
                          << ", objects=" << objectCount << ", repeat=" << repeat
                          << ", nodes=" << validation.nodes.size()
                          << ", leaves=" << validation.leafCount
                          << ", max_mass_error=" << maxMassError
                          << ", max_com_error=" << maxComError << std::endl;
            }
        }
    }

    objs.clear();
    objectGpuState(0, 0);
    RunOctreeConstruction(boundingBoxShaderProgram, octreeBuildShaderProgram, locations, 0);
    RunMassAggregation(massAggregationShaderProgram, 0, false);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    gpuMemoryManager.WaitForGpuCompletion();
    const MassAggregationValidation emptyValidation = ValidateMassAggregation(0);
    if (!emptyValidation.nodes.empty() || emptyValidation.activeObjectCount != 0)
    {
        throw std::runtime_error("Empty Octree mass aggregation validation returned node data");
    }
    summaryFile << "empty,0,0,1,0,0,0,0,0,0,PASS\n";
    std::cout << "Mass aggregation PASS: empty Octree, root aggregate is zero" << std::endl;

    if (!metadataFile.good() || !inputFile.good() || !nodeFile.good() || !summaryFile.good())
    {
        throw std::runtime_error("Failed while writing mass aggregation benchmark results");
    }
    return runDirectory.string();
}

std::string RunAccelerationBenchmark(const ShaderPrograms &shaderPrograms,
                                     const ComputeUniformLocations &locations)
{
    constexpr std::size_t repetitions = 3;
    constexpr double softeningEpsilonMeters = 10.0;
    constexpr double mapeReferenceMagnitudeThreshold = 1.0e-12;
    const std::array<std::size_t, 3> objectCounts = {32, 128, 202};
    const std::array<const char *, 3> distributions = {"uniform", "clustered", "disk"};
    const std::array<float, 3> thetaValues = {0.3f, 0.5f, 0.8f};
    const std::filesystem::path runDirectory = CreateBenchmarkRunDirectory();
    const std::filesystem::path accelerationDirectory = runDirectory / "acceleration";
    std::filesystem::create_directory(accelerationDirectory);

    std::ofstream metadataFile(runDirectory / "run_metadata.json");
    std::ofstream inputFile(accelerationDirectory / "acceleration_input.csv");
    std::ofstream objectFile(accelerationDirectory / "acceleration_per_object.csv");
    std::ofstream summaryFile(accelerationDirectory / "acceleration_summary.csv");
    if (!metadataFile || !inputFile || !objectFile || !summaryFile)
    {
        throw std::runtime_error("Unable to create acceleration benchmark output files");
    }

    metadataFile << "{\n"
                 << "  \"phase\": 2,\n"
                 << "  \"name\": \"acceleration_comparison\",\n"
                 << "  \"base_seed\": " << kBenchmarkBaseSeed << ",\n"
                 << "  \"repetitions\": " << repetitions << ",\n"
                 << "  \"object_counts\": [32, 128, 202],\n"
                 << "  \"distributions\": [\"uniform\", \"clustered\", \"disk\"],\n"
                 << "  \"theta_values\": [0.3, 0.5, 0.8],\n"
                 << "  \"gravitational_constant\": " << std::setprecision(17)
                 << kGravitationalConstant << ",\n"
                 << "  \"softening_epsilon_meters\": " << softeningEpsilonMeters << ",\n"
                 << "  \"mape_reference_magnitude_threshold_m_s2\": "
                 << mapeReferenceMagnitudeThreshold << ",\n"
                 << "  \"acceleration_unit\": \"m/s^2\",\n"
                 << "  \"distance_conversion_meters_per_simulation_unit\": 1000.0,\n"
                 << "  \"snapshot_reused_for_direct_and_barnes_hut\": true,\n"
                 << "  \"repetitions_use_identical_saved_input\": true,\n"
                 << "  \"cases\": [\n";
    bool firstCase = true;
    for (std::size_t distributionIndex = 0; distributionIndex < distributions.size();
         ++distributionIndex)
    {
        for (const std::size_t objectCount : objectCounts)
        {
            if (!firstCase)
            {
                metadataFile << ",\n";
            }
            metadataFile << "    {\"distribution\": \"" << distributions[distributionIndex]
                         << "\", \"object_count\": " << objectCount << ", \"seed\": "
                         << BenchmarkSeed(distributionIndex, objectCount) << "}";
            firstCase = false;
        }
    }
    metadataFile << "\n  ]\n}\n";
    inputFile << "distribution,object_count,seed,object_id,mass,position_x,position_y,position_z,"
                 "velocity_x,velocity_y,velocity_z\n";
    objectFile << "distribution,object_count,seed,repeat,theta,object_id,direct_ax,direct_ay,"
                  "direct_az,barnes_hut_ax,barnes_hut_ay,barnes_hut_az,direct_magnitude,"
                  "barnes_hut_magnitude,acceleration_vector_error,mape_sample_used\n";
    summaryFile << "distribution,object_count,seed,repeat,theta,acceleration_rmse,"
                   "acceleration_magnitude_mape_percent,mape_samples_used,mape_samples_excluded,"
                   "mape_reference_threshold_m_s2\n";
    inputFile << std::setprecision(9);
    objectFile << std::setprecision(17);
    summaryFile << std::setprecision(17);

    const auto *controls =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    const auto *accelerations =
        gpuMemoryManager.Get(BufferRole::ObjectAcceleration).MappedPtr<GpuObjectAcceleration>();
    if (controls == nullptr || accelerations == nullptr)
    {
        throw std::runtime_error("Object buffers are unavailable for acceleration benchmark");
    }

    for (std::size_t distributionIndex = 0; distributionIndex < distributions.size();
         ++distributionIndex)
    {
        const char *distribution = distributions[distributionIndex];
        for (const std::size_t objectCount : objectCounts)
        {
            const std::uint32_t seed = BenchmarkSeed(distributionIndex, objectCount);
            const std::vector<Object> snapshot =
                CreateBenchmarkObjects(distribution, objectCount, seed);
            // Both algorithms receive the same saved state; direct results are kept in ID order.
            const std::vector<TrajectoryObjectState> directObjects =
                CreateTrajectoryObjectStates(snapshot);
            const std::vector<glm::dvec3> directAccelerations =
                CalculateDirectAccelerations(directObjects, kGravitationalConstant,
                                             softeningEpsilonMeters);
            for (const Object &object : snapshot)
            {
                inputFile << distribution << ',' << objectCount << ',' << seed << ','
                          << object.id << ',' << object.mass << ',' << object.position.x << ','
                          << object.position.y << ',' << object.position.z << ','
                          << object.velocity.x << ',' << object.velocity.y << ','
                          << object.velocity.z << '\n';
            }

            for (std::size_t repeat = 1; repeat <= repetitions; ++repeat)
            {
                for (const float theta : thetaValues)
                {
                    objs = snapshot;
                    objectGpuState(0, objectCount);
                    RunOctreeConstruction(shaderPrograms.boundingBoxCompute,
                                          shaderPrograms.octreeBuildCompute, locations,
                                          objectCount);
                    RunMassAggregation(shaderPrograms.massAggregationCompute, objectCount, false);
                    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
                    gpuMemoryManager.WaitForGpuCompletion();
                    ValidateMassAggregation(objectCount);

                    RunBarnesHutCompute(shaderPrograms.barnesHutCompute, locations, objectCount,
                                        static_cast<float>(softeningEpsilonMeters), theta);
                    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                                    GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT |
                                    GL_SHADER_STORAGE_BARRIER_BIT);
                    gpuMemoryManager.WaitForGpuCompletion();

                    const GpuOctreeMetadata metadata = ReadOctreeMetadata();
                    if (static_cast<GpuOctreeBuildStatus>(metadata.configuration.w) !=
                        GpuOctreeBuildStatus::Valid)
                    {
                        throw std::runtime_error(
                            "Barnes-Hut failed during acceleration benchmark; status=" +
                            std::to_string(metadata.configuration.w));
                    }

                    std::vector<std::pair<std::uint32_t, glm::dvec3>> barnesHutAccelerations;
                    barnesHutAccelerations.reserve(objectCount);
                    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
                    {
                        const std::uint32_t objectId = controls[objectIndex].id;
                        const glm::vec4 value = accelerations[objectIndex].acceleration;
                        if (objectId == kInvalidObjectId ||
                            !std::isfinite(value.x) || !std::isfinite(value.y) ||
                            !std::isfinite(value.z))
                        {
                            throw std::runtime_error(
                                "Barnes-Hut produced invalid acceleration data or object IDs");
                        }
                        barnesHutAccelerations.emplace_back(objectId, glm::dvec3(value));
                    }
                    std::sort(barnesHutAccelerations.begin(), barnesHutAccelerations.end(),
                              [](const auto &left, const auto &right) {
                                  return left.first < right.first;
                              });
                    if (barnesHutAccelerations.size() != directAccelerations.size())
                    {
                        throw std::runtime_error(
                            "Direct and Barnes-Hut snapshots contain different object IDs");
                    }
                    for (std::size_t index = 1; index < barnesHutAccelerations.size(); ++index)
                    {
                        if (barnesHutAccelerations[index - 1].first ==
                            barnesHutAccelerations[index].first)
                        {
                            throw std::runtime_error(
                                "Barnes-Hut returned duplicate object IDs");
                        }
                    }

                    std::vector<glm::dvec3> directVectors;
                    std::vector<glm::dvec3> barnesHutVectors;
                    directVectors.reserve(objectCount);
                    barnesHutVectors.reserve(objectCount);
                    for (std::size_t index = 0; index < directObjects.size(); ++index)
                    {
                        if (barnesHutAccelerations[index].first != directObjects[index].id)
                        {
                            throw std::runtime_error(
                                "Direct and Barnes-Hut results have different object IDs");
                        }
                        directVectors.push_back(directAccelerations[index]);
                        barnesHutVectors.push_back(barnesHutAccelerations[index].second);
                    }

                    const double vectorRmse =
                        CalculateVectorRMSE(barnesHutVectors, directVectors);
                    const MagnitudeMAPEResult mape =
                        CalculateMagnitudeMAPE(barnesHutVectors, directVectors,
                                               mapeReferenceMagnitudeThreshold,
                                               &VectorMagnitude);
                    for (std::size_t index = 0; index < directObjects.size(); ++index)
                    {
                        const glm::dvec3 &direct = directVectors[index];
                        const glm::dvec3 &barnesHut = barnesHutVectors[index];
                        const double directMagnitude = VectorMagnitude(direct);
                        const double barnesHutMagnitude = VectorMagnitude(barnesHut);
                        const double vectorError = VectorError(barnesHut, direct);
                        objectFile << distribution << ',' << objectCount << ',' << seed << ','
                                   << repeat << ',' << theta << ',' << directObjects[index].id
                                   << ',' << direct.x << ',' << direct.y << ',' << direct.z << ','
                                   << barnesHut.x << ',' << barnesHut.y << ',' << barnesHut.z
                                   << ',' << directMagnitude << ',' << barnesHutMagnitude << ','
                                   << vectorError << ','
                                   << (directMagnitude > mapeReferenceMagnitudeThreshold ? 1 : 0)
                                   << '\n';
                    }
                    summaryFile << distribution << ',' << objectCount << ',' << seed << ','
                                << repeat << ',' << theta << ',' << vectorRmse << ',';
                    if (mape.percentage.has_value())
                    {
                        summaryFile << *mape.percentage;
                    }
                    summaryFile << ',' << mape.includedSamples << ',' << mape.excludedSamples
                                << ',' << mapeReferenceMagnitudeThreshold << '\n';

                    std::cout << "Acceleration error: objects=" << objectCount
                              << ", distribution=" << distribution << ", repeat=" << repeat
                              << ", theta=" << theta << ", vector_rmse=" << vectorRmse
                              << ", magnitude_mape_percent=";
                    if (mape.percentage.has_value())
                    {
                        std::cout << *mape.percentage;
                    }
                    else
                    {
                        std::cout << "unavailable";
                    }
                    std::cout << ", mape_used=" << mape.includedSamples
                              << ", mape_excluded=" << mape.excludedSamples << std::endl;
                }
            }
        }
    }

    if (!metadataFile.good() || !inputFile.good() || !objectFile.good() || !summaryFile.good())
    {
        throw std::runtime_error("Failed while writing acceleration benchmark results");
    }
    return runDirectory.string();
}

namespace
{
using TrajectoryHistory = std::vector<std::vector<TrajectoryObjectState>>;

TrajectoryHistory CalculateDirectTrajectory(const std::vector<Object> &initialObjects,
                                            int numberOfSteps, int sampleInterval,
                                            double timeStep, double epsilonMeters)
{
    std::vector<TrajectoryObjectState> currentState =
        CreateTrajectoryObjectStates(initialObjects);
    TrajectoryHistory history;
    history.push_back(currentState);

    for (int step = 1; step <= numberOfSteps; ++step)
    {
        const std::vector<glm::dvec3> accelerations =
            CalculateDirectAccelerations(currentState, kGravitationalConstant, epsilonMeters);
        std::vector<TrajectoryObjectState> nextState = currentState;

        for (std::size_t index = 0; index < currentState.size(); ++index)
        {
            glm::dvec3 velocity = currentState[index].velocity + accelerations[index] * timeStep;
            for (std::size_t otherIndex = 0; otherIndex < currentState.size(); ++otherIndex)
            {
                if (otherIndex == index)
                {
                    continue;
                }
                const double distance =
                    VectorMagnitude(currentState[otherIndex].position - currentState[index].position);
                if (distance > 0.0 &&
                    currentState[otherIndex].radius + currentState[index].radius > distance)
                {
                    velocity *= -0.2;
                }
            }

            nextState[index].velocity = velocity;
            // Semi-implicit Euler: update velocity first, then use it to update position.
            nextState[index].position =
                currentState[index].position + nextState[index].velocity * timeStep;
            if (!std::isfinite(nextState[index].position.x) ||
                !std::isfinite(nextState[index].position.y) ||
                !std::isfinite(nextState[index].position.z) ||
                !std::isfinite(nextState[index].velocity.x) ||
                !std::isfinite(nextState[index].velocity.y) ||
                !std::isfinite(nextState[index].velocity.z))
            {
                throw std::overflow_error(
                    "Direct trajectory became non-finite at step " + std::to_string(step) +
                    " for object ID " + std::to_string(currentState[index].id));
            }
        }
        currentState.swap(nextState);

        if (step % sampleInterval == 0)
        {
            history.push_back(currentState);
        }
    }
    return history;
}

std::vector<TrajectoryObjectState> CaptureGpuTrajectoryState(std::size_t objectCount)
{
    gpuMemoryManager.WaitForGpuCompletion();
    const auto *states =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    const auto *physical =
        gpuMemoryManager.Get(BufferRole::ObjectPhysical).MappedPtr<GpuObjectPhysical>();
    const auto *derived =
        gpuMemoryManager.Get(BufferRole::ObjectDerived).MappedPtr<GpuObjectDerived>();
    const auto *controls =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    if (states == nullptr || physical == nullptr || derived == nullptr || controls == nullptr)
    {
        throw std::runtime_error("GPU buffers are unavailable for trajectory capture");
    }

    std::vector<TrajectoryObjectState> result;
    result.reserve(objectCount);
    for (std::size_t index = 0; index < objectCount; ++index)
    {
        const glm::dvec3 position(states[index].position);
        const glm::dvec3 velocity(states[index].velocity);
        const double mass = physical[index].massDensity.x;
        const double radius = derived[index].radiusRs.x;
        if (controls[index].id == kInvalidObjectId || !std::isfinite(mass) || mass < 0.0 ||
            !std::isfinite(radius) || radius < 0.0 ||
            !std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
            !std::isfinite(velocity.y) || !std::isfinite(velocity.z))
        {
            throw std::runtime_error("GPU trajectory contains invalid state data");
        }
        result.push_back({controls[index].id, mass, radius, position, velocity});
    }

    std::sort(result.begin(), result.end(),
              [](const TrajectoryObjectState &left, const TrajectoryObjectState &right) {
                  return left.id < right.id;
              });
    for (std::size_t index = 1; index < result.size(); ++index)
    {
        if (result[index - 1].id == result[index].id)
        {
            throw std::runtime_error("GPU trajectory contains duplicate object IDs");
        }
    }
    return result;
}

void ValidateTrajectoryGpuStep(std::size_t objectCount)
{
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT);
    gpuMemoryManager.WaitForGpuCompletion();

    // Check the tree and its mass aggregates before accepting each sampled trajectory segment.
    ValidateMassAggregation(objectCount);
#if defined(_DEBUG) || !defined(NDEBUG)
    ValidateCollisionDebug(objectCount);
#endif
    const auto *accelerations =
        gpuMemoryManager.Get(BufferRole::ObjectAcceleration).MappedPtr<GpuObjectAcceleration>();
    if (accelerations == nullptr)
    {
        throw std::runtime_error("GPU acceleration buffer is unavailable for trajectory checks");
    }
    for (std::size_t index = 0; index < objectCount; ++index)
    {
        const glm::vec4 acceleration = accelerations[index].acceleration;
        if (!std::isfinite(acceleration.x) || !std::isfinite(acceleration.y) ||
            !std::isfinite(acceleration.z))
        {
            throw std::runtime_error("Barnes-Hut produced non-finite trajectory acceleration");
        }
    }
}

TrajectoryHistory CalculateBarnesHutTrajectory(const std::vector<Object> &initialObjects,
                                               const ShaderPrograms &shaderPrograms,
                                               const ComputeUniformLocations &locations,
                                               std::size_t objectCount, float theta,
                                               int numberOfSteps, int sampleInterval,
                                               double timeStep, double epsilonMeters)
{
    objs = initialObjects;
    objectGpuState(0, objectCount);

    // Reset both state buffers and collision counts to the same initial conditions.
    TrajectoryHistory history;
    history.reserve(static_cast<std::size_t>(numberOfSteps / sampleInterval + 1));
    history.push_back(CaptureGpuTrajectoryState(objectCount));

    for (int step = 0; step < numberOfSteps; ++step)
    {
        RunOctreeConstruction(shaderPrograms.boundingBoxCompute,
                              shaderPrograms.octreeBuildCompute, locations, objectCount, false);
        RunMassAggregation(shaderPrograms.massAggregationCompute, objectCount, false);
        RunBarnesHutCompute(shaderPrograms.barnesHutCompute, locations, objectCount,
                            static_cast<float>(epsilonMeters), theta, false);
        RunCollisionCompute(shaderPrograms.collisionCompute, locations, objectCount, false);

        if (step % sampleInterval == 0 || step == numberOfSteps - 1)
        {
            ValidateTrajectoryGpuStep(objectCount);
        }

        RunIntegration(shaderPrograms.integrateCompute, locations, objectCount, false, timeStep);
        gpuMemoryManager.SwapCurrentNext();

        if ((step + 1) % sampleInterval == 0)
        {
            history.push_back(CaptureGpuTrajectoryState(objectCount));
        }
    }
    return history;
}

void WriteTrajectoryComparison(std::ofstream &objectFile, std::ofstream &summaryFile,
                               const char *distribution, std::size_t objectCount,
                               std::uint32_t seed, std::size_t repeat, float theta, int step,
                               double time, double mapeReferenceMagnitudeThreshold,
                               const std::vector<TrajectoryObjectState> &directState,
                               const std::vector<TrajectoryObjectState> &barnesHutState)
{
    if (directState.size() != objectCount || barnesHutState.size() != objectCount)
    {
        throw std::runtime_error("Direct and Barnes-Hut trajectory sizes do not match");
    }

    std::vector<glm::dvec3> directPositions;
    std::vector<glm::dvec3> barnesHutPositions;
    std::vector<glm::dvec3> directVelocities;
    std::vector<glm::dvec3> barnesHutVelocities;
    directPositions.reserve(objectCount);
    barnesHutPositions.reserve(objectCount);
    directVelocities.reserve(objectCount);
    barnesHutVelocities.reserve(objectCount);

    for (std::size_t index = 0; index < objectCount; ++index)
    {
        if (directState[index].id != barnesHutState[index].id)
        {
            throw std::runtime_error("Trajectory comparison could not match object IDs");
        }
        directPositions.push_back(directState[index].position);
        barnesHutPositions.push_back(barnesHutState[index].position);
        directVelocities.push_back(directState[index].velocity);
        barnesHutVelocities.push_back(barnesHutState[index].velocity);
    }

    const double positionRmse = CalculateVectorRMSE(barnesHutPositions, directPositions);
    const MagnitudeMAPEResult positionMape =
        CalculateMagnitudeMAPE(barnesHutPositions, directPositions,
                               mapeReferenceMagnitudeThreshold, &VectorMagnitude);
    const double velocityRmse = CalculateVectorRMSE(barnesHutVelocities, directVelocities);

    for (std::size_t index = 0; index < objectCount; ++index)
    {
        const double directMagnitude = VectorMagnitude(directPositions[index]);
        const double barnesHutMagnitude = VectorMagnitude(barnesHutPositions[index]);
        objectFile << distribution << ',' << objectCount << ',' << seed << ',' << repeat << ','
                   << theta << ',' << step << ',' << time << ',' << directState[index].id << ','
                   << directPositions[index].x << ',' << directPositions[index].y << ','
                   << directPositions[index].z << ',' << barnesHutPositions[index].x << ','
                   << barnesHutPositions[index].y << ',' << barnesHutPositions[index].z << ','
                   << directVelocities[index].x << ',' << directVelocities[index].y << ','
                   << directVelocities[index].z << ',' << barnesHutVelocities[index].x << ','
                   << barnesHutVelocities[index].y << ',' << barnesHutVelocities[index].z << ','
                   << VectorError(barnesHutPositions[index], directPositions[index]) << ','
                   << std::abs(barnesHutMagnitude - directMagnitude) << ','
                   << VectorError(barnesHutVelocities[index], directVelocities[index]) << ','
                   << (directMagnitude > mapeReferenceMagnitudeThreshold ? 1 : 0) << '\n';
    }

    summaryFile << distribution << ',' << objectCount << ',' << seed << ',' << repeat << ','
                << theta << ',' << step << ',' << time << ',' << positionRmse << ',';
    if (positionMape.percentage.has_value())
    {
        summaryFile << *positionMape.percentage;
    }
    summaryFile << ',' << velocityRmse << ',' << positionMape.includedSamples << ','
                << positionMape.excludedSamples << ',' << mapeReferenceMagnitudeThreshold << '\n';

    std::cout << objectCount << ' ' << distribution << ' ' << repeat << ' ' << theta << ' ' << time
              << ' ' << positionRmse << ' ';
    if (positionMape.percentage.has_value())
    {
        std::cout << *positionMape.percentage;
    }
    else
    {
        std::cout << "unavailable";
    }
    std::cout << ' ' << velocityRmse << '\n';
}
} // namespace

std::string RunTrajectoryBenchmark(const ShaderPrograms &shaderPrograms,
                                   const ComputeUniformLocations &locations)
{
    constexpr int numberOfSteps = 960;
    constexpr int sampleInterval = 96;
    constexpr std::size_t repetitions = 3;
    constexpr double timeStep = 1.0 / 96.0;
    constexpr double totalTimeSeconds = 10.0;
    constexpr double softeningEpsilonMeters = 10.0;
    constexpr double mapeReferenceMagnitudeThreshold = 1.0e-12;
    const std::array<std::size_t, 3> objectCounts = {32, 128, 202};
    const std::array<const char *, 3> distributions = {"uniform", "clustered", "disk"};
    const std::array<float, 3> thetaValues = {0.3f, 0.5f, 0.8f};
    const std::filesystem::path runDirectory = CreateBenchmarkRunDirectory();
    const std::filesystem::path trajectoryDirectory = runDirectory / "trajectory";
    std::filesystem::create_directory(trajectoryDirectory);

    std::ofstream metadataFile(runDirectory / "run_metadata.json");
    std::ofstream inputFile(trajectoryDirectory / "trajectory_input.csv");
    std::ofstream objectFile(trajectoryDirectory / "trajectory_per_object.csv");
    std::ofstream summaryFile(trajectoryDirectory / "trajectory_summary.csv");
    if (!metadataFile || !inputFile || !objectFile || !summaryFile)
    {
        throw std::runtime_error("Unable to create trajectory benchmark output files");
    }

    metadataFile << "{\n"
                 << "  \"phase\": 3,\n"
                 << "  \"name\": \"trajectory_comparison\",\n"
                 << "  \"base_seed\": " << kBenchmarkBaseSeed << ",\n"
                 << "  \"repetitions\": " << repetitions << ",\n"
                 << "  \"object_counts\": [32, 128, 202],\n"
                 << "  \"distributions\": [\"uniform\", \"clustered\", \"disk\"],\n"
                 << "  \"theta_values\": [0.3, 0.5, 0.8],\n"
                 << "  \"time_step_seconds\": " << timeStep << ",\n"
                 << "  \"total_time_seconds\": " << totalTimeSeconds << ",\n"
                 << "  \"step_count\": " << numberOfSteps << ",\n"
                 << "  \"sample_interval_steps\": " << sampleInterval << ",\n"
                 << "  \"gravitational_constant\": " << std::setprecision(17)
                 << kGravitationalConstant << ",\n"
                 << "  \"softening_epsilon_meters\": " << softeningEpsilonMeters << ",\n"
                 << "  \"position_mape_reference_threshold_simulation_units\": "
                 << mapeReferenceMagnitudeThreshold << ",\n"
                 << "  \"position_unit\": \"simulation distance unit\",\n"
                 << "  \"velocity_unit\": \"simulation distance unit per second\",\n"
                 << "  \"integrator\": \"semi_implicit_euler\",\n"
                 << "  \"collision_velocity_scale_per_overlap\": -0.2,\n"
                 << "  \"direct_state_precision\": \"double\",\n"
                 << "  \"barnes_hut_gpu_state_precision\": \"float\",\n"
                 << "  \"snapshot_reused_for_direct_and_barnes_hut\": true,\n"
                 << "  \"repetitions_use_identical_saved_input\": true,\n"
                 << "  \"cases\": [\n";
    bool firstCase = true;
    for (std::size_t distributionIndex = 0; distributionIndex < distributions.size();
         ++distributionIndex)
    {
        for (const std::size_t objectCount : objectCounts)
        {
            if (!firstCase)
            {
                metadataFile << ",\n";
            }
            metadataFile << "    {\"distribution\": \"" << distributions[distributionIndex]
                         << "\", \"object_count\": " << objectCount << ", \"seed\": "
                         << BenchmarkSeed(distributionIndex, objectCount) << "}";
            firstCase = false;
        }
    }
    metadataFile << "\n  ]\n}\n";
    inputFile << "distribution,object_count,seed,object_id,mass,position_x,position_y,position_z,"
                 "velocity_x,velocity_y,velocity_z\n";
    objectFile << "distribution,object_count,seed,repeat,theta,step,time_seconds,object_id,"
                  "direct_px,direct_py,direct_pz,barnes_hut_px,barnes_hut_py,barnes_hut_pz,"
                  "direct_vx,direct_vy,direct_vz,barnes_hut_vx,barnes_hut_vy,barnes_hut_vz,"
                  "position_vector_error,position_magnitude_error,velocity_vector_error,"
                  "position_mape_sample_used\n";
    summaryFile << "distribution,object_count,seed,repeat,theta,step,time_seconds,position_rmse,"
                   "position_magnitude_mape_percent,velocity_rmse,mape_samples_used,"
                   "mape_samples_excluded,mape_reference_threshold_simulation_units\n";
    inputFile << std::setprecision(9);
    objectFile << std::setprecision(17);
    summaryFile << std::setprecision(17);
    std::cout << "Trajectory summary: object_count distribution repeat theta time_s position_RMSE "
                 "position_MAPE_percent velocity_RMSE\n";
    std::cout << std::setprecision(4);

    for (std::size_t distributionIndex = 0; distributionIndex < distributions.size();
         ++distributionIndex)
    {
        const char *distribution = distributions[distributionIndex];
        for (const std::size_t objectCount : objectCounts)
        {
            const std::uint32_t seed = BenchmarkSeed(distributionIndex, objectCount);
            const std::vector<Object> initialObjects =
                CreateBenchmarkObjects(distribution, objectCount, seed);
            for (const Object &object : initialObjects)
            {
                inputFile << distribution << ',' << objectCount << ',' << seed << ','
                          << object.id << ',' << object.mass << ',' << object.position.x << ','
                          << object.position.y << ',' << object.position.z << ','
                          << object.velocity.x << ',' << object.velocity.y << ','
                          << object.velocity.z << '\n';
            }

            for (std::size_t repeat = 1; repeat <= repetitions; ++repeat)
            {
                const TrajectoryHistory directHistory =
                    CalculateDirectTrajectory(initialObjects, numberOfSteps, sampleInterval,
                                              timeStep, softeningEpsilonMeters);

                for (const float theta : thetaValues)
                {
                    const TrajectoryHistory barnesHutHistory =
                        CalculateBarnesHutTrajectory(initialObjects, shaderPrograms, locations,
                                                     objectCount, theta, numberOfSteps,
                                                     sampleInterval, timeStep,
                                                     softeningEpsilonMeters);
                    if (directHistory.size() != barnesHutHistory.size())
                    {
                        throw std::runtime_error(
                            "Direct and Barnes-Hut trajectory sample counts do not match");
                    }

                    for (std::size_t sampleIndex = 0; sampleIndex < directHistory.size();
                         ++sampleIndex)
                    {
                        const int step = static_cast<int>(sampleIndex) * sampleInterval;
                        const double time = static_cast<double>(step) * timeStep;
                        WriteTrajectoryComparison(
                            objectFile, summaryFile, distribution, objectCount, seed, repeat,
                            theta, step, time, mapeReferenceMagnitudeThreshold,
                            directHistory[sampleIndex], barnesHutHistory[sampleIndex]);
                    }
                }
            }
        }
    }

    if (!metadataFile.good() || !inputFile.good() || !objectFile.good() || !summaryFile.good())
    {
        throw std::runtime_error("Failed while writing trajectory benchmark results");
    }
    return runDirectory.string();
}

void RunMassAggregation(GLuint computeShaderProgram, size_t objectCount, bool validateDebug)
{
    glUseProgram(computeShaderProgram);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

#if defined(_DEBUG) || !defined(NDEBUG)
    if (validateDebug)
    {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT |
                        GL_SHADER_STORAGE_BARRIER_BIT);
        gpuMemoryManager.WaitForGpuCompletion();
        try
        {
            ValidateMassAggregation(objectCount);
        }
        catch (const std::exception &error)
        {
            std::cerr << "Mass aggregation debug validation failed: " << error.what() << std::endl;
            throw;
        }
    }
#else
    (void)objectCount;
    (void)validateDebug;
#endif
}

void RunBarnesHutCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                         size_t objectCount, float epsilon, float theta, bool validateDebug)
{
    if (objectCount == 0)
    {
        return;
    }
    if (objectCount > std::numeric_limits<GLuint>::max() || !(theta > 0.0f) ||
        !std::isfinite(theta) || !(epsilon >= 0.0f) || !std::isfinite(epsilon))
    {
        throw std::invalid_argument("Invalid Barnes-Hut object count, theta, or softening value");
    }

    constexpr GLuint localSizeX = 64;
    const GLuint groupCountX =
        static_cast<GLuint>((objectCount + localSizeX - 1) / localSizeX);
    glUseProgram(computeShaderProgram);
    glUniform1ui(locations.barnesHutObjectCount, static_cast<GLuint>(objectCount));
    glUniform1d(locations.barnesHutGravitationalConstant, kGravitationalConstant);
    glUniform1f(locations.barnesHutEpsilon, epsilon);
    glUniform1f(locations.barnesHutTheta, theta);
    glDispatchCompute(groupCountX, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

#if defined(_DEBUG) || !defined(NDEBUG)
    if (validateDebug)
    {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT |
                        GL_SHADER_STORAGE_BARRIER_BIT);
        gpuMemoryManager.WaitForGpuCompletion();
        try
        {
            ValidateBarnesHutDebug(objectCount);
        }
        catch (const std::exception &error)
        {
            std::cerr << "Barnes-Hut debug validation failed: " << error.what() << std::endl;
            throw;
        }
    }
#else
    (void)validateDebug;
#endif
}

void RunCollisionCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                         size_t objectCount, bool validateDebug)
{
    if (objectCount == 0)
    {
        return;
    }
    constexpr GLuint localSizeX = 64;
    const GLuint groupCountX =
        static_cast<GLuint>((objectCount + localSizeX - 1) / localSizeX);
    glUseProgram(computeShaderProgram);
    glUniform1ui(locations.collisionObjectCount, static_cast<GLuint>(objectCount));
    glDispatchCompute(groupCountX, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

#if defined(_DEBUG) || !defined(NDEBUG)
    if (validateDebug)
    {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT |
                        GL_SHADER_STORAGE_BARRIER_BIT);
        gpuMemoryManager.WaitForGpuCompletion();
        try
        {
            ValidateCollisionDebug(objectCount);
        }
        catch (const std::exception &error)
        {
            std::cerr << "Collision debug validation failed: " << error.what() << std::endl;
            throw;
        }
    }
#else
    (void)validateDebug;
#endif
}

void RunIntegration(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                    size_t objectCount, bool paused, double timeStep)
{
    if (objectCount == 0)
    {
        return;
    }

    constexpr GLuint localSizeX = 64;
    const GLuint groupCountX = static_cast<GLuint>((objectCount + localSizeX - 1) / localSizeX);
    glUseProgram(computeShaderProgram);
    glUniform1ui(locations.integrateObjectCount, static_cast<GLuint>(objectCount));
    glUniform1ui(locations.integratePaused, static_cast<GLuint>(paused));
    glUniform1f(locations.integrateDt, static_cast<float>(timeStep));
    glDispatchCompute(groupCountX, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void RunObjectStateSync(GLFWwindow *window, size_t &objectCount)
{
    // 1. Update CPU-side properties of the object being created (mass/rs/radius).
    UpdateInitializingObject(window);

    // 2. Resize/materialize GPU object buffers when the object count changes.
    const size_t currentObjectCount = objs.size();
    if (currentObjectCount != objectCount)
    {
        SynchronizeObjectStateBuffers(objectCount, currentObjectCount);
    }

    // 3. Push the initializing object's properties to the GPU. Must run after step 2 because
    //    SynchronizeObjectGpuProperties skips indices beyond the buffers' LogicalCount.
    if (!objs.empty() && objs.back().initializing)
    {
        SynchronizeObjectGpuProperties(objs.size() - 1);
    }
}

void RunGpuStateTransition(GLuint nbodyShaderProgram, GLuint integrateShaderProgram,
                           const ComputeUniformLocations &locations, size_t objectCount,
                           float epsilon, bool paused, double timeStep)
{
    if (objectCount == 0)
    {
        return;
    }

    // Ensure that any CPU writes to persistently mapped buffers are visible to the GPU before running
    glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
    // Run the N-body compute shader to calculate accelerations based on current positions and masses.
    RunNBodyCompute(nbodyShaderProgram, locations, objectCount, epsilon);
    // Run the integration compute shader
    RunIntegration(integrateShaderProgram, locations, objectCount, paused, timeStep);
    // Ping-pong the current and next state buffers for the next frame's computations.
    gpuMemoryManager.SwapCurrentNext();

    static bool reportedRoles = false;
    if (!reportedRoles)
    {
        std::cout << "GPU state roles after first transition: CurrentState buffer="
                  << gpuMemoryManager.Get(BufferRole::CurrentState).Handle()
                  << ", NextState buffer=" << gpuMemoryManager.Get(BufferRole::NextState).Handle()
                  << std::endl;
        reportedRoles = true;
    }
}

void RunBarnesHutStateTransition(GLuint barnesHutShaderProgram, GLuint collisionShaderProgram,
                                GLuint integrateShaderProgram,
                                const ComputeUniformLocations &locations, size_t objectCount,
                                float epsilon, float theta, bool paused, double timeStep)
{
    if (objectCount == 0)
    {
        return;
    }

    glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
    RunBarnesHutCompute(barnesHutShaderProgram, locations, objectCount, epsilon, theta);
    RunCollisionCompute(collisionShaderProgram, locations, objectCount);
    RunIntegration(integrateShaderProgram, locations, objectCount, paused, timeStep);
    gpuMemoryManager.SwapCurrentNext();

    static bool reportedRoles = false;
    if (!reportedRoles)
    {
        std::cout << "GPU Barnes-Hut state roles after first transition: CurrentState buffer="
                  << gpuMemoryManager.Get(BufferRole::CurrentState).Handle()
                  << ", NextState buffer="
                  << gpuMemoryManager.Get(BufferRole::NextState).Handle() << std::endl;
        reportedRoles = true;
    }
}

void CaptureGpuTrajectorySample(std::vector<std::vector<glm::vec3>> &positions)
{
    gpuMemoryManager.WaitForGpuCompletion();
    const auto *gpuState =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    if (gpuState == nullptr)
    {
        throw std::runtime_error("CurrentState mapping is unavailable for trajectory export");
    }

    const size_t gpuObjectCount = gpuMemoryManager.Get(BufferRole::CurrentState).LogicalCount();
    positions.emplace_back();
    positions.back().reserve(gpuObjectCount);
    for (size_t index = 0; index < gpuObjectCount; ++index)
    {
        positions.back().emplace_back(gpuState[index].position);
    }
}

void RunGridCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                    size_t objectCount)
{
    glUseProgram(computeShaderProgram);
    glUniform1ui(locations.gridObjectCount, static_cast<GLuint>(objectCount));
    const GLuint gridWidth = static_cast<GLuint>(gridDivisions + 1);
    const GLuint gridHeight = static_cast<GLuint>(gridDivisions + 1);
    glUniform1ui(locations.gridWidth, gridWidth);
    glUniform1ui(locations.gridHeight, gridHeight);
    // Round up work group counts
    const GLuint groupCountX = (gridWidth + kGridLocalSizeX - 1) / kGridLocalSizeX;
    const GLuint groupCountY = (gridHeight + kGridLocalSizeY - 1) / kGridLocalSizeY;
    // Dispatch the compute shader with the calculated number of work groups.
    glDispatchCompute(groupCountX, groupCountY, 1);

    // grid.comp writes DeformedGrid as an SSBO; the grid VAO fetches the same buffer as vertex
    // attributes. This barrier orders that compute write -> vertex fetch dependency on the GPU.
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

#if defined(_DEBUG) || !defined(NDEBUG)
    static bool gridValidated = false;
    if (!gridValidated)
    {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        gpuMemoryManager.WaitForGpuCompletion();
        try
        {
            ValidateGridDeformationDebug(objectCount,
                                         static_cast<std::size_t>(gridWidth) * gridHeight);
        }
        catch (const std::exception &error)
        {
            std::cerr << "Grid deformation debug validation failed: " << error.what()
                      << std::endl;
            throw;
        }
        gridValidated = true;
    }
#endif
}

void Cleanup(const ShaderPrograms &shaderPrograms)
{
    DestroyObjectRenderingResources();
    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridEBO);
    gpuMemoryManager.Shutdown();
    glDeleteProgram(shaderPrograms.gridRendering);
    glDeleteProgram(shaderPrograms.gridCompute);
    glDeleteProgram(shaderPrograms.boundingBoxCompute);
    glDeleteProgram(shaderPrograms.octreeBuildCompute);
    glDeleteProgram(shaderPrograms.massAggregationCompute);
    glDeleteProgram(shaderPrograms.barnesHutCompute);
    glDeleteProgram(shaderPrograms.collisionCompute);
    glDeleteProgram(shaderPrograms.objectRendering);
    glDeleteProgram(shaderPrograms.nbodyCompute);
    glDeleteProgram(shaderPrograms.integrateCompute);

    objs.clear();
    gridVAO = 0;
    gridEBO = 0;
    objectStateCapacity = 0;
    gridNodeCount = 0;
    gridIndexCount = 0;

    glfwTerminate();
}

void BeginFrame() { glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }

void UpdateInitializingObject(GLFWwindow *window)
{
    if (objs.empty() || !objs.back().initializing)
        return;

    Object &initializingObject = objs.back();
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        initializingObject.mass *= 1.0 + 1.0 * deltaTime;
        initializingObject.rs = (2 * kGravitationalConstant * initializingObject.mass) /
                                (kSpeedOfLight * kSpeedOfLight);
    }

    initializingObject.radius =
        pow((3 * initializingObject.mass / initializingObject.density) / (4 * 3.14159265359f),
            1.0f / 3.0f) /
        100000.0f;
}

void UpdateCamera(const ShaderPrograms &shaderPrograms,
                  const ShaderUniformLocations &uniformLocations, glm::vec3 cameraPosition)
{
    const glm::mat4 view = glm::lookAt(cameraPosition, cameraPosition + cameraFront, cameraUp);

    glUseProgram(shaderPrograms.gridRendering);
    glUniformMatrix4fv(uniformLocations.gridView, 1, GL_FALSE, glm::value_ptr(view));

    glUseProgram(shaderPrograms.objectRendering);
    glUniformMatrix4fv(uniformLocations.objectView, 1, GL_FALSE, glm::value_ptr(view));
}

void KeyCallback(GLFWwindow *window, int key, int scanCode, int action, int mods)
{
    (void)scanCode;
    float cameraSpeed = 1000.0f * deltaTime;
    bool shiftPressed = (mods & GLFW_MOD_SHIFT) != 0;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= cameraSpeed * glm::normalize(glm::cross(cameraFront, cameraUp));
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += cameraSpeed * glm::normalize(glm::cross(cameraFront, cameraUp));
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraUp;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraUp;

    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        pause = true;
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_RELEASE)
        pause = false;

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        glfwTerminate();
        glfwWindowShouldClose(window);
        running = false;
    }

    if (!objs.empty() && objs.back().initializing)
    {
        Object &lastObj = objs.back();
        bool positionChanged = false;
        if (key == GLFW_KEY_UP && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            if (!shiftPressed)
            {
                lastObj.position[1] += 0.5;
            }
            else
            {
                lastObj.position[2] += 0.5;
            }
            positionChanged = true;
        }
        if (key == GLFW_KEY_DOWN && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            if (!shiftPressed)
            {
                lastObj.position[1] -= 0.5;
            }
            else
            {
                lastObj.position[2] -= 0.5;
            }
            positionChanged = true;
        }
        if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            lastObj.position[0] += 0.5;
            positionChanged = true;
        }
        if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            lastObj.position[0] -= 0.5;
            positionChanged = true;
        }
        if (positionChanged)
        {
            UploadInitializingObjectState(objs.size() - 1);
        }
    }
}

void MouseCallback(GLFWwindow *window, double xPosition, double yPosition)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS)
    {
        lastX = static_cast<float>(xPosition);
        lastY = static_cast<float>(yPosition);
        return;
    }
    // 0----> +x
    // |
    // |
    // +y
    float xOffset = xPosition - lastX;
    float yOffset = lastY - yPosition;
    lastX = xPosition;
    lastY = yPosition;
    // Adjust the sensitivity of the mouse movement for smoother camera rotation.
    float sensitivity = 0.1f;
    xOffset *= sensitivity;
    yOffset *= sensitivity;
    // Update the yaw and pitch angles based on mouse movement.
    yaw += xOffset;
    pitch += yOffset;
    // Constrain the pitch angle to prevent the camera from flipping upside down.
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    (void)mods;
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        double xPosition = 0.0;
        double yPosition = 0.0;
        glfwGetCursorPos(window, &xPosition, &yPosition);
        lastX = static_cast<float>(xPosition);
        lastY = static_cast<float>(yPosition);
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            objs.emplace_back(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0f, 0.0f, 0.0f), initMass);
            objs.back().initializing = true;
        }
        if (action == GLFW_RELEASE && !objs.empty())
        {
            objs.back().initializing = false;
            objs.back().launched = true;
            SynchronizeObjectGpuProperties(objs.size() - 1);
        }
    }
}

void ScrollCallback(GLFWwindow *window, double xOffset, double yOffset)
{
    (void)window;
    (void)xOffset;
    float cameraSpeed = 50000.0f * deltaTime;
    if (yOffset > 0)
        cameraPos += cameraSpeed * cameraFront;
    else if (yOffset < 0)
        cameraPos -= cameraSpeed * cameraFront;
}

glm::vec3 SphericalToCartesian(float radius, float theta, float phi)
{
    float x = radius * sin(theta) * cos(phi);
    float y = radius * cos(theta);
    float z = radius * sin(theta) * sin(phi);
    return glm::vec3(x, y, z);
}

void DrawGrid(GLuint renderingShaderProgram, GLuint gridVAO, size_t indexCount,
              const ShaderUniformLocations &uniformLocations)
{
    glUseProgram(renderingShaderProgram);
    glUniform4f(uniformLocations.gridObjectColor, 1.0f, 1.0f, 1.0f, 0.25f);
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(uniformLocations.gridModel, 1, GL_FALSE, glm::value_ptr(model));
    glBindVertexArray(gridVAO);
    // glPointSize(5.0f);
    glDrawElements(GL_LINES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void DrawObjects(GLuint objectRenderingShaderProgram, const std::vector<Object> &objects)
{
    // Check conditions before rendering to avoid OpenGL errors and ensure valid state.
    if (objects.empty())
    {
        return;
    }

    if (objectMesh.objectVAO == 0 || objectMesh.vertexCount <= 0)
    {
        throw std::runtime_error("Shared object rendering geometry is unavailable");
    }
    if (!gpuMemoryManager.Contains(BufferRole::CurrentState) ||
        !gpuMemoryManager.Contains(BufferRole::ObjectDerived) ||
        !gpuMemoryManager.Contains(BufferRole::ObjectRender))
    {
        throw std::runtime_error("Object rendering buffers are unavailable");
    }

    if (objects.size() > static_cast<size_t>(std::numeric_limits<GLsizei>::max()))
    {
        throw std::length_error("Object instance count exceeds OpenGL limits");
    }
    const size_t validStateCount = gpuMemoryManager.Get(BufferRole::CurrentState).LogicalCount();
    const size_t validDerivedCount = gpuMemoryManager.Get(BufferRole::ObjectDerived).LogicalCount();
    const size_t validRenderCount = gpuMemoryManager.Get(BufferRole::ObjectRender).LogicalCount();
    if (objects.size() > validStateCount || objects.size() > validDerivedCount ||
        objects.size() > validRenderCount)
    {
        throw std::out_of_range("Object instance count exceeds valid GPU records");
    }
    // Check GPU state indices match the order of objects in the vector to ensure correct rendering.
    for (size_t index = 0; index < objects.size(); ++index)
    {
        if (objects[index].gpuStateIndex != index)
        {
            throw std::logic_error("Object instance order does not match packed GPU state slots");
        }
    }

    glUseProgram(objectRenderingShaderProgram);
    gpuMemoryManager.Bind(BufferRole::CurrentState);
    gpuMemoryManager.Bind(BufferRole::ObjectDerived);
    gpuMemoryManager.Bind(BufferRole::ObjectRender);
    glBindVertexArray(objectMesh.objectVAO);
    glDrawArraysInstanced(GL_TRIANGLES, 0, objectMesh.vertexCount,
                          static_cast<GLsizei>(objects.size()));
    glBindVertexArray(0);
}

std::vector<float> CreateGridVertices(float size, int divisions)
{
    std::vector<float> vertices;
    int numNodesPerSide = divisions + 1;
    vertices.reserve(numNodesPerSide * numNodesPerSide * 3);

    float step = size / divisions;
    float halfSize = size / 2.0f;
    float baseY = -800.0f;

    for (int zStep = 0; zStep <= divisions; ++zStep)
    {
        float z = -halfSize + zStep * step;
        for (int xStep = 0; xStep <= divisions; ++xStep)
        {
            float x = -halfSize + xStep * step;
            vertices.push_back(x);
            vertices.push_back(baseY);
            vertices.push_back(z);
        }
    }
    return vertices;
}

std::vector<unsigned int> CreateGridIndices(int divisions)
{
    std::vector<unsigned int> indices;
    int rowSize = divisions + 1;
    size_t totalIndices = 4 * divisions * (divisions + 1);
    indices.reserve(totalIndices);

    for (int z = 0; z <= divisions; ++z)
    {
        for (int x = 0; x < divisions; ++x)
        {
            int current = z * rowSize + x;
            indices.push_back(current);
            indices.push_back(current + 1);
        }
    }

    for (int z = 0; z < divisions; ++z)
    {
        for (int x = 0; x <= divisions; ++x)
        {
            int current = z * rowSize + x;
            indices.push_back(current);
            indices.push_back(current + rowSize);
        }
    }

    return indices;
}

std::vector<Object> CreateRandomOrbiters(int count, const glm::vec3 &center, float centralMass)
{
    (void)centralMass;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> radiusDist(4000.0f, 12000.0f);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * glm::pi<float>());
    std::uniform_real_distribution<float> massDist(1.0e20f, 3.0e23f);
    std::uniform_real_distribution<float> speedDist(120.0f, 260.0f);
    std::uniform_real_distribution<float> tiltDist(-0.15f, 0.15f);

    std::vector<Object> generated;
    generated.reserve(count);

    for (int i = 0; i < count; ++i)
    {
        float radius = radiusDist(rng);
        float angle = angleDist(rng);
        float tilt = tiltDist(rng);
        float orbitalSpeed = speedDist(rng);

        glm::vec3 position(center.x + radius * std::cos(angle), center.y + radius * tilt,
                           center.z + radius * std::sin(angle));
        glm::vec3 velocity(-std::sin(angle) * orbitalSpeed, 0.0f, std::cos(angle) * orbitalSpeed);

        float mass = massDist(rng);
        generated.emplace_back(position, velocity, mass, 3344.0f);
        generated.back().color =
            glm::vec4(0.2f + 0.8f * static_cast<float>((i * 7) % 5) / 4.0f,
                      0.2f + 0.8f * static_cast<float>((i * 11) % 5) / 4.0f,
                      0.2f + 0.8f * static_cast<float>((i * 13) % 5) / 4.0f, 1.0f);
    }

    return generated;
}

std::vector<Object> CreateObjects(const std::string &objectFilePath, int randomObjectCount)
{
    if (!objectFilePath.empty())
    {
        std::ifstream objectFile(objectFilePath);
        if (!objectFile)
        {
            throw std::runtime_error("Unable to open object file: " + objectFilePath);
        }

        std::vector<Object> loadedObjects;
        std::string line;
        size_t lineNumber = 0;
        while (std::getline(objectFile, line))
        {
            ++lineNumber;
            if (line.empty() || line.front() == '#')
            {
                continue;
            }

            std::stringstream values(line);
            char separator = 0;
            float positionX = 0.0f;
            float positionY = 0.0f;
            float positionZ = 0.0f;
            float velocityX = 0.0f;
            float velocityY = 0.0f;
            float velocityZ = 0.0f;
            float mass = 0.0f;
            float density = 3344.0f;

            if (!(values >> positionX >> separator && separator == ',' &&
                  values >> positionY >> separator && separator == ',' &&
                  values >> positionZ >> separator && separator == ',' &&
                  values >> velocityX >> separator && separator == ',' &&
                  values >> velocityY >> separator && separator == ',' &&
                  values >> velocityZ >> separator && separator == ',' &&
                  values >> mass >> separator && separator == ',' && values >> density))
            {
                throw std::runtime_error("Invalid object data at line " +
                                         std::to_string(lineNumber) + " in " + objectFilePath);
            }

            loadedObjects.emplace_back(glm::vec3(positionX, positionY, positionZ),
                                       glm::vec3(velocityX, velocityY, velocityZ), mass, density);
        }
        return loadedObjects;
    }

    std::vector<Object> defaultObjects = {
        Object(glm::vec3(3844, 0, 0), glm::vec3(0, 0, 228), 7.34767309e22f, 3344),
        Object(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), 5.97219e24f, 5515)};
    auto randomBodies =
        CreateRandomOrbiters(randomObjectCount, glm::vec3(0.0f, 0.0f, 0.0f), 5.97219e24f);
    defaultObjects.insert(defaultObjects.end(), randomBodies.begin(), randomBodies.end());
    return defaultObjects;
}

// GLuint CreateObjectCellID(const std::vector<Object> &objects, GLuint &cellIDBuffer)
// {
//     std::vector<GLuint> cellIDs(objects.size());
//     for (size_t i = 0; i < objects.size(); ++i)
//     {
//         const Object &obj = objects[i];
//         glm::ivec3 cellIndex = glm::ivec3(
//             static_cast<int>((obj.position.x + gridSize / 2.0f) / (gridSize / gridDivisions)),
//             static_cast<int>((obj.position.y + gridSize / 2.0f) / (gridSize / gridDivisions)),
//             static_cast<int>((obj.position.z + gridSize / 2.0f) / (gridSize / gridDivisions)));
//         cellIDs[i] = static_cast<GLuint>(cellIndex.x + cellIndex.y * (gridDivisions + 1) +
//                                           cellIndex.z * (gridDivisions + 1) * (gridDivisions +
//                                           1));
//     }

//     glGenBuffers(1, &cellIDBuffer);
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellIDBuffer);
//     glBufferData(GL_SHADER_STORAGE_BUFFER, cellIDs.size() * sizeof(GLuint), cellIDs.data(),
//                  GL_STATIC_DRAW);
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, cellIDBuffer);

//     return cellIDBuffer;
// }