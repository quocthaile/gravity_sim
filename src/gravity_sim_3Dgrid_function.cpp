#include "gravity_sim_3Dgrid_function.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

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
float MeasureGpuMilliseconds(const std::function<void()> &commands)
{
    GLuint query = 0;
    glGenQueries(1, &query);
    if (query == 0)
    {
        throw std::runtime_error("Unable to create an OpenGL timer query");
    }
    glBeginQuery(GL_TIME_ELAPSED, query);
    commands();
    glEndQuery(GL_TIME_ELAPSED);
    GLuint64 elapsedNanoseconds = 0;
    glGetQueryObjectui64v(query, GL_QUERY_RESULT, &elapsedNanoseconds);
    glDeleteQueries(1, &query);
    return static_cast<float>(static_cast<double>(elapsedNanoseconds) / 1.0e6);
}

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

GLFWwindow *StartGLU()
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
                     size_t objectCount, float epsilon, float *gpuMilliseconds)
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

    const auto dispatchNBody = [groupCountX]
    {
        glDispatchCompute(groupCountX, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    if (gpuMilliseconds != nullptr)
    {
        *gpuMilliseconds = MeasureGpuMilliseconds(dispatchNBody);
    }
    else
    {
        dispatchNBody();
    }

    static size_t lastReportedObjectCount = std::numeric_limits<size_t>::max();
    if (objectCount != lastReportedObjectCount)
    {
        std::cout << "GPU N-body dispatch: objects=" << objectCount
                  << ", work groups=" << groupCountX << " x 1 x 1" << std::endl;
        lastReportedObjectCount = objectCount;
    }
}

#if defined(_DEBUG) || !defined(NDEBUG)
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

GpuOctreeMetadata ReadOctreeMetadataDebug()
{
    const GpuBuffer &metadataBuffer = gpuMemoryManager.Get(BufferRole::OctreeMetadata);
    GpuOctreeMetadata metadata{};
    glBindBuffer(GL_COPY_READ_BUFFER, metadataBuffer.Handle());
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(metadata), &metadata);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    return metadata;
}

void ValidateMassAggregationDebug(std::size_t objectCount)
{
    const GpuOctreeMetadata metadata = ReadOctreeMetadataDebug();
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
    double expectedMass = 0.0;
    double weightedX = 0.0;
    double weightedY = 0.0;
    double weightedZ = 0.0;
    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
    {
        if (controls[objectIndex].initializing != 0u)
        {
            continue;
        }
        const double mass = physical[objectIndex].massDensity.x;
        const glm::vec3 position = states[objectIndex].position;
        if (!std::isfinite(mass) || !std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z))
        {
            throw std::runtime_error("Mass validation found non-finite active-object input");
        }
        ++expectedActiveCount;
        expectedMass += mass;
        weightedX += static_cast<double>(position.x) * mass;
        weightedY += static_cast<double>(position.y) * mass;
        weightedZ += static_cast<double>(position.z) * mass;
    }

    if (metadata.counts.x != expectedActiveCount)
    {
        throw std::runtime_error("Mass aggregation active-object count differs from ObjectControl");
    }
    if (expectedActiveCount == 0)
    {
        if (status != GpuOctreeBuildStatus::Empty)
        {
            throw std::runtime_error("Mass aggregation did not preserve the empty-tree status");
        }
        return;
    }
    if (status != GpuOctreeBuildStatus::Valid || metadata.counts.y == 0u ||
        metadata.counts.w >= metadata.counts.y)
    {
        throw std::runtime_error("Mass aggregation did not produce a valid root aggregate");
    }

    const auto aggregates =
        ReadOctreeBuffer<GpuMassAggregate>(BufferRole::MassField, metadata.counts.y);
    for (const GpuMassAggregate &aggregate : aggregates)
    {
        if (!std::isfinite(aggregate.centerMass.x) ||
            !std::isfinite(aggregate.centerMass.y) ||
            !std::isfinite(aggregate.centerMass.z) ||
            !std::isfinite(aggregate.centerMass.w))
        {
            throw std::runtime_error("Mass aggregation produced a non-finite node aggregate");
        }
    }

    const glm::vec4 root = aggregates[metadata.counts.w].centerMass;
    if (!IsNearlyEqual(root.w, static_cast<float>(expectedMass)))
    {
        throw std::runtime_error("Octree root mass differs from the CPU reference sum");
    }
    if (expectedMass != 0.0)
    {
        const glm::vec3 expectedCenter(
            static_cast<float>(weightedX / expectedMass),
            static_cast<float>(weightedY / expectedMass),
            static_cast<float>(weightedZ / expectedMass));
        if (!IsNearlyEqual(root.x, expectedCenter.x) ||
            !IsNearlyEqual(root.y, expectedCenter.y) ||
            !IsNearlyEqual(root.z, expectedCenter.z))
        {
            throw std::runtime_error("Octree root center of mass differs from the CPU reference");
        }
    }
}

void ValidateBarnesHutDebug(std::size_t objectCount)
{
    const GpuOctreeMetadata metadata = ReadOctreeMetadataDebug();
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
} // namespace
#endif

void RunOctreeConstruction(GLuint boundingBoxShaderProgram, GLuint octreeBuildShaderProgram,
                           const ComputeUniformLocations &locations, size_t objectCount,
                           float *boundsGpuMilliseconds, float *constructionGpuMilliseconds)
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
    const auto dispatchBounds = []
    {
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    if (boundsGpuMilliseconds != nullptr)
    {
        *boundsGpuMilliseconds = MeasureGpuMilliseconds(dispatchBounds);
    }
    else
    {
        dispatchBounds();
    }

    glUseProgram(octreeBuildShaderProgram);
    glUniform1ui(locations.octreeBuildObjectCount, static_cast<GLuint>(objectCount));
    glUniform1ui(locations.octreeBuildObjectCapacity, static_cast<GLuint>(objectCapacity));
    const auto dispatchConstruction = []
    {
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    if (constructionGpuMilliseconds != nullptr)
    {
        *constructionGpuMilliseconds = MeasureGpuMilliseconds(dispatchConstruction);
    }
    else
    {
        dispatchConstruction();
    }

#if defined(_DEBUG) || !defined(NDEBUG)
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
#endif
}

void RunMassAggregation(GLuint computeShaderProgram, size_t objectCount,
                        float *gpuMilliseconds)
{
    glUseProgram(computeShaderProgram);
    const auto dispatchAggregation = []
    {
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    if (gpuMilliseconds != nullptr)
    {
        *gpuMilliseconds = MeasureGpuMilliseconds(dispatchAggregation);
    }
    else
    {
        dispatchAggregation();
    }

#if defined(_DEBUG) || !defined(NDEBUG)
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT);
    gpuMemoryManager.WaitForGpuCompletion();
    try
    {
        ValidateMassAggregationDebug(objectCount);
    }
    catch (const std::exception &error)
    {
        std::cerr << "Mass aggregation debug validation failed: " << error.what() << std::endl;
        throw;
    }
#else
    (void)objectCount;
#endif
}

void RunBarnesHutCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                         size_t objectCount, float epsilon, float theta,
                         float *gpuMilliseconds)
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
    const auto dispatchBarnesHut = [groupCountX]
    {
        glDispatchCompute(groupCountX, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    if (gpuMilliseconds != nullptr)
    {
        *gpuMilliseconds = MeasureGpuMilliseconds(dispatchBarnesHut);
    }
    else
    {
        dispatchBarnesHut();
    }

#if defined(_DEBUG) || !defined(NDEBUG)
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
#endif
}

void RunCollisionCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                         size_t objectCount, float *gpuMilliseconds)
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
    const auto dispatchCollision = [groupCountX]
    {
        glDispatchCompute(groupCountX, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    if (gpuMilliseconds != nullptr)
    {
        *gpuMilliseconds = MeasureGpuMilliseconds(dispatchCollision);
    }
    else
    {
        dispatchCollision();
    }

#if defined(_DEBUG) || !defined(NDEBUG)
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
#endif
}

void RunIntegration(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                    size_t objectCount, bool paused, double timeStep,
                    float *gpuMilliseconds)
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
    const auto dispatchIntegration = [groupCountX]
    {
        glDispatchCompute(groupCountX, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    if (gpuMilliseconds != nullptr)
    {
        *gpuMilliseconds = MeasureGpuMilliseconds(dispatchIntegration);
    }
    else
    {
        dispatchIntegration();
    }
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
                    size_t objectCount, float *gpuMilliseconds)
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
    const auto dispatchGrid = [groupCountX, groupCountY]
    {
        glDispatchCompute(groupCountX, groupCountY, 1);
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
    };
    if (gpuMilliseconds != nullptr)
    {
        *gpuMilliseconds = MeasureGpuMilliseconds(dispatchGrid);
    }
    else
    {
        dispatchGrid();
    }
}

void RunBarnesHutValidationSuite(const ShaderPrograms &programs,
                                 const ComputeUniformLocations &locations, size_t objectCount,
                                 float epsilon, double timeStep)
{
    if (objectCount == 0)
    {
        throw std::invalid_argument("Barnes-Hut benchmark requires at least one object");
    }

    auto *currentState =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    auto *nextState = gpuMemoryManager.Get(BufferRole::NextState).MappedPtr<GpuObjectState>();
    auto *physical =
        gpuMemoryManager.Get(BufferRole::ObjectPhysical).MappedPtr<GpuObjectPhysical>();
    auto *controls =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    auto *derived =
        gpuMemoryManager.Get(BufferRole::ObjectDerived).MappedPtr<GpuObjectDerived>();
    auto *accelerations =
        gpuMemoryManager.Get(BufferRole::ObjectAcceleration).MappedPtr<GpuObjectAcceleration>();
    auto *collisions =
        gpuMemoryManager.Get(BufferRole::ObjectCollision).MappedPtr<GpuObjectCollision>();
    if (currentState == nullptr || nextState == nullptr || physical == nullptr ||
        controls == nullptr || derived == nullptr || accelerations == nullptr ||
        collisions == nullptr)
    {
        throw std::runtime_error("GPU object mappings are unavailable for the benchmark");
    }

    std::vector<GpuObjectState> originalCurrent(currentState, currentState + objectCount);
    std::vector<GpuObjectState> originalNext(nextState, nextState + objectCount);
    std::vector<GpuObjectPhysical> originalPhysical(physical, physical + objectCount);
    std::vector<GpuObjectControl> originalControls(controls, controls + objectCount);
    std::vector<GpuObjectDerived> originalDerived(derived, derived + objectCount);
    std::vector<GpuObjectAcceleration> originalAccelerations(accelerations,
                                                              accelerations + objectCount);
    std::vector<GpuObjectCollision> originalCollisions(collisions, collisions + objectCount);

    const auto restoreOriginalData = [&]
    {
        gpuMemoryManager.WaitForGpuCompletion();
        std::copy(originalCurrent.begin(), originalCurrent.end(),
                  gpuMemoryManager.Get(BufferRole::CurrentState)
                      .MappedPtr<GpuObjectState>());
        std::copy(originalNext.begin(), originalNext.end(),
                  gpuMemoryManager.Get(BufferRole::NextState).MappedPtr<GpuObjectState>());
        std::copy(originalPhysical.begin(), originalPhysical.end(),
                  gpuMemoryManager.Get(BufferRole::ObjectPhysical)
                      .MappedPtr<GpuObjectPhysical>());
        std::copy(originalControls.begin(), originalControls.end(),
                  gpuMemoryManager.Get(BufferRole::ObjectControl)
                      .MappedPtr<GpuObjectControl>());
        std::copy(originalDerived.begin(), originalDerived.end(),
                  gpuMemoryManager.Get(BufferRole::ObjectDerived).MappedPtr<GpuObjectDerived>());
        std::copy(originalAccelerations.begin(), originalAccelerations.end(),
                  gpuMemoryManager.Get(BufferRole::ObjectAcceleration)
                      .MappedPtr<GpuObjectAcceleration>());
        std::copy(originalCollisions.begin(), originalCollisions.end(),
                  gpuMemoryManager.Get(BufferRole::ObjectCollision)
                      .MappedPtr<GpuObjectCollision>());
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
    };

    struct BenchmarkSnapshot
    {
        std::vector<GpuObjectState> states;
        std::vector<GpuObjectPhysical> physical;
        std::vector<GpuObjectControl> controls;
        std::vector<GpuObjectDerived> derived;
    };

    const auto uploadSnapshot = [&](const BenchmarkSnapshot &snapshot)
    {
        std::copy(snapshot.states.begin(), snapshot.states.end(),
                  gpuMemoryManager.Get(BufferRole::CurrentState)
                      .MappedPtr<GpuObjectState>());
        std::copy(snapshot.states.begin(), snapshot.states.end(),
                  gpuMemoryManager.Get(BufferRole::NextState).MappedPtr<GpuObjectState>());
        std::copy(snapshot.physical.begin(), snapshot.physical.end(),
                  gpuMemoryManager.Get(BufferRole::ObjectPhysical)
                      .MappedPtr<GpuObjectPhysical>());
        std::copy(snapshot.controls.begin(), snapshot.controls.end(),
                  gpuMemoryManager.Get(BufferRole::ObjectControl)
                      .MappedPtr<GpuObjectControl>());
        std::copy(snapshot.derived.begin(), snapshot.derived.end(),
                  gpuMemoryManager.Get(BufferRole::ObjectDerived)
                      .MappedPtr<GpuObjectDerived>());
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
    };

    const auto validateOctreeStatus = [&]
    {
        GpuOctreeMetadata metadata{};
        const GLuint metadataBuffer = gpuMemoryManager.Get(BufferRole::OctreeMetadata).Handle();
        glBindBuffer(GL_COPY_READ_BUFFER, metadataBuffer);
        glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(metadata), &metadata);
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        if (metadata.configuration.w !=
            static_cast<std::uint32_t>(GpuOctreeBuildStatus::Valid))
        {
            throw std::runtime_error("Benchmark pass reported invalid Octree status " +
                                     std::to_string(metadata.configuration.w));
        }
    };

    const auto makeSnapshot = [&](std::size_t count, const std::string &distribution,
                                  std::uint32_t seed)
    {
        BenchmarkSnapshot snapshot;
        snapshot.states.resize(count);
        snapshot.physical.resize(count);
        snapshot.controls.resize(count);
        snapshot.derived.resize(count);

        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> massDistribution(1.0e20f, 3.0e23f);
        std::uniform_real_distribution<float> coordinateDistribution(-10000.0f, 10000.0f);
        std::uniform_real_distribution<float> angleDistribution(
            0.0f, 2.0f * glm::pi<float>());
        std::uniform_real_distribution<float> radiusDistribution(1000.0f, 10000.0f);
        std::normal_distribution<float> clusterDistribution(0.0f, 300.0f);
        std::normal_distribution<float> diskThicknessDistribution(0.0f, 100.0f);

        for (std::size_t index = 0; index < count; ++index)
        {
            glm::vec3 position(0.0f);
            if (distribution == "uniform")
            {
                position = glm::vec3(coordinateDistribution(generator),
                                     coordinateDistribution(generator),
                                     coordinateDistribution(generator));
            }
            else if (distribution == "clustered")
            {
                const glm::vec3 center((index % 2 == 0) ? -5000.0f : 5000.0f,
                                       (index % 3 == 0) ? -2500.0f : 2500.0f,
                                       (index % 4 == 0) ? -4000.0f : 4000.0f);
                position = center + glm::vec3(clusterDistribution(generator),
                                              clusterDistribution(generator),
                                              clusterDistribution(generator));
            }
            else
            {
                const float angle = angleDistribution(generator);
                const float radius = radiusDistribution(generator);
                position = glm::vec3(radius * std::cos(angle),
                                     diskThicknessDistribution(generator),
                                     radius * std::sin(angle));
            }

            const float mass = massDistribution(generator);
            snapshot.states[index] = {glm::vec4(position, 0.0f), glm::vec4(0.0f)};
            snapshot.physical[index] = {glm::vec4(mass, 3344.0f, 0.0f, 0.0f)};
            snapshot.controls[index] =
                {0u, 1u, 0u, index < originalControls.size() ? originalControls[index].id : 0u};
            snapshot.derived[index] = {glm::vec4(0.0f)};
        }
        return snapshot;
    };

    const auto calculateErrors = [&](const std::vector<GpuObjectAcceleration> &reference,
                                     std::size_t count)
    {
        double squaredAbsoluteError = 0.0;
        double squaredRelativeError = 0.0;
        double maximumAbsoluteError = 0.0;
        double maximumRelativeError = 0.0;
        for (std::size_t index = 0; index < count; ++index)
        {
            const glm::vec3 expected = reference[index].acceleration;
            const glm::vec3 actual = accelerations[index].acceleration;
            const double absoluteError = glm::length(actual - expected);
            const double referenceMagnitude = glm::length(expected);
            const double relativeError =
                absoluteError / std::max(referenceMagnitude, 1.0e-12);
            squaredAbsoluteError += absoluteError * absoluteError;
            squaredRelativeError += relativeError * relativeError;
            maximumAbsoluteError = std::max(maximumAbsoluteError, absoluteError);
            maximumRelativeError = std::max(maximumRelativeError, relativeError);
        }
        const double countAsDouble = static_cast<double>(count);
        return std::array<double, 4>{
            std::sqrt(squaredAbsoluteError / countAsDouble),
            std::sqrt(squaredRelativeError / countAsDouble),
            maximumAbsoluteError,
            maximumRelativeError,
        };
    };

    struct TimingAverage
    {
        double bounds = 0.0;
        double construction = 0.0;
        double aggregation = 0.0;
        double direct = 0.0;
        double collision = 0.0;
        double integration = 0.0;
        double grid = 0.0;
    };

    constexpr std::array<float, 3> thetaValues = {0.3f, 0.5f, 0.8f};
    constexpr std::array<const char *, 3> distributions = {"uniform", "clustered", "disk"};
    std::vector<std::size_t> objectCounts = {
        std::min<std::size_t>(32, objectCount),
        std::min<std::size_t>(128, objectCount),
        objectCount,
    };
    std::sort(objectCounts.begin(), objectCounts.end());
    objectCounts.erase(std::unique(objectCounts.begin(), objectCounts.end()), objectCounts.end());

    std::filesystem::create_directories("numerical_error");
    std::ofstream report("numerical_error/barnes_hut_benchmark.csv", std::ios::trunc);
    if (!report)
    {
        throw std::runtime_error("Unable to create numerical_error/barnes_hut_benchmark.csv");
    }
    report << "renderer,version,distribution,object_count,theta,repeat_count,"
              "rms_absolute_acceleration_error,rms_relative_acceleration_error,"
              "max_absolute_acceleration_error,max_relative_acceleration_error,"
              "bounds_ms,construction_ms,mass_aggregation_ms,direct_nbody_ms,barnes_hut_ms,"
              "collision_ms,integration_ms,grid_ms,direct_total_ms,barnes_hut_total_ms,"
              "trajectory_steps,trajectory_time_seconds,rms_position_error,max_position_error\n";
    report << std::setprecision(10);

    const std::string renderer(reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    const std::string version(reinterpret_cast<const char *>(glGetString(GL_VERSION)));
    constexpr std::size_t timingRepeats = 3;
    constexpr std::size_t trajectorySteps = 24;
    const double trajectorySeconds = trajectorySteps * timeStep;
    bool trajectoryWritten = false;
    std::array<double, thetaValues.size()> trajectoryRmsPositionErrors{};
    std::array<double, thetaValues.size()> trajectoryMaximumPositionErrors{};

    try
    {
        for (std::size_t count : objectCounts)
        {
            for (std::size_t distributionIndex = 0;
                 distributionIndex < distributions.size(); ++distributionIndex)
            {
                const std::string distribution = distributions[distributionIndex];
                BenchmarkSnapshot snapshot =
                    makeSnapshot(count, distribution,
                                 static_cast<std::uint32_t>(101u + count * 17u +
                                                            distributionIndex * 131u));
                uploadSnapshot(snapshot);

                RunOctreeConstruction(programs.boundingBoxCompute, programs.octreeBuildCompute,
                                      locations, count);
                validateOctreeStatus();
                RunMassAggregation(programs.massAggregationCompute, count);
                validateOctreeStatus();
                RunNBodyCompute(programs.nbodyCompute, locations, count, epsilon);
                RunCollisionCompute(programs.collisionCompute, locations, count);
                RunIntegration(programs.integrateCompute, locations, count, false, timeStep);
                RunGridCompute(programs.gridCompute, locations, count);
                for (float theta : thetaValues)
                {
                    RunBarnesHutCompute(programs.barnesHutCompute, locations, count, epsilon,
                                        theta);
                }

                TimingAverage timing;
                std::array<double, thetaValues.size()> barnesHutTimings{};
                std::array<std::array<double, 4>, thetaValues.size()> errors{};
                for (std::size_t repeat = 0; repeat < timingRepeats; ++repeat)
                {
                    float boundsMs = 0.0f;
                    float constructionMs = 0.0f;
                    float aggregationMs = 0.0f;
                    float directMs = 0.0f;
                    float collisionMs = 0.0f;
                    float integrationMs = 0.0f;
                    float gridMs = 0.0f;
                    RunOctreeConstruction(programs.boundingBoxCompute, programs.octreeBuildCompute,
                                          locations, count, &boundsMs, &constructionMs);
                    RunMassAggregation(programs.massAggregationCompute, count, &aggregationMs);
                    RunNBodyCompute(programs.nbodyCompute, locations, count, epsilon, &directMs);
                    std::vector<GpuObjectAcceleration> directAcceleration(
                        accelerations, accelerations + count);
                    RunCollisionCompute(programs.collisionCompute, locations, count, &collisionMs);
                    RunIntegration(programs.integrateCompute, locations, count, false, timeStep,
                                   &integrationMs);
                    RunGridCompute(programs.gridCompute, locations, count, &gridMs);

                    timing.bounds += boundsMs;
                    timing.construction += constructionMs;
                    timing.aggregation += aggregationMs;
                    timing.direct += directMs;
                    timing.collision += collisionMs;
                    timing.integration += integrationMs;
                    timing.grid += gridMs;

                    for (std::size_t thetaIndex = 0; thetaIndex < thetaValues.size(); ++thetaIndex)
                    {
                        float barnesHutMs = 0.0f;
                        RunBarnesHutCompute(programs.barnesHutCompute, locations, count, epsilon,
                                             thetaValues[thetaIndex], &barnesHutMs);
                        validateOctreeStatus();
                        barnesHutTimings[thetaIndex] += barnesHutMs;
                        errors[thetaIndex] =
                            calculateErrors(directAcceleration, count);
                        for (double error : errors[thetaIndex])
                        {
                            if (!std::isfinite(error))
                            {
                                throw std::runtime_error(
                                    "Benchmark found a non-finite Barnes-Hut force error");
                            }
                        }
                    }
                }

                if (!trajectoryWritten && count >= std::min<std::size_t>(32, objectCount) &&
                    distribution == "clustered")
                {
                    const std::size_t trajectoryCount = std::min<std::size_t>(32, count);
                    const BenchmarkSnapshot trajectorySnapshot =
                        makeSnapshot(trajectoryCount, distribution, 918273u);
                    uploadSnapshot(trajectorySnapshot);
                    for (std::size_t step = 0; step < trajectorySteps; ++step)
                    {
                        RunGpuStateTransition(programs.nbodyCompute, programs.integrateCompute,
                                              locations, trajectoryCount, epsilon, false, timeStep);
                    }
                    gpuMemoryManager.WaitForGpuCompletion();
                    const auto *directFinal =
                        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
                    std::vector<glm::vec3> directPositions(trajectoryCount);
                    for (std::size_t index = 0; index < trajectoryCount; ++index)
                    {
                        directPositions[index] = directFinal[index].position;
                    }

                    for (std::size_t thetaIndex = 0; thetaIndex < thetaValues.size(); ++thetaIndex)
                    {
                        uploadSnapshot(trajectorySnapshot);
                        for (std::size_t step = 0; step < trajectorySteps; ++step)
                        {
                            RunOctreeConstruction(programs.boundingBoxCompute,
                                                  programs.octreeBuildCompute, locations,
                                                  trajectoryCount);
                            validateOctreeStatus();
                            RunMassAggregation(programs.massAggregationCompute, trajectoryCount);
                            validateOctreeStatus();
                            RunBarnesHutStateTransition(
                                programs.barnesHutCompute, programs.collisionCompute,
                                programs.integrateCompute, locations, trajectoryCount, epsilon,
                                thetaValues[thetaIndex], false, timeStep);
                            validateOctreeStatus();
                        }
                        gpuMemoryManager.WaitForGpuCompletion();
                        const auto *barnesHutFinal = gpuMemoryManager.Get(BufferRole::CurrentState)
                                                         .MappedPtr<GpuObjectState>();
                        double squaredPositionError = 0.0;
                        double maximumPositionError = 0.0;
                        for (std::size_t index = 0; index < trajectoryCount; ++index)
                        {
                            const double positionError =
                                glm::length(glm::vec3(barnesHutFinal[index].position) -
                                            directPositions[index]);
                            squaredPositionError += positionError * positionError;
                            maximumPositionError =
                                std::max(maximumPositionError, positionError);
                        }
                        trajectoryRmsPositionErrors[thetaIndex] =
                            std::sqrt(squaredPositionError / trajectoryCount);
                        trajectoryMaximumPositionErrors[thetaIndex] = maximumPositionError;
                    }
                    trajectoryWritten = true;
                }

                const double repeatCount = static_cast<double>(timingRepeats);
                const double commonMs =
                    (timing.integration + timing.grid) / repeatCount;
                const double collisionMs = timing.collision / repeatCount;
                const double directTotalMs = timing.direct / repeatCount + commonMs;
                const double barnesHutPassMs =
                    (timing.bounds + timing.construction + timing.aggregation) / repeatCount;
                for (std::size_t thetaIndex = 0; thetaIndex < thetaValues.size(); ++thetaIndex)
                {
                    const double barnesHutForceMs =
                        barnesHutTimings[thetaIndex] / repeatCount;
                    const double barnesHutTotalMs =
                        barnesHutPassMs + barnesHutForceMs + collisionMs + commonMs;
                    const bool hasTrajectory =
                        trajectoryWritten && count == std::min<std::size_t>(32, objectCount) &&
                        distribution == "clustered";
                    report << '"' << renderer << "\",\"" << version << "\","
                           << distribution << ',' << count << ',' << thetaValues[thetaIndex] << ','
                           << timingRepeats << ',' << errors[thetaIndex][0] << ','
                           << errors[thetaIndex][1] << ',' << errors[thetaIndex][2] << ','
                           << errors[thetaIndex][3] << ','
                           << timing.bounds / repeatCount << ','
                           << timing.construction / repeatCount << ','
                           << timing.aggregation / repeatCount << ','
                           << timing.direct / repeatCount << ','
                           << barnesHutForceMs << ','
                           << timing.collision / repeatCount << ','
                           << timing.integration / repeatCount << ',' << timing.grid / repeatCount
                           << ',' << directTotalMs << ',' << barnesHutTotalMs << ','
                           << (hasTrajectory ? trajectorySteps : 0) << ','
                           << (hasTrajectory ? trajectorySeconds : 0.0) << ','
                           << (hasTrajectory ? trajectoryRmsPositionErrors[thetaIndex] : 0.0)
                           << ','
                           << (hasTrajectory
                                   ? trajectoryMaximumPositionErrors[thetaIndex]
                                   : 0.0)
                           << '\n';
                }
            }
        }
        report.flush();
        if (!report.good())
        {
            throw std::runtime_error("Failed while writing Barnes-Hut benchmark report");
        }
    }
    catch (...)
    {
        restoreOriginalData();
        throw;
    }
    restoreOriginalData();
    std::cout << "Wrote same-snapshot Barnes-Hut validation and GPU timings to "
                 "numerical_error/barnes_hut_benchmark.csv"
              << std::endl;
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