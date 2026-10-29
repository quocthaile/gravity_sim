#include "gravity_sim_3Dgrid_function.hpp"

#include "render/asset_paths.hpp"
#include "render/shared_grid.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

// Simulation and compute configuration.
extern const double kGravitationalConstant = 6.6743e-11;
const glm::vec3 kInitialCameraPosition(0.0f, 1000.0f, 5000.0f);
extern const float kSpeedOfLight = 299792458.0f; // đổi từ float qua double
float initMass = 5.0f * std::pow(10.0f, 20.0f) / 5.0f; // đổi từ float qua double
int numRandomObjects = 200;
float gridSize = 30000.0f;
int gridDivisions = 200;

// Must match layout(local_size_x = 16, local_size_y = 16) in grid.comp.
extern const GLuint kGridLocalSizeX = 16;
extern const GLuint kGridLocalSizeY = 16;

// Mutable simulation and camera state.
bool running = true;
bool pause = false;
glm::vec3 cameraPos = kInitialCameraPosition;
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 0.0f, 0.0f);
float lastX = 400.0f;
float lastY = 300.0f;
float yaw = -90.0f;
float pitch = 0.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// OpenGL resources and simulation objects shared by the render loop and callbacks.
std::vector<Object> objs = {};
GLuint gridVAO = 0;
GLuint gridEBO = 0;
GpuMemoryManager gpuMemoryManager;
size_t gridNodeCount = 0;
size_t gridIndexCount = 0;
size_t objectStateCapacity = 0;

// Mass aggregation
float CEll_SIZE = gridSize / static_cast<float>(gridDivisions);
float worldMin = -gridSize / 2.0f;
float worldMax = gridSize / 2.0f;
GLuint objectCellID = 0;

// numerical error
double dt = 1.0 / 96.0;

void SavePositions(const char *filename, const std::vector<std::vector<glm::vec3>> &positions,
                   double stepDt)
{
    std::filesystem::path outPath(filename);
    if (outPath.has_parent_path())
    {
        std::error_code ec;
        std::filesystem::create_directories(outPath.parent_path(), ec);
    }

    std::ofstream file(outPath);

    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << filename << '\n';
        return;
    }

    file << "time,step,object_id,x,y,z\n";
    file << std::setprecision(10);

    for (size_t step = 0; step < positions.size(); ++step)
    {
        double simulationTime = static_cast<double>(step) * stepDt;

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

void CaptureGpuTrajectorySampleV2(std::vector<TrajectorySampleV2> &samples,
                                  std::uint64_t sceneGeneration, std::uint64_t simulationStep,
                                  double nominalTimeSeconds)
{
    if (!gpuMemoryManager.Contains(BufferRole::CurrentState) ||
        !gpuMemoryManager.Contains(BufferRole::ObjectControl))
    {
        throw std::runtime_error(
            "CurrentState/ObjectControl buffers are unavailable for trajectory v2 capture");
    }

    gpuMemoryManager.WaitForGpuCompletion();
    const auto *gpuState =
        gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
    const auto *gpuControl =
        gpuMemoryManager.Get(BufferRole::ObjectControl).MappedPtr<GpuObjectControl>();
    if (gpuState == nullptr || gpuControl == nullptr)
    {
        throw std::runtime_error("Persistent mappings are unavailable for trajectory v2 capture");
    }

    const size_t gpuObjectCount = gpuMemoryManager.Get(BufferRole::CurrentState).LogicalCount();
    TrajectorySampleV2 sample;
    sample.sceneGeneration = sceneGeneration;
    sample.simulationStep = simulationStep;
    sample.nominalTimeSeconds = nominalTimeSeconds;
    sample.entries.reserve(gpuObjectCount);

    for (size_t index = 0; index < gpuObjectCount; ++index)
    {
        TrajectoryRecordEntryV2 rec;
        rec.logicalObjectId = gpuControl[index].id;
        rec.position = glm::vec3(gpuState[index].position);
        sample.entries.push_back(rec);
    }
    samples.push_back(std::move(sample));
}

bool SavePositionsV2(const std::string &filename, const std::vector<TrajectorySampleV2> &samples)
{
    std::filesystem::path outPath(filename);
    if (outPath.has_parent_path())
    {
        std::error_code ec;
        std::filesystem::create_directories(outPath.parent_path(), ec);
    }

    std::ofstream file(outPath);
    if (!file.is_open())
    {
        return false;
    }

    file << "# schema_version=2\n";
    file << "scene_generation,step,nominal_time_s,logical_object_id,x,y,z\n";
    file << std::setprecision(10);

    for (const auto &sample : samples)
    {
        for (const auto &entry : sample.entries)
        {
            file << sample.sceneGeneration << ',' << sample.simulationStep << ','
                 << sample.nominalTimeSeconds << ',' << entry.logicalObjectId << ','
                 << entry.position.x << ',' << entry.position.y << ',' << entry.position.z << '\n';
        }
    }
    return file.good();
}

struct ObjectMesh
{
    GLuint objectVAO = 0;
    GLuint objectVBO = 0;
    GLsizei vertexCount = 0;
};

ObjectMesh objectMesh;

GLuint GetUnitSphereVao() { return objectMesh.objectVAO; }
GLuint GetUnitSphereVbo() { return objectMesh.objectVBO; }
GLsizei GetUnitSphereVertexCount() { return objectMesh.vertexCount; }

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
    const std::filesystem::path resolved = grav_render::ResolveAssetPath(filePath);
    std::ifstream shaderFile(resolved, std::ios::in | std::ios::binary);
    if (!shaderFile)
    {
        throw std::runtime_error("Unable to open shader file: " + resolved.string());
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
    shaderPrograms.gridSourceCompute = CreateComputeProgram("shaders/grid_sources.comp");
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

    locations.nbodyObjectCount =
        getRequiredLocation(programs.nbodyCompute, "nbodyCompute", "u_objectCount");
    locations.nbodyGravitationalConstant = getRequiredLocation(
        programs.nbodyCompute, "nbodyCompute", "u_gravitationalConstant");
    locations.nbodyEpsilon = getRequiredLocation(programs.nbodyCompute, "nbodyCompute", "u_epsilon");

    locations.integrateObjectCount =
        getRequiredLocation(programs.integrateCompute, "integrateCompute", "u_objectCount");
    locations.integratePaused =
        getRequiredLocation(programs.integrateCompute, "integrateCompute", "u_paused");

    if (programs.gridSourceCompute != 0)
    {
        locations.gridSourceObjectCount =
            getRequiredLocation(programs.gridSourceCompute, "gridSourceCompute", "u_objectCount");
    }

    if (programs.gridCompute != 0)
    {
        locations.gridSourceCount =
            getRequiredLocation(programs.gridCompute, "gridCompute", "u_sourceCount");
        locations.gridWidth =
            getRequiredLocation(programs.gridCompute, "gridCompute", "u_gridWidth");
        locations.gridHeight =
            getRequiredLocation(programs.gridCompute, "gridCompute", "u_gridHeight");
        locations.gridMetersPerWorldUnit =
            getRequiredLocation(programs.gridCompute, "gridCompute", "u_metersPerWorldUnit");
        locations.gridDeformationAmplitudeWorld = getRequiredLocation(
            programs.gridCompute, "gridCompute", "u_deformationAmplitudeWorld");
        locations.gridClampDelta =
            getRequiredLocation(programs.gridCompute, "gridCompute", "u_clampDelta");
    }
}

glm::mat4 CreateScenePerspectiveProjection(float aspect)
{
    const float safeAspect = aspect > 0.01f ? aspect : (1920.0f / 1080.0f);
    return glm::perspective(glm::radians(45.0f), safeAspect, 0.1f, 750000.0f);
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

GLFWwindow *StartGLU(int width, int height, const char *title, bool visible)
{
    // Initialize GLFW and create a window with an OpenGL context.
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return nullptr;
    }
    // Request OpenGL 4.4 core for immutable buffer storage and persistent mapping.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    // Create a window with the specified dimensions and title.
    GLFWwindow *window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window (OpenGL 4.4 Core required)." << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    // Initialize GLEW to load OpenGL function pointers.
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        glfwDestroyWindow(window);
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

    GLint maxStorageBufferBindings = 0;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxStorageBufferBindings);
    if (maxStorageBufferBindings <= static_cast<GLint>(BindingFor(BufferRole::ObjectCollision)))
    {
        std::cerr << "OpenGL implementation does not provide at least 16 SSBO binding points."
                  << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    // Print compute limits for debugging purposes.
    PrintComputeLimits();
    // Enable depth testing and set the viewport dimensions.
    int fbWidth = width;
    int fbHeight = height;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, std::max(fbWidth, 1), std::max(fbHeight, 1));
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

    GLint success = GL_FALSE;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        GLint infoLogLength = 0;
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::string infoLog(static_cast<size_t>(std::max(infoLogLength, 1)), '\0');
        glGetShaderInfoLog(vertexShader, infoLogLength, nullptr, infoLog.data());
        glDeleteShader(vertexShader);
        throw std::runtime_error("Vertex shader compilation failed (" + vertexFilePath +
                                 "): " + infoLog);
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSourceData, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        GLint infoLogLength = 0;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::string infoLog(static_cast<size_t>(std::max(infoLogLength, 1)), '\0');
        glGetShaderInfoLog(fragmentShader, infoLogLength, nullptr, infoLog.data());
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        throw std::runtime_error("Fragment shader compilation failed (" + fragmentFilePath +
                                 "): " + infoLog);
    }

    GLuint renderingShaderProgram = glCreateProgram();
    glAttachShader(renderingShaderProgram, vertexShader);
    glAttachShader(renderingShaderProgram, fragmentShader);
    glLinkProgram(renderingShaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGetProgramiv(renderingShaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        GLint infoLogLength = 0;
        glGetProgramiv(renderingShaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::string infoLog(static_cast<size_t>(std::max(infoLogLength, 1)), '\0');
        glGetProgramInfoLog(renderingShaderProgram, infoLogLength, nullptr, infoLog.data());
        glDeleteProgram(renderingShaderProgram);
        throw std::runtime_error("Shader program linking failed (" + vertexFilePath + " + " +
                                 fragmentFilePath + "): " + infoLog);
    }

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
        throw std::runtime_error("Compute shader compilation failed (" + filePath +
                                 "): " + infoLog);
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
        throw std::runtime_error("Compute program linking failed (" + filePath + "): " + infoLog);
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
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertexCount * sizeof(float)), vertices,
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    if (EBO != nullptr)
    {
        glGenBuffers(1, EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(indexCount * sizeof(unsigned int)), indices,
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

    gpuMemoryManager.WaitForGpuCompletion();
    gpuMemoryManager.Resize(BufferRole::CurrentState, newCapacity);
    gpuMemoryManager.Resize(BufferRole::NextState, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectPhysical, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectDerived, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectControl, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectAcceleration, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectRender, newCapacity);
    gpuMemoryManager.Resize(BufferRole::ObjectCollision, newCapacity);

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
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(gridIndices.size() * sizeof(unsigned int)),
                 gridIndices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void InitializeObjectComputeBuffers(size_t initialObjectCount)
{
    GLint maxStorageBufferBindings = 0;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxStorageBufferBindings);
    if (maxStorageBufferBindings <= static_cast<GLint>(BindingFor(BufferRole::ObjectCollision)))
    {
        throw std::runtime_error("OpenGL does not provide SSBO binding 15 for ObjectCollision");
    }

    objectStateCapacity = 0;
    objectStateCapacity = CalculateObjectStateBufferCapacity(initialObjectCount);
    if (objectStateCapacity == 0)
    {
        objectStateCapacity = 256;
    }

    gpuMemoryManager.Create({BufferRole::CurrentState, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::ReadWrite,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectState),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::NextState, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::ReadWrite,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectState),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectPhysical, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::Write,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectPhysical),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectDerived, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::Write,
                             MappingMode::PersistentCoherent, sizeof(GpuObjectDerived),
                             objectStateCapacity});
    gpuMemoryManager.Create({BufferRole::ObjectControl, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::Write,
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
}

void BindAllObjectComputeBuffers()
{
    for (BufferRole role : kGravityObjectBufferRoles)
    {
        if (gpuMemoryManager.Contains(role))
        {
            gpuMemoryManager.Bind(role);
        }
    }
}

void RetireInitializingObjects()
{
    while (!objs.empty() && objs.back().initializing)
    {
        objs.pop_back();
    }
}

void ReleaseObjectComputeBuffers(bool alreadyQuiescent)
{
    RetireInitializingObjects();
    gpuMemoryManager.DestroyRoles(kGravityObjectBufferRoles.data(),
                                  kGravityObjectBufferRoles.size(), alreadyQuiescent);
    objectStateCapacity = 0;
    std::vector<Object>().swap(objs);
}

void ComputePipeline(const std::vector<glm::vec4> &gridVertices)
{
    InitializeObjectComputeBuffers(objs.size());

    if (!gpuMemoryManager.Contains(BufferRole::BaseGrid))
    {
        gpuMemoryManager.Create({BufferRole::BaseGrid, GL_SHADER_STORAGE_BUFFER,
                                 GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                                 sizeof(glm::vec4), gridVertices.size()});
    }
    else
    {
        gpuMemoryManager.Resize(BufferRole::BaseGrid, gridVertices.size());
    }
    gpuMemoryManager.Upload(BufferRole::BaseGrid, gridVertices.data(),
                            gridVertices.size() * sizeof(glm::vec4), gridVertices.size());

    if (!gpuMemoryManager.Contains(BufferRole::DeformedGrid))
    {
        gpuMemoryManager.Create({BufferRole::DeformedGrid, GL_SHADER_STORAGE_BUFFER,
                                 GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                                 sizeof(glm::vec4), gridVertices.size()});
    }
    else
    {
        gpuMemoryManager.Resize(BufferRole::DeformedGrid, gridVertices.size());
    }

    if (!gpuMemoryManager.Contains(BufferRole::GridSources))
    {
        gpuMemoryManager.Create({BufferRole::GridSources, GL_SHADER_STORAGE_BUFFER,
                                 GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                                 sizeof(grav_render::GridSource),
                                 std::max<std::size_t>(objectStateCapacity, 16)},
                                false);
    }
    BindAllObjectComputeBuffers();
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

void RunIntegration(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                    size_t objectCount, bool paused)
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
    glDispatchCompute(groupCountX, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void RunObjectStateSync(GLFWwindow *window, size_t &objectCount)
{
    // 1. Update CPU-side properties of the object being created (mass/rs/radius).
    if (window != nullptr)
    {
        UpdateInitializingObject(window);
    }

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
                           float epsilon, bool paused)
{
    if (objectCount == 0)
    {
        return;
    }

    // Rebind all 8 object SSBO roles explicitly because pass-scoped binding 1 is shared with
    // GridSources during grid materialization and grid deformation passes.
    BindAllObjectComputeBuffers();

    // Ensure that any CPU writes to persistently mapped buffers are visible to the GPU before running
    glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
    // Run the N-body compute shader to calculate accelerations based on current positions and masses.
    RunNBodyCompute(nbodyShaderProgram, locations, objectCount, epsilon);
    // Run the integration compute shader
    RunIntegration(integrateShaderProgram, locations, objectCount, paused);
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

void RunGridSourceMaterialization(GLuint gridSourceShaderProgram,
                                  const ComputeUniformLocations &locations, size_t objectCount)
{
    if (!gpuMemoryManager.Contains(BufferRole::GridSources))
    {
        gpuMemoryManager.Create({BufferRole::GridSources, GL_SHADER_STORAGE_BUFFER,
                                 GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                                 sizeof(grav_render::GridSource),
                                 std::max<std::size_t>(objectCount, 16)},
                                false);
    }
    else if (gpuMemoryManager.Get(BufferRole::GridSources).Capacity() < objectCount)
    {
        gpuMemoryManager.Resize(BufferRole::GridSources,
                                CalculateObjectStateBufferCapacity(objectCount), false);
    }
    gpuMemoryManager.Get(BufferRole::GridSources).SetLogicalCount(objectCount);

    if (objectCount == 0)
    {
        return;
    }

    gpuMemoryManager.Bind(BufferRole::CurrentState, 0);
    gpuMemoryManager.Bind(BufferRole::ObjectDerived, 12);
    gpuMemoryManager.Bind(BufferRole::GridSources, 1);

    constexpr GLuint localSizeX = 64;
    const GLuint groupCountX = static_cast<GLuint>((objectCount + localSizeX - 1) / localSizeX);
    glUseProgram(gridSourceShaderProgram);
    glUniform1ui(locations.gridSourceObjectCount, static_cast<GLuint>(objectCount));
    glDispatchCompute(groupCountX, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void RunGridCompute(GLuint computeShaderProgram, const ComputeUniformLocations &locations,
                    size_t sourceCount, float metersPerWorldUnit, float deformationAmplitudeWorld,
                    float clampDelta)
{
    if (gpuMemoryManager.Contains(BufferRole::GridSources))
    {
        gpuMemoryManager.Bind(BufferRole::GridSources, 1);
    }
    if (gpuMemoryManager.Contains(BufferRole::BaseGrid))
    {
        gpuMemoryManager.Bind(BufferRole::BaseGrid, 4);
    }
    if (gpuMemoryManager.Contains(BufferRole::DeformedGrid))
    {
        gpuMemoryManager.Bind(BufferRole::DeformedGrid, 5);
    }

    glUseProgram(computeShaderProgram);
    glUniform1ui(locations.gridSourceCount, static_cast<GLuint>(sourceCount));
    const GLuint gridWidth = static_cast<GLuint>(gridDivisions + 1);
    const GLuint gridHeight = static_cast<GLuint>(gridDivisions + 1);
    glUniform1ui(locations.gridWidth, gridWidth);
    glUniform1ui(locations.gridHeight, gridHeight);
    glUniform1f(locations.gridMetersPerWorldUnit, metersPerWorldUnit);
    glUniform1f(locations.gridDeformationAmplitudeWorld, deformationAmplitudeWorld);
    glUniform1f(locations.gridClampDelta, clampDelta);
    // Round up work group counts
    const GLuint groupCountX = (gridWidth + kGridLocalSizeX - 1) / kGridLocalSizeX;
    const GLuint groupCountY = (gridHeight + kGridLocalSizeY - 1) / kGridLocalSizeY;
    // Dispatch the compute shader with the calculated number of work groups.
    glDispatchCompute(groupCountX, groupCountY, 1);

    // grid.comp writes DeformedGrid as an SSBO; the grid VAO fetches the same buffer as vertex
    // attributes. This barrier orders that compute write -> vertex fetch dependency on the GPU.
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}

void Cleanup(const ShaderPrograms &shaderPrograms)
{
    DestroyObjectRenderingResources();
    if (gridVAO != 0)
    {
        glDeleteVertexArrays(1, &gridVAO);
    }
    if (gridEBO != 0)
    {
        glDeleteBuffers(1, &gridEBO);
    }
    gpuMemoryManager.Shutdown();
    if (shaderPrograms.gridRendering != 0)
        glDeleteProgram(shaderPrograms.gridRendering);
    if (shaderPrograms.gridCompute != 0)
        glDeleteProgram(shaderPrograms.gridCompute);
    if (shaderPrograms.gridSourceCompute != 0)
        glDeleteProgram(shaderPrograms.gridSourceCompute);
    if (shaderPrograms.objectRendering != 0)
        glDeleteProgram(shaderPrograms.objectRendering);
    if (shaderPrograms.nbodyCompute != 0)
        glDeleteProgram(shaderPrograms.nbodyCompute);
    if (shaderPrograms.integrateCompute != 0)
        glDeleteProgram(shaderPrograms.integrateCompute);

    std::vector<Object>().swap(objs);
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
    if (window == nullptr || objs.empty() || !objs.back().initializing)
        return;

    Object &initializingObject = objs.back();
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        initializingObject.mass *= 1.0f + 1.0f * deltaTime;
        initializingObject.rs = static_cast<float>(
            (2.0 * kGravitationalConstant * initializingObject.mass) /
            (static_cast<double>(kSpeedOfLight) * kSpeedOfLight));
    }

    initializingObject.radius =
        std::pow((3.0f * initializingObject.mass / initializingObject.density) /
                     (4.0f * 3.14159265359f),
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

    if (key == GLFW_KEY_K)
    {
        if (action == GLFW_PRESS)
            pause = true;
        else if (action == GLFW_RELEASE)
            pause = false;
    }

    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
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
                lastObj.position[1] += 0.5f;
            }
            else
            {
                lastObj.position[2] += 0.5f;
            }
            positionChanged = true;
        }
        if (key == GLFW_KEY_DOWN && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            if (!shiftPressed)
            {
                lastObj.position[1] -= 0.5f;
            }
            else
            {
                lastObj.position[2] -= 0.5f;
            }
            positionChanged = true;
        }
        if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            lastObj.position[0] += 0.5f;
            positionChanged = true;
        }
        if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            lastObj.position[0] -= 0.5f;
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
    float xOffset = static_cast<float>(xPosition) - lastX;
    float yOffset = lastY - static_cast<float>(yPosition);
    lastX = static_cast<float>(xPosition);
    lastY = static_cast<float>(yPosition);
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
    front.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
    front.y = std::sin(glm::radians(pitch));
    front.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
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
            objs.emplace_back(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), initMass);
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
    float x = radius * std::sin(theta) * std::cos(phi);
    float y = radius * std::cos(theta);
    float z = radius * std::sin(theta) * std::sin(phi);
    return glm::vec3(x, y, z);
}

void DrawGrid(GLuint renderingShaderProgram, GLuint gridVAOHandle, size_t indexCount,
              const ShaderUniformLocations &uniformLocations)
{
    glUseProgram(renderingShaderProgram);
    glUniform4f(uniformLocations.gridObjectColor, 1.0f, 1.0f, 1.0f, 0.25f);
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(uniformLocations.gridModel, 1, GL_FALSE, glm::value_ptr(model));
    glBindVertexArray(gridVAOHandle);
    // glPointSize(5.0f);
    glDrawElements(GL_LINES, static_cast<GLsizei>(indexCount), GL_UNSIGNED_INT, nullptr);
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
    vertices.reserve(static_cast<size_t>(numNodesPerSide) * numNodesPerSide * 3);

    float step = size / static_cast<float>(divisions);
    float halfSize = size / 2.0f;
    float baseY = -800.0f;

    for (int zStep = 0; zStep <= divisions; ++zStep)
    {
        float z = -halfSize + static_cast<float>(zStep) * step;
        for (int xStep = 0; xStep <= divisions; ++xStep)
        {
            float x = -halfSize + static_cast<float>(xStep) * step;
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
    size_t totalIndices =
        static_cast<size_t>(4) * static_cast<size_t>(divisions) * static_cast<size_t>(divisions + 1);
    indices.reserve(totalIndices);

    for (int z = 0; z <= divisions; ++z)
    {
        for (int x = 0; x < divisions; ++x)
        {
            unsigned int current = static_cast<unsigned int>(z * rowSize + x);
            indices.push_back(current);
            indices.push_back(current + 1);
        }
    }

    for (int z = 0; z < divisions; ++z)
    {
        for (int x = 0; x <= divisions; ++x)
        {
            unsigned int current = static_cast<unsigned int>(z * rowSize + x);
            indices.push_back(current);
            indices.push_back(current + static_cast<unsigned int>(rowSize));
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
    generated.reserve(static_cast<size_t>(std::max(count, 0)));

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
        std::filesystem::path resolvedPath;
        if (!grav_render::TryResolveAssetPath(objectFilePath, resolvedPath))
        {
            resolvedPath = std::filesystem::path(objectFilePath);
        }
        std::ifstream objectFile(resolvedPath);
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
            // Trim trailing CR from Windows CRLF files
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            {
                line.pop_back();
            }
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
            if (!std::isfinite(positionX) || !std::isfinite(positionY) ||
                !std::isfinite(positionZ) || !std::isfinite(velocityX) ||
                !std::isfinite(velocityY) || !std::isfinite(velocityZ) ||
                !std::isfinite(mass) || mass <= 0.0f || !std::isfinite(density) ||
                density <= 0.0f)
            {
                throw std::runtime_error("Non-finite or non-positive mass/density at line " +
                                         std::to_string(lineNumber) + " in " + objectFilePath);
            }

            loadedObjects.emplace_back(glm::vec3(positionX, positionY, positionZ),
                                       glm::vec3(velocityX, velocityY, velocityZ), mass, density);
        }
        if (loadedObjects.empty())
        {
            throw std::runtime_error("Object file contained zero valid objects: " + objectFilePath);
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

std::vector<Object> CreateLegacyThreeBodyObjects()
{
    // Provenance: Initial conditions imported from src/legacy/gravity_sim.cpp:227-232.
    // Note: Evaluated under the active engine's solver (GPU snapshot N-body or F6 CPU baseline),
    // not claiming identical trajectories to the legacy GLOW prototype.
    std::vector<Object> bodies;
    bodies.reserve(3);

    bodies.emplace_back(glm::vec3(-5000.0f, 650.0f, -350.0f), glm::vec3(0.0f, 0.0f, 1500.0f),
                        5.97219e22f, 5515.0f);
    bodies.back().color = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);

    bodies.emplace_back(glm::vec3(5000.0f, 650.0f, -350.0f), glm::vec3(0.0f, 0.0f, -1500.0f),
                        5.97219e22f, 5515.0f);
    bodies.back().color = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);

    bodies.emplace_back(glm::vec3(0.0f, 0.0f, -350.0f), glm::vec3(0.0f, 0.0f, 0.0f), 1.989e25f,
                        5515.0f);
    bodies.back().color = glm::vec4(1.0f, 0.929f, 0.176f, 1.0f);
    return bodies;
}
