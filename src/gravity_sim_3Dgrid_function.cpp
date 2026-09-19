#include "gravity_sim_3Dgrid_function.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

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

void InitializeRenderingResources(GLuint &shaderProgram, GLint &modelLocation,
                                  GLint &objectColorLocation, GLint &viewLocation,
                                  glm::vec3 &cameraPosition)
{
    std::string vertexShaderSource = LoadShaderSource("shaders/grid.vert");
    std::string fragmentShaderSource = LoadShaderSource("shaders/grid.frag");
    std::string computeShaderSource = LoadShaderSource("shaders/grid.comp");
    shaderProgram = CreateShaderProgram(vertexShaderSource.c_str(), fragmentShaderSource.c_str());
    gridComputeProgram = CreateComputeProgram(computeShaderSource.c_str());

    modelLocation = glGetUniformLocation(shaderProgram, "model");
    objectColorLocation = glGetUniformLocation(shaderProgram, "objectColor");
    viewLocation = glGetUniformLocation(shaderProgram, "view");
    glUseProgram(shaderProgram);

    glm::mat4 projection =
        glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 750000.0f);
    GLint projectionLocation = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, glm::value_ptr(projection));
    cameraPosition = glm::vec3(0.0f, 1000.0f, 5000.0f);
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
    // Set the OpenGL version to 4.3 and use the core profile.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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
    // Print the OpenGL version and compute limits for debugging purposes.
    PrintComputeLimits();
    // Enable depth testing and set the viewport dimensions.
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, 1920, 1080);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return window;
}

GLuint CreateShaderProgram(const char *vertexSource, const char *fragmentSource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, nullptr);
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
    glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
    }

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

GLuint CreateComputeProgram(const char *computeSource)
{
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeSource, nullptr);
    glCompileShader(computeShader);

    GLint success = GL_FALSE;
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(computeShader, 512, nullptr, infoLog);
        std::cerr << "Compute shader compilation failed: " << infoLog << std::endl;
    }

    GLuint computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);

    glGetProgramiv(computeProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(computeProgram, 512, nullptr, infoLog);
        std::cerr << "Compute program linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(computeShader);
    return computeProgram;
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
        capacity *= 2;
    }
    return capacity;
}

void ManageObjectStateBufferCapacity(std::vector<objectStateCpu> &stateData, size_t objectCount)
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

    stateData.resize(objectCount);
    if (newCapacity == objectStateCapacity)
    {
        return;
    }

    if (newCapacity < objectStateCapacity)
    {
        std::vector<objectStateCpu> compactedStateData(stateData.begin(), stateData.end());
        compactedStateData.reserve(newCapacity);
        stateData.swap(compactedStateData);
    }
    else
    {
        stateData.reserve(newCapacity);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, objectDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, newCapacity * sizeof(objectStateCpu), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    objectStateCapacity = newCapacity;
}

void UploadObjectState(size_t objectCount)
{
    if (objectCount == 0)
    {
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, objectDataSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, objectCount * sizeof(objectStateCpu),
                    objectData.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void RenderingPipeline(const std::vector<glm::vec4> &gpuGridVertices)
{
    std::vector<unsigned int> gridIndices = CreateGridIndices(gridDivisions);
    gridNodeCount = gpuGridVertices.size();
    gridIndexCount = gridIndices.size();

    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);
    glGenBuffers(1, &gridEBO);
    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, gpuGridVertices.size() * sizeof(glm::vec4),
                 gpuGridVertices.data(), GL_DYNAMIC_DRAW);
    // location 0, 3 positions, type float, not normalized, stride is size of glm::vec4, offset is 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);
    glEnableVertexAttribArray(0);
    // Bind the EBO and upload the index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndices.size() * sizeof(unsigned int),
                 gridIndices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

void ComputePipeline(const std::vector<glm::vec4> &gpuGridVertices)
{
    // Create the Shader Storage Buffer Object (SSBO) for sphere state data.
    glGenBuffers(1, &objectDataSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, objectDataSSBO);
    ManageObjectStateBufferCapacity(objectData, objs.size());
    // Bind objectDataSSBO: binding point 0
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, objectDataSSBO);

    // Create SSBO for the base grid positions readonly buffer.
    glGenBuffers(1, &baseGridSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, baseGridSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gpuGridVertices.size() * sizeof(glm::vec4),
                 gpuGridVertices.data(), GL_STATIC_DRAW);
    // Bind baseGridSSBO: binding point 1
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, baseGridSSBO);

    glGenBuffers(1, &deformedGridSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, deformedGridSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gpuGridVertices.size() * sizeof(glm::vec4), nullptr,
                 GL_DYNAMIC_DRAW);
    // Bind deformedGridSSBO: binding point 2
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, deformedGridSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void InitializeGpuComputation()
{
    std::vector<glm::vec4> gpuGridVertices = CreateBaseGridGPU();
    RenderingPipeline(gpuGridVertices);
    ComputePipeline(gpuGridVertices);
}

void RunGridCompute(size_t objectCount)
{
    glUseProgram(gridComputeProgram);
    glUniform1ui(glGetUniformLocation(gridComputeProgram, "u_objectCount"),
                 static_cast<GLuint>(objectCount));
    const GLuint gridWidth = static_cast<GLuint>(gridDivisions + 1);
    const GLuint gridHeight = static_cast<GLuint>(gridDivisions + 1);
    glUniform1ui(glGetUniformLocation(gridComputeProgram, "u_gridWidth"), gridWidth);
    glUniform1ui(glGetUniformLocation(gridComputeProgram, "u_gridHeight"), gridHeight);

    const GLuint groupCountX = (gridWidth + kGridLocalSizeX - 1) / kGridLocalSizeX;
    const GLuint groupCountY = (gridHeight + kGridLocalSizeY - 1) / kGridLocalSizeY;
    glDispatchCompute(groupCountX, groupCountY, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBuffer(GL_COPY_READ_BUFFER, deformedGridSSBO);
    glBindBuffer(GL_COPY_WRITE_BUFFER, gridVBO);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                        gridNodeCount * sizeof(glm::vec4));
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}

void Cleanup(GLuint shaderProgram, GLuint computeProgram)
{
    for (auto &obj : objs)
    {
        glDeleteVertexArrays(1, &obj.VAO);
        glDeleteBuffers(1, &obj.VBO);
    }

    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    glDeleteBuffers(1, &gridEBO);
    glDeleteBuffers(1, &objectDataSSBO);
    glDeleteBuffers(1, &baseGridSSBO);
    glDeleteBuffers(1, &deformedGridSSBO);
    glDeleteProgram(shaderProgram);
    glDeleteProgram(computeProgram);

    objs.clear();
    objectData.clear();
    gridVAO = 0;
    gridVBO = 0;
    gridEBO = 0;
    objectDataSSBO = 0;
    baseGridSSBO = 0;
    deformedGridSSBO = 0;
    gridComputeProgram = 0;
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

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        Object &initializingObject = objs.back();
        initializingObject.mass *= 1.0 + 1.0 * deltaTime;
        initializingObject.rs = (2 * kGravitationalConstant * initializingObject.mass) /
                                (kSpeedOfLight * kSpeedOfLight);
        initializingObject.radius =
            pow((3 * initializingObject.mass / initializingObject.density) / (4 * 3.14159265359f),
                1.0f / 3.0f) /
            100000.0f;
        initializingObject.UpdateVertexBuffer();
    }
}

void UpdateCamera(GLuint shaderProgram, GLint viewLocation, glm::vec3 cameraPosition)
{
    glUseProgram(shaderProgram);
    glm::mat4 view = glm::lookAt(cameraPosition, cameraPosition + cameraFront, cameraUp);
    glUniformMatrix4fv(viewLocation, 1, GL_FALSE, glm::value_ptr(view));
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
        if (key == GLFW_KEY_UP && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            if (!shiftPressed)
                lastObj.position[1] += 0.5;
            else
                lastObj.position[2] += 0.5;
        }
        if (key == GLFW_KEY_DOWN && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            if (!shiftPressed)
                lastObj.position[1] -= 0.5;
            else
                lastObj.position[2] -= 0.5;
        }
        if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT))
            lastObj.position[0] += 0.5;
        if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT))
            lastObj.position[0] -= 0.5;
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

void DrawGrid(GLuint shaderProgram, GLuint gridVAO, size_t indexCount, GLint objectColorLocation)
{
    glUseProgram(shaderProgram);
    glUniform4f(objectColorLocation, 1.0f, 1.0f, 1.0f, 0.25f);
    glm::mat4 model = glm::mat4(1.0f);
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glBindVertexArray(gridVAO);
    // glPointSize(5.0f);
    glDrawElements(GL_LINES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void DrawObjects(const std::vector<Object> &objects, GLint modelLocation, GLint objectColorLocation)
{
    for (const auto &obj : objects)
    {
        glUniform4f(objectColorLocation, obj.color.r, obj.color.g, obj.color.b, obj.color.a);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, obj.position);
        glUniformMatrix4fv(modelLocation, 1, GL_FALSE, glm::value_ptr(model));
        glBindVertexArray(obj.VAO);
        glDrawArrays(GL_TRIANGLES, 0, obj.vertexCount / 3);
    }
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
    std::mt19937 rng(std::random_device{}());
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
