#include "gravity_sim_3Dgrid_function.hpp"
#include <chrono>

// Simulation and compute configuration.
const double kGravitationalConstant = 6.6743e-11;
const float kSpeedOfLight = 299792458.0; // đổi từ float qua double
float initMass = 5.0f * pow(10, 20) / 5; // đổi từ float qua double
int numRandomObjects = 200;
float gridSize = 30000.0f;
int gridDivisions = 200;

// Must match layout(local_size_x = 16, local_size_y = 16) in grid.comp.
extern constexpr GLuint kGridLocalSizeX = 16;
extern constexpr GLuint kGridLocalSizeY = 16;

// Mutable simulation and camera state.
bool running = true;
bool pause = false;
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 1.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
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
float CEll_SIZE = gridSize / gridDivisions;
float worldMin = -gridSize / 2.0f;
float worldMax = gridSize / 2.0f;
GLuint objectCellID = 0;

// numerical error
double dt = 1.0 / 96.0;

int main()
{
    float epsilon = 10.0f;

    // Initialize OpenGL resources.
    GLuint renderingShaderProgram = 0;
    GLuint computeShaderProgram = 0;
    GLuint objectRenderingShaderProgram = 0;
    GLuint nbodyShaderProgram = 0;
    GLuint integrateShaderProgram = 0;
    GLint modelLoc = -1;
    GLint objectColorLoc = -1;
    GLint viewLoc = -1;
    GLint objectViewLoc = -1;
    // Initialize GLFW, create a window, and set up OpenGL context.
    GLFWwindow *window = StartGLU();
    if (window == nullptr)
    {
        throw std::runtime_error("Failed to initialize GLFW and create a window.");
        return -1;
    }

    InitializeGlfwCallbacks(window);
    // Pass a file path here to load objects from CSV instead of using defaults.
    auto startTime = std::chrono::high_resolution_clock::now();
    objs = CreateObjects({}, numRandomObjects);
    std::cout << "Object list: " << std::endl;
    for (size_t i = 0; i < objs.size(); ++i)
    {
        const Object &obj = objs[i];
        std::cout << "Object " << i + 1 << ":";
        std::cout << "   Mass: " << obj.mass << std::endl;
        std::cout << "   Radius: " << obj.radius << std::endl;
        std::cout << "   Position: " << obj.position.x << ", " << obj.position.y << ", "
                  << obj.position.z << std::endl;
        std::cout << "   Velocity: " << obj.velocity.x << ", " << obj.velocity.y << ", "
                  << obj.velocity.z << std::endl;
        std::cout << "   Density: " << obj.density << std::endl;
    }

    auto endObjectTime = std::chrono::high_resolution_clock::now();
    auto objectCreationDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(endObjectTime - startTime);
    std::cout << "Object creation time: " << objectCreationDuration.count() << " ms" << std::endl;
    size_t objectCount = objs.size();
    std::cout << "Initial object count: " << objectCount << std::endl;

    // Create shader programs, then initialize rendering uniforms and camera position.
    CreateShaderProgram(renderingShaderProgram, computeShaderProgram, objectRenderingShaderProgram,
                        nbodyShaderProgram, integrateShaderProgram);

    // Initialize rendering state, including shader uniforms and camera position.
    InitializeRenderingState(renderingShaderProgram, modelLoc, objectColorLoc, viewLoc, cameraPos);
    glUseProgram(objectRenderingShaderProgram);
    objectViewLoc = glGetUniformLocation(objectRenderingShaderProgram, "view");
    const GLint objectProjectionLoc =
        glGetUniformLocation(objectRenderingShaderProgram, "projection");
    const glm::mat4 objectProjection =
        glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 750000.0f);
    glUniformMatrix4fv(objectProjectionLoc, 1, GL_FALSE, glm::value_ptr(objectProjection));
    InitializeObjectRenderingResources();

    // Initialize the grid pipeline for GPU computation.
    auto startGridTime = std::chrono::high_resolution_clock::now();
    std::vector<glm::vec4> gridVertices = CreateBaseGridGPU();
    std::vector<unsigned int> gridIndices = CreateGridIndices(gridDivisions);

    // Create GPU buffers first: the grid VAO sources vertices directly from DeformedGrid.
    ComputePipeline(gridVertices);
    objectGpuState(0, objectCount);
    InitializeGridRenderingResources(gridVertices, gridIndices);

    // Initialize the grid compute pipeline with the base positions of the grid vertices.
    auto endGridTime = std::chrono::high_resolution_clock::now();
    std::cout << "Grid initialization time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endGridTime - startGridTime)
                     .count()
              << " ms" << std::endl;

    // mass aggregation
    std::cout << "Cell size: " << CEll_SIZE << std::endl;
    // CreateObjectCellID(objs, objectCellID);

    int frameCount = 0;
    double fps = 0.0;
    double frameTime = 0.0;
    std::cout << "Starting main loop..." << std::endl;
    auto startLoopTime = std::chrono::high_resolution_clock::now();

    double T = 10.0;
    int numSteps = static_cast<int>(T / dt);
    std::vector<std::vector<glm::vec3>> positions;
    constexpr int trajectorySampleInterval = 96;
    const size_t trajectorySampleCount = static_cast<size_t>(
        (numSteps + trajectorySampleInterval - 1) / trajectorySampleInterval + 1);
    positions.reserve(trajectorySampleCount);
    bool saved = false;

    // Capture the initial GPU state for trajectory export.
    const auto captureGpuTrajectorySample = [&]()
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
    };
    captureGpuTrajectorySample();

    int numofFrames = 0;
    while (!glfwWindowShouldClose(window) && running == true)
    {
        std::cout << "Frame: " << numofFrames << ", Delta Time: " << deltaTime << " s" << std::endl;
        // Calculate delta time for smooth motion and physics updates.
        float currentFrame = glfwGetTime(); // đổi từ float qua double
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        BeginFrame();
        UpdateCamera(renderingShaderProgram, viewLoc, cameraPos);
        UpdateCamera(objectRenderingShaderProgram, objectViewLoc, cameraPos);

        RunObjectStateSync(window, objectCount);
        RunGpuStateTransition(nbodyShaderProgram, integrateShaderProgram, objectCount, epsilon,
                              pause);
        if (!saved && (numofFrames + 1) % trajectorySampleInterval == 0)
        {
            captureGpuTrajectorySample();
            if (positions.size() >= trajectorySampleCount)
            {
                SavePositions("numerical_error/optimized_positions.csv", positions,
                              dt * trajectorySampleInterval);
                saved = true;
            }
        }
        RunGridCompute(computeShaderProgram, objectCount);
        // Render the deformed grid and all objects in the scene.
        DrawGrid(renderingShaderProgram, gridVAO, gridIndexCount, objectColorLoc);
        // Render each object in the simulation.
        DrawObjects(objectRenderingShaderProgram, objs);
        // Swap buffers and poll for events.
        glfwSwapBuffers(window);
        glfwPollEvents();
        // Update frame count and calculate FPS every second.
        frameCount++;
        auto CurrentFrameTime = std::chrono::high_resolution_clock::now();
        auto frameDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(CurrentFrameTime - startLoopTime);
        if (frameDuration.count() >= 1000.0)
        {
            fps = frameCount / (frameDuration.count() / 1000.0);
            frameTime = 1.0 / fps;
            std::cout << "Object count: " << objectCount << std::endl;
            std::cout << "CurrentState buffer: "
                      << gpuMemoryManager.Get(BufferRole::CurrentState).Handle()
                      << ", NextState buffer: "
                      << gpuMemoryManager.Get(BufferRole::NextState).Handle() << std::endl;
            std::cout << "FPS: " << fps << std::endl
                      << "Frame Time: " << frameTime << " s" << std::endl;
            startLoopTime = CurrentFrameTime;
            frameCount = 0;
        }
        numofFrames++;
    }
    if (!saved && !positions.empty())
    {
        SavePositions("numerical_error/optimized_positions.csv", positions,
                      dt * trajectorySampleInterval);
    }
    // Main loop ends when window is closed or running is set to false.
    // Cleanup OpenGL resources and exit.
    Cleanup(renderingShaderProgram, computeShaderProgram, objectRenderingShaderProgram,
            nbodyShaderProgram, integrateShaderProgram);
    return 0;
}
