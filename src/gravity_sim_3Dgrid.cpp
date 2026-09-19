#include "gravity_sim_3Dgrid_function.h"

// Simulation and compute configuration.
const double kGravitationalConstant = 6.6743e-11;
const float kSpeedOfLight = 299792458.0;
float initMass = 5.0f * pow(10, 20) / 5;
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
GLuint gridVBO = 0;
GLuint gridEBO = 0;
GLuint objectDataSSBO;
GLuint gridComputeProgram;
GLuint baseGridSSBO = 0;
GLuint deformedGridSSBO = 0;
size_t gridNodeCount = 0;
size_t gridIndexCount = 0;
std::vector<objectStateCpu> objectData;
size_t objectStateCapacity = 0;

int main()
{
    // Initialize OpenGL resources.
    GLuint shaderProgram = 0;
    GLint modelLoc = -1;
    GLint objectColorLoc = -1;
    GLint viewLoc = -1;
    // Initialize GLFW, create a window, and set up OpenGL context.
    GLFWwindow *window = StartGLU();
    InitializeGlfwCallbacks(window);
    // Pass a file path here to load objects from CSV instead of using defaults.
    objs = CreateObjects({}, numRandomObjects);
    size_t objectCount = objs.size();
    // Set up rendering resources, including shaders and camera position.
    InitializeRenderingResources(shaderProgram, modelLoc, objectColorLoc, viewLoc, cameraPos);
    // Initialize the grid pipeline for GPU computation.
    InitializeGpuComputation();
    // Main render loop: update object states, run compute shader, and render scene.
    while (!glfwWindowShouldClose(window) && running == true)
    {
        // Calculate delta time for smooth motion and physics updates.
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        BeginFrame();
        UpdateCamera(shaderProgram, viewLoc, cameraPos);
        UpdateInitializingObject(window);

        // N-body gravitational interactions between all objects in the simulation.
        float epsilon = 10.0f;
        for (auto &obj : objs)
        {
            for (auto &obj2 : objs)
            {
                if (&obj2 != &obj && !obj.initializing && !obj2.initializing)
                {
                    float dx = obj2.GetPosition()[0] - obj.GetPosition()[0];
                    float dy = obj2.GetPosition()[1] - obj.GetPosition()[1];
                    float dz = obj2.GetPosition()[2] - obj.GetPosition()[2];
                    float distance = sqrt(dx * dx + dy * dy + dz * dz);
                    if (distance > 0)
                    {
                        std::vector<float> direction = {dx / distance, dy / distance,
                                                        dz / distance};
                        distance *= 1000;
                        double gravitationalForce =
                            (kGravitationalConstant * obj.mass * obj2.mass) /
                            (distance * distance + epsilon * epsilon);
                        float acceleration = gravitationalForce / obj.mass;
                        std::vector<float> accelerationVector = {direction[0] * acceleration,
                                                                 direction[1] * acceleration,
                                                                 direction[2] * acceleration};
                        if (!pause)
                        {
                            obj.Accelerate(accelerationVector[0], accelerationVector[1],
                                           accelerationVector[2]);
                        }
                        obj.velocity *= obj.CheckCollision(obj2);
                    }
                }
            }
            if (obj.initializing)
            {
                obj.radius =
                    pow(((3 * obj.mass / obj.density) / (4 * 3.14159265359)), (1.0f / 3.0f)) /
                    100000;
                obj.UpdateVertexBuffer();
            }
            if (!pause)
            {
                obj.UpdatePosition();
            }
        }

        // Upload CPU state, compute the grid on the GPU, then render both grid and objects.
        size_t checkObjectCount = objs.size();
        if (checkObjectCount != objectCount)
        {
            objectCount = checkObjectCount;
            ManageObjectStateBufferCapacity(objectData, objectCount);
        }
        // Update the object data with the current positions, velocities, and masses of all objects.
        for (size_t i = 0; i < objectCount; ++i)
        {
            objectData[i].position_mass = glm::vec4(objs[i].GetPosition(), objs[i].mass);
            objectData[i].velocity_radius = glm::vec4(objs[i].velocity, objs[i].rs);
        }
        // Upload the object state data to the GPU for use in the compute shader.
        UploadObjectState(objectCount);
        // Run the compute shader to update the grid based on the current state of the objects.
        RunGridCompute(objectCount);
        // Render the deformed grid and all objects in the scene.
        DrawGrid(shaderProgram, gridVAO, gridIndexCount, objectColorLoc);
        // Render each object in the simulation.
        DrawObjects(objs, modelLoc, objectColorLoc);
        // Swap buffers and poll for events.
        glfwSwapBuffers(window);
        glfwPollEvents();
    } // Main loop ends when window is closed or running is set to false.
    // Cleanup OpenGL resources and exit.
    Cleanup(shaderProgram, gridComputeProgram);
    return 0;
}
