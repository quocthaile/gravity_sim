#include "gravity_sim_3Dgrid_function.h"

// Simulation and compute configuration.
const double kGravitationalConstant = 6.6743e-11;
const float kSpeedOfLight = 299792458.0;
float initMass = 5.0f * pow(10, 20) / 5;
int numRandomObjects = 200;
float gridSize = 30000.0f;
int gridDivisions = 200;

// Must match layout(local_size_x = 16, local_size_y = 16) in grid_compute.glsl.
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
GLuint shaderProgram;
GLuint sphreStateSSBO;
GLuint gridComputeProgram;
GLuint baseGridSSBO = 0;
GLuint deformedGridSSBO = 0;
size_t gridNodeCount = 0;
size_t gridIndexCount = 0;
std::vector<SphreStateCpu> sphreStateData;

int main()
{
    // Initialize GLFW, create a window, and set up OpenGL context.
    GLFWwindow *window = StartGLU();
    InitializeGlfwCallbacks(window);

    // Load shader sources and create shader programs.
    std::string vertexShaderSource = LoadShaderSource("shaders/vertex_shader.glsl");
    std::string fragmentShaderSource = LoadShaderSource("shaders/fragment_shader.glsl");
    std::string computeShaderSource = LoadShaderSource("shaders/grid_compute.glsl");
    shaderProgram = CreateShaderProgram(vertexShaderSource.c_str(), fragmentShaderSource.c_str());
    gridComputeProgram = CreateComputeProgram(computeShaderSource.c_str());
    // Get uniform locations for model, object color, and view matrices.
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    // Set the shader program to use for rendering.
    glUseProgram(shaderProgram);

    // projection matrix
    glm::mat4 projection =
        glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 750000.0f);
    GLint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    cameraPos = glm::vec3(0.0f, 1000.0f, 5000.0f);

    // Initialize the simulation with two primary objects: a moon and a planet.
    objs = {Object(glm::vec3(3844, 0, 0), glm::vec3(0, 0, 228), 7.34767309 * pow(10, 22), 3344),
            Object(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), 5.97219 * pow(10, 24), 5515)};
    // Add random objects to the simulation
    auto randomBodies =
        CreateRandomOrbiters(numRandomObjects, glm::vec3(0.0f, 0.0f, 0.0f), 5.97219e24f);
    objs.insert(objs.end(), randomBodies.begin(), randomBodies.end());

    // Initialize the grid pipeline for GPU computation.
    InitializeGridPipeline();

    while (!glfwWindowShouldClose(window) && running == true)
    {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        UpdateCamera(shaderProgram, viewLoc, cameraPos);
        if (!objs.empty() && objs.back().initializing)
        {
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
            {
                // Increase mass by 1% per second
                objs.back().mass *= 1.0 + 1.0 * deltaTime;
                objs.back().rs = (2 * kGravitationalConstant * objs.back().mass) /
                                 (kSpeedOfLight * kSpeedOfLight);
                // Update radius based on new mass
                objs.back().radius =
                    pow((3 * objs.back().mass / objs.back().density) / (4 * 3.14159265359f),
                        1.0f / 3.0f) /
                    100000.0f;
                // Update vertex data
                objs.back().UpdateVertexBuffer();
            }
        }

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
        glUseProgram(shaderProgram);
        const size_t objectCount = objs.size();
        if (sphreStateData.size() < objectCount)
        {
            sphreStateData.resize(objectCount);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sphreStateSSBO);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sphreStateData.size() * sizeof(SphreStateCpu),
                         sphreStateData.data(), GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        int activeObjs = static_cast<int>(objectCount);

        for (size_t i = 0; i < objectCount; ++i)
        {
            sphreStateData[i].position_mass = glm::vec4(objs[i].GetPosition(), objs[i].mass);
            sphreStateData[i].velocity_radius = glm::vec4(objs[i].velocity, objs[i].rs);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sphreStateSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, activeObjs * sizeof(sphreStateData[0]),
                        sphreStateData.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        // Run the compute shader to update the grid based on the current state of the objects.
        RunGridCompute(activeObjs);

        glUseProgram(shaderProgram);
        glUniform4f(objectColorLoc, 1.0f, 1.0f, 1.0f, 0.25f);
        DrawGrid(shaderProgram, gridVAO, gridIndexCount);
        // Render each object in the simulation.
        for (auto &obj : objs)
        {

            glUniform4f(objectColorLoc, obj.color.r, obj.color.g, obj.color.b, obj.color.a);
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, obj.position);
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glBindVertexArray(obj.VAO);
            glDrawArrays(GL_TRIANGLES, 0, obj.vertexCount / 3);
        }
        // Swap buffers and poll for events.
        glfwSwapBuffers(window);
        glfwPollEvents();
    } // Main loop ends when window is closed or running is set to false.
    // Cleanup OpenGL resources and exit.
    Cleanup(shaderProgram);
    return 0;
}
