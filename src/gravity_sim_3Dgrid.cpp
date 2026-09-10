#include "gravity_sim_3Dgrid_function.h"

// Physical constants and initial simulation configuration.
const double G = 6.6743e-11;
const float c = 299792458.0;
float initMass = 5.0f * pow(10, 20) / 5;
int numRandomObjects = 100;
float gridSize = 30000.0f;
int gridDivisions = 200;

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
GLuint gridVAO, gridVBO, gridEBO;
GLuint sphreStateSSBO;
size_t gridIndexCount = 0;
std::array<sphreStateCPU, maxObjects> sphreStateData{};

int main()
{
    GLFWwindow *window = StartGLU();
    std::string vertexShaderSource = LoadShaderSource("shaders/vertex_shader.glsl");
    std::string fragmentShaderSource = LoadShaderSource("shaders/fragment_shader.glsl");
    GLuint shaderProgram = CreateShaderProgram(vertexShaderSource.c_str(), fragmentShaderSource.c_str());

    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    glUseProgram(shaderProgram);

    InitializeGlfwCallbacks(window);

    // projection matrix
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 750000.0f);
    GLint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    cameraPos = glm::vec3(0.0f, 1000.0f, 5000.0f);

    objs = {
        Object(glm::vec3(3844, 0, 0), glm::vec3(0, 0, 228), 7.34767309 * pow(10, 22), 3344),
        Object(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), 5.97219 * pow(10, 24), 5515),
    };
    // Add random objects to the simulation
    auto randomBodies = CreateRandomOrbiters(numRandomObjects, glm::vec3(0.0f, 0.0f, 0.0f), 5.97219e24f);
    objs.insert(objs.end(), randomBodies.begin(), randomBodies.end());

    InitializeRenderingPipeline();
    InitializeSimulationPipeline();

    // Lấy vị trí Uniforms
    GLint isGridLoc = glGetUniformLocation(shaderProgram, "u_isGrid");
    GLint numObjsLoc = glGetUniformLocation(shaderProgram, "u_numObjs");

    // std::cout << "Earth radius: " << objs[1].radius << std::endl;
    // std::cout << "Moon radius: " << objs[0].radius << std::endl;

    while (!glfwWindowShouldClose(window) && running == true)
    {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        UpdateCam(shaderProgram, viewLoc, cameraPos);
        if (!objs.empty() && objs.back().Initalizing)
        {
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
            {
                // Increase mass by 1% per second
                objs.back().mass *= 1.0 + 1.0 * deltaTime;
                objs.back().rs = (2 * G * objs.back().mass) / (c * c);

                // Update radius based on new mass
                objs.back().radius =
                    pow((3 * objs.back().mass / objs.back().density) / (4 * 3.14159265359f), 1.0f / 3.0f) / 100000.0f;

                // Update vertex data
                objs.back().UpdateVertices();
            }
        }

        // Draw the grid
        glUseProgram(shaderProgram);
        glUniform1i(isGridLoc, 1);
        int activeObjs = static_cast<int>(objs.size());
        if (activeObjs > maxObjects)
            activeObjs = maxObjects;

        for (int i = 0; i < activeObjs; ++i)
        {
            sphreStateData[i].position_mass = glm::vec4(objs[i].GetPos(), objs[i].mass);
            sphreStateData[i].velocity_radius = glm::vec4(objs[i].velocity, objs[i].rs);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sphreStateSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, activeObjs * sizeof(sphreStateData[0]), sphreStateData.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glUniform1i(numObjsLoc, activeObjs);

        // 3. Vẽ lưới - GPU sẽ tự động làm biến dạng lưới trong Vertex Shader
        glUniform4f(objectColorLoc, 1.0f, 1.0f, 1.0f, 0.25f);
        DrawGrid(shaderProgram, gridVAO, gridIndexCount);
        glUniform1i(isGridLoc, 0);
        // Draw the triangle
        float epsilon = 10.0f;
        for (auto &obj : objs)
        {
            glUniform4f(objectColorLoc, obj.color.r, obj.color.g, obj.color.b, obj.color.a);

            for (auto &obj2 : objs)
            {
                if (&obj2 != &obj && !obj.Initalizing && !obj2.Initalizing)
                {
                    float dx = obj2.GetPos()[0] - obj.GetPos()[0];
                    float dy = obj2.GetPos()[1] - obj.GetPos()[1];
                    float dz = obj2.GetPos()[2] - obj.GetPos()[2];
                    float distance = sqrt(dx * dx + dy * dy + dz * dz);

                    if (distance > 0)
                    {
                        std::vector<float> direction = {dx / distance, dy / distance, dz / distance};
                        distance *= 1000;
                        // double Gforce = (G * obj.mass * obj2.mass) /
                        // (distance * distance);
                        double Gforce = (G * obj.mass * obj2.mass) / (distance * distance + epsilon * epsilon);

                        float acc1 = Gforce / obj.mass;
                        std::vector<float> acc = {direction[0] * acc1, direction[1] * acc1, direction[2] * acc1};
                        if (!pause)
                        {
                            obj.accelerate(acc[0], acc[1], acc[2]);
                        }

                        // collision
                        obj.velocity *= obj.CheckCollision(obj2);
                    }
                }
            }
            if (obj.Initalizing)
            {
                obj.radius = pow(((3 * obj.mass / obj.density) / (4 * 3.14159265359)), (1.0f / 3.0f)) / 100000;
                obj.UpdateVertices();
            }

            // update positions
            if (!pause)
            {
                obj.UpdatePos();
            }

            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, obj.position); // Apply position here
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glBindVertexArray(obj.VAO);
            glDrawArrays(GL_TRIANGLES, 0, obj.vertexCount / 3);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    Cleanup(shaderProgram);
    return 0;
}
