#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <array>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <random>
#include <vector>

const char *vertexShaderSource = R"glsl(
#version 430 core
layout(location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

#define MAX_OBJS 128
layout(std430, binding = 0) readonly buffer ObjectState
{
    vec4 data[MAX_OBJS];
};

uniform int u_numObjs;
uniform bool u_isGrid; 

void main() 
{
    vec3 vertexPos = aPos;
    if (u_isGrid) 
    {
        float totalDisplacement = 0.0;
        for(int i = 0; i < u_numObjs; ++i) 
        {
            vec3 objPos = data[i].xyz;
            float objRs = data[i].w;
            vec3 toObject = objPos - vertexPos;
            float distance_m = length(toObject) * 1000.0;
            
            distance_m = max(distance_m, objRs * 1.0001); 
            
            if (objRs > 0.0) 
            {
                totalDisplacement -= (2.0e11 * objRs) / distance_m;
            }
        }
        vertexPos.y = aPos.y + totalDisplacement;
    }
    gl_Position = projection * view * model * vec4(vertexPos, 1.0);
}
)glsl";

const char *fragmentShaderSource = R"glsl(
#version 330 core
out vec4 FragColor;
uniform vec4 objectColor;
void main() 
{
    FragColor = objectColor;
}
)glsl";

const double G = 6.6743e-11;
const float c = 299792458.0;

// Global simulation and camera state.
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

float initMass = 5.0f * pow(10, 20) / 5;
int numRandomObjects = 100;
float gridSize = 30000.0f;
int gridDivisions = 200;

class Object;

// Function prototypes, kept together in implementation order.
GLFWwindow *StartGLU();
GLuint CreateShaderProgram(const char *vertexSource, const char *fragmentSource);
void CreateMeshBuffers(GLuint &VAO, GLuint &VBO, const float *vertices, size_t vertexCount, GLuint *EBO = nullptr,
                       const unsigned int *indices = nullptr, size_t indexCount = 0);
void InitializeRenderingPipeline();
void InitializeSimulationPipeline();
void Cleanup(GLuint shaderProgram);
void UpdateCam(GLuint shaderProgram, GLint viewLoc, glm::vec3 cameraPos);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void InitializeGlfwCallbacks(GLFWwindow *window);
glm::vec3 sphericalToCartesian(float r, float theta, float phi);
void DrawGrid(GLuint shaderProgram, GLuint gridVAO, size_t vertexCount);
std::vector<float> CreateGridVertices(float size, int divisions);
std::vector<unsigned int> CreateGridIndices(int divisions);
std::vector<Object> CreateRandomOrbiters(int count, const glm::vec3 &center, float centralMass);

class Object
{
  public:
    GLuint VAO, VBO;
    glm::vec3 position = glm::vec3(400, 300, 0);
    glm::vec3 velocity = glm::vec3(0, 0, 0);
    size_t vertexCount;
    glm::vec4 color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

    bool Initalizing = false;
    bool Launched = false;
    bool target = false;

    float mass;
    float density;
    float radius;
    float rs;

    glm::vec3 LastPos = position;
    // Constructor object with initial position, velocity, mass, and density. Calculates radius and Schwarzschild
    // radius.
    Object(glm::vec3 initPosition, glm::vec3 initVelocity, float mass, float density = 3344)
    {
        this->position = initPosition;
        this->velocity = initVelocity;
        this->mass = mass;
        this->density = density;
        this->radius = pow(((3 * this->mass / this->density) / (4 * 3.14159265359)), (1.0f / 3.0f)) / 100000;
        this->rs = (2 * G * this->mass) / (c * c);
        // Generate vertices (centered at origin)
        std::vector<float> vertices = DrawSphereMesh();
        vertexCount = vertices.size();
        CreateMeshBuffers(VAO, VBO, vertices.data(), vertexCount);
    }
    // Generate sphere mesh (CPU)
    std::vector<float> DrawSphereMesh()
    {
        std::vector<float> vertices;
        int stacks = 10;
        int sectors = 10;
        for (float i = 0.0f; i <= stacks; ++i)
        {
            float theta1 = (i / stacks) * glm::pi<float>();
            float theta2 = (i + 1) / stacks * glm::pi<float>();
            for (float j = 0.0f; j < sectors; ++j)
            {
                float phi1 = j / sectors * 2 * glm::pi<float>();
                float phi2 = (j + 1) / sectors * 2 * glm::pi<float>();
                glm::vec3 v1 = sphericalToCartesian(radius, theta1, phi1);
                glm::vec3 v2 = sphericalToCartesian(radius, theta1, phi2);
                glm::vec3 v3 = sphericalToCartesian(radius, theta2, phi1);
                glm::vec3 v4 = sphericalToCartesian(radius, theta2, phi2);
                // Triangle 1: v1-v2-v3
                vertices.insert(vertices.end(), {v1.x, v1.y, v1.z});
                vertices.insert(vertices.end(), {v2.x, v2.y, v2.z});
                vertices.insert(vertices.end(), {v3.x, v3.y, v3.z});
                // Triangle 2: v2-v4-v3
                vertices.insert(vertices.end(), {v2.x, v2.y, v2.z});
                vertices.insert(vertices.end(), {v4.x, v4.y, v4.z});
                vertices.insert(vertices.end(), {v3.x, v3.y, v3.z});
            }
        }
        return vertices;
    }

    void UpdatePos()
    {
        this->position += this->velocity * deltaTime;
        this->radius = pow(((3 * this->mass / this->density) / (4 * 3.14159265359)), (1.0f / 3.0f)) / 100000;
    }
    void UpdateVertices()
    {
        // Generate new vertices with current radius
        std::vector<float> vertices = DrawSphereMesh();

        // Update the VBO with new vertex data
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    }
    glm::vec3 GetPos() const { return this->position; }
    void accelerate(float x, float y, float z) { this->velocity += glm::vec3(x, y, z) * deltaTime; }
    float CheckCollision(const Object &other)
    {
        float dx = other.position[0] - this->position[0];
        float dy = other.position[1] - this->position[1];
        float dz = other.position[2] - this->position[2];
        float distance = std::pow(dx * dx + dy * dy + dz * dz, (1.0f / 2.0f));
        if (other.radius + this->radius > distance)
        {
            return -0.2f;
        }
        return 1.0f;
    }
};

// Global OpenGL resources and simulation objects.
std::vector<Object> objs = {};
GLuint gridVAO, gridVBO, gridEBO;
GLuint objectStateSSBO;
size_t gridIndexCount = 0;
constexpr int maxObjects = 128;
std::array<glm::vec4, maxObjects> objectStateData{};

int main()
{
    GLFWwindow *window = StartGLU();
    GLuint shaderProgram = CreateShaderProgram(vertexShaderSource, fragmentShaderSource);

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
            objectStateData[i] = glm::vec4(objs[i].GetPos(), objs[i].rs);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, objectStateSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, activeObjs * sizeof(objectStateData[0]), objectStateData.data());
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

void InitializeGlfwCallbacks(GLFWwindow *window)
{
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

GLFWwindow *StartGLU()
{
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW, panic" << std::endl;
        return nullptr;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(1920, 1080, "GRAVITY SIMULATION - 3D GRID", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        glfwTerminate();
        return nullptr;
    }

    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, 1920, 1080);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,
                GL_ONE_MINUS_SRC_ALPHA); // Standard blending for transparency

    return window;
}

GLuint CreateShaderProgram(const char *vertexSource, const char *fragmentSource)
{
    // Vertex shader
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

    // Fragment shader
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

    // Shader program
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
void CreateMeshBuffers(GLuint &VAO, GLuint &VBO, const float *vertices, size_t vertexCount, GLuint *EBO,
                       const unsigned int *indices, size_t indexCount)
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
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(unsigned int), indices, GL_STATIC_DRAW);
    }

    glBindVertexArray(0);
}

void InitializeRenderingPipeline()
{
    std::vector<float> gridVertices = CreateGridVertices(gridSize, gridDivisions);
    std::vector<unsigned int> gridIndices = CreateGridIndices(gridDivisions);
    gridIndexCount = gridIndices.size();

    CreateMeshBuffers(gridVAO, gridVBO, gridVertices.data(), gridVertices.size(), &gridEBO, gridIndices.data(),
                      gridIndices.size());
}

void InitializeSimulationPipeline()
{
    glGenBuffers(1, &objectStateSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, objectStateSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(objectStateData), objectStateData.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, objectStateSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void Cleanup(GLuint shaderProgram)
{
    for (auto &obj : objs)
    {
        glDeleteVertexArrays(1, &obj.VAO);
        glDeleteBuffers(1, &obj.VBO);
    }

    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    glDeleteBuffers(1, &gridEBO);
    glDeleteBuffers(1, &objectStateSSBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
}

void UpdateCam(GLuint shaderProgram, GLint viewLoc, glm::vec3 cameraPos)
{
    glUseProgram(shaderProgram);
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    float cameraSpeed = 1000.0f * deltaTime;
    bool shiftPressed = (mods & GLFW_MOD_SHIFT) != 0;
    Object &lastObj = objs[objs.size() - 1];

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        cameraPos += cameraSpeed * cameraFront;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        cameraPos -= cameraSpeed * cameraFront;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        cameraPos -= cameraSpeed * glm::normalize(glm::cross(cameraFront, cameraUp));
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        cameraPos += cameraSpeed * glm::normalize(glm::cross(cameraFront, cameraUp));
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        cameraPos += cameraSpeed * cameraUp;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        cameraPos -= cameraSpeed * cameraUp;
    }

    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
    {
        pause = true;
    }
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_RELEASE)
    {
        pause = false;
    }

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        glfwTerminate();
        glfwWindowShouldClose(window);
        running = false;
    }

    // init arrows pos up down left right
    if (!objs.empty() && objs[objs.size() - 1].Initalizing)
    {
        if (key == GLFW_KEY_UP && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            if (!shiftPressed)
            {
                objs[objs.size() - 1].position[1] += 0.5;
            }
        };
        if (key == GLFW_KEY_DOWN && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            if (!shiftPressed)
            {
                objs[objs.size() - 1].position[1] -= 0.5;
            }
        }
        if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            objs[objs.size() - 1].position[0] += 0.5;
        };
        if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            objs[objs.size() - 1].position[0] -= 0.5;
        };
        if (key == GLFW_KEY_UP && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            objs[objs.size() - 1].position[2] += 0.5;
        };

        if (key == GLFW_KEY_DOWN && (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            objs[objs.size() - 1].position[2] -= 0.5;
        }
    };
};
void mouse_callback(GLFWwindow *window, double xpos, double ypos)
{

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

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
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            objs.emplace_back(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0f, 0.0f, 0.0f), initMass);
            objs[objs.size() - 1].Initalizing = true;
        };
        if (action == GLFW_RELEASE)
        {
            objs[objs.size() - 1].Initalizing = false;
            objs[objs.size() - 1].Launched = true;
        };
    };
    // if (!objs.empty() && button == GLFW_MOUSE_BUTTON_RIGHT &&
    // objs[objs.size()-1].Initalizing) {
    //     if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    //         objs[objs.size()-1].mass *= 1.2;}
    //         std::cout<<"MASS: "<<objs[objs.size()-1].mass<<std::endl;
    // }
};
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    float cameraSpeed = 50000.0f * deltaTime;
    if (yoffset > 0)
    {
        cameraPos += cameraSpeed * cameraFront;
    }
    else if (yoffset < 0)
    {
        cameraPos -= cameraSpeed * cameraFront;
    }
}

glm::vec3 sphericalToCartesian(float r, float theta, float phi)
{
    float x = r * sin(theta) * cos(phi);
    float y = r * cos(theta);
    float z = r * sin(theta) * sin(phi);
    return glm::vec3(x, y, z);
};
void DrawGrid(GLuint shaderProgram, GLuint gridVAO, size_t indexCount)
{
    glUseProgram(shaderProgram);
    glm::mat4 model = glm::mat4(1.0f); // Identity matrix for the grid
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

    glBindVertexArray(gridVAO);
    glPointSize(5.0f);
    // glDrawArrays(GL_LINES, 0, vertexCount / 3);
    glDrawElements(GL_LINES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

std::vector<float> CreateGridVertices(float size, int divisions)
{
    std::vector<float> vertices;
    int numNodesPerSide = divisions + 1;
    vertices.reserve(numNodesPerSide * numNodesPerSide * 3);

    float step = size / divisions;
    float halfSize = size / 2.0f;
    // Áp dụng độ lệch không gian -900.0f để né điểm kỳ dị Schwarzschild
    // Khi dữ liệu được nạp vào VRAM, thuộc tính aPos.y của mọi đỉnh lưới sẽ
    // khởi tạo với giá trị $ - 900.0$. Phép toán vector nội suy vec3 toObject =
    // u_objPos[i] - vertexPos; bên trong GPU sẽ tự động duy trì khoảng cách
    // không gian tối thiểu là 900 đơn vị theo trục tung
    float baseY = -900.0f;

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
};
std::vector<unsigned int> CreateGridIndices(int divisions)
{
    std::vector<unsigned int> indices;
    int rowSize = divisions + 1;
    // Tính toán trước tổng số lượng index
    // Số cạnh ngang: divisions * (divisions + 1)
    // Số cạnh dọc: divisions * (divisions + 1)
    // Mỗi cạnh cần 2 indices, tổng cộng: 4 * divisions * (divisions + 1)
    size_t totalIndices = 4 * divisions * (divisions + 1);
    indices.reserve(totalIndices);
    // Đường ngang (Horizontal)
    for (int z = 0; z <= divisions; ++z)
    {
        for (int x = 0; x < divisions; ++x)
        {
            int current = z * rowSize + x;
            indices.push_back(current);
            indices.push_back(current + 1);
        }
    }
    // Đường dọc (Vertical)
    for (int z = 0; z < divisions; ++z)
    {
        for (int x = 0; x <= divisions; ++x)
        {
            int current = z * rowSize + x; // index
            indices.push_back(current);
            indices.push_back(current + rowSize);
        }
    }

    return indices;
}
// z = 0, x = 0 => curent  = 0x4 + 0 = 0 ----- 0/1
// z = 0, x = 1 => curent  = 0x4 + 1 = 1 ----- 1/2
// z = 0, x = 2 => curent  = 0x4 + 2 = 2 ----- 2/3

// z=0 ->z<3
// z = 0, x = 0 => curent  = 0x4 + 0 = 0 ----- 0/4
// z = 0, x = 1 => curent  = 0x4 + 1 = 1 ----- 1/2
// z = 0, x = 2 => curent  = 0x4 + 2 = 2 ----- 2/3
// z = 0, x = 3 => curent  = 0x4 + 3 = 3 ----- 3/4

// x = 0, z = 0 => current = 0*4 + 0 = 0 ----- 0/1
// x = 0, z = 1 => current = 0*4 + 1 = 1 ----- 1/2

std::vector<Object> CreateRandomOrbiters(int count, const glm::vec3 &center, float centralMass)
{
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
        generated.back().color = glm::vec4(0.2f + 0.8f * static_cast<float>((i * 7) % 5) / 4.0f,
                                           0.2f + 0.8f * static_cast<float>((i * 11) % 5) / 4.0f,
                                           0.2f + 0.8f * static_cast<float>((i * 13) % 5) / 4.0f, 1.0f);
    }

    return generated;
}