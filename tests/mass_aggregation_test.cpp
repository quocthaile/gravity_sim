#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct State
{
    glm::vec4 position;
    glm::vec4 velocity;
};
struct Control
{
    std::uint32_t initializing, launched, target, id;
};
struct Node
{
    glm::vec4 centerHalf;
    glm::uvec4 children0, children1, data;
};
static_assert(sizeof(Node) == 64);
constexpr double gravity = 6.6743e-11;
constexpr float epsilon = 10.0f;
constexpr GLuint depth = 8;

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

GLuint program(const std::string &path)
{
    std::ifstream input(path);
    require(input.good(), "Cannot read shader: " + path);
    const std::string source((std::istreambuf_iterator<char>(input)), {});
    const char *text = source.c_str();
    const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        std::array<char, 8192> log{};
        glGetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
        throw std::runtime_error(path + ": " + log.data());
    }
    const GLuint result = glCreateProgram();
    glAttachShader(result, shader);
    glLinkProgram(result);
    glDeleteShader(shader);
    glGetProgramiv(result, GL_LINK_STATUS, &success);
    if (!success)
    {
        std::array<char, 8192> log{};
        glGetProgramInfoLog(result, static_cast<GLsizei>(log.size()), nullptr, log.data());
        throw std::runtime_error(path + ": " + log.data());
    }
    return result;
}

std::string shaderPath(const std::string &directory, const char *name)
{
    return (std::filesystem::path(directory) / name).string();
}

GLuint buffer(GLuint binding, GLsizeiptr size, const void *data)
{
    GLuint result = 0;
    glGenBuffers(1, &result);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, result);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, result);
    return result;
}

template <typename T> std::vector<T> read(GLuint handle, std::size_t count)
{
    std::vector<T> result(count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, handle);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       static_cast<GLsizeiptr>(count * sizeof(T)), result.data());
    return result;
}

void buildTree(GLuint builder, GLuint count, GLuint capacity, GLuint aggregationStartQuery = 0)
{
    glUseProgram(builder);
    glUniform1ui(glGetUniformLocation(builder, "u_objectCount"), count);
    glUniform1ui(glGetUniformLocation(builder, "u_nodeCapacity"), capacity);
    const GLint passLocation = glGetUniformLocation(builder, "u_pass");
    const GLint depthLocation = glGetUniformLocation(builder, "u_depth");
    const auto dispatch = [&](GLuint pass, GLuint level, GLuint groups)
    {
        glUniform1ui(passLocation, pass);
        glUniform1ui(depthLocation, level);
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    const GLuint objects = (count + 63) / 64;
    const GLuint nodes = (capacity + 63) / 64;
    dispatch(0, 0, 1);
    for (GLuint level = 0; level < depth; ++level)
    {
        dispatch(1, level, objects);
        dispatch(2, level, nodes);
    }
    dispatch(3, 0, objects);
    if (aggregationStartQuery != 0)
        glQueryCounter(aggregationStartQuery, GL_TIMESTAMP);
    for (GLuint level = depth + 1; level > 0; --level)
        dispatch(4, level - 1, nodes);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
}

void testDirect(const std::string &shaderDirectory, int scenario)
{
    std::vector<State> states = {
        {{0, 0, 0, 0}, {}}, {{1, 0, 0, 0}, {}}, {{0, 0, 0, 0}, {}},
        {{0, 2, 0, 0}, {}}, {{3, 0, 0, 0}, {}}, {{-2, 1, 3, 0}, {}}};
    std::vector<glm::vec4> physical = {
        {1e20f, 0, 0, 0}, {2e20f, 0, 0, 0}, {3e20f, 0, 0, 0},
        {4e20f, 0, 0, 0}, {5e20f, 0, 0, 0}, {6e20f, 0, 0, 0}};
    std::vector<Control> controls = {
        {0, 0, 0, 1}, {0, 0, 0, 2}, {0, 0, 0, 3},
        {1, 0, 0, 4}, {0, 0, 0, 5}, {0, 0, 0, 6}};
    if (scenario == 1)
        for (auto &control : controls) control.initializing = 1;
    if (scenario == 2)
        for (auto &state : states) state.position = glm::vec4(0);
    if (scenario == 3)
        physical[0].x = -physical[0].x;
    if (scenario == 4)
    {
        states.resize(1);
        physical.resize(1);
        controls.resize(1);
    }
    if (scenario == 5 || scenario == 6 || scenario == 8 || scenario == 9)
    {
        const std::size_t sampleCount = scenario >= 8 ? 1024 : 202;
        std::mt19937 generator(123);
        std::uniform_real_distribution<float> distribution(-1000.0f, 1000.0f);
        states.clear();
        physical.clear();
        controls.clear();
        for (std::size_t i = 0; i < sampleCount; ++i)
        {
            glm::vec3 position(distribution(generator), distribution(generator),
                               distribution(generator));
            if (scenario == 6 || scenario == 9)
                position = position * 0.02f + glm::vec3(i % 2 == 0 ? 500.0f : -500.0f, 0, 0);
            states.push_back({glm::vec4(position, 0), glm::vec4(0)});
            physical.push_back(glm::vec4(1e20f * float(1 + i % 7), 0, 0, 0));
            controls.push_back({0, 0, 0, static_cast<std::uint32_t>(i + 1)});
        }
    }
    if (scenario == 7)
        for (auto &object : physical) object.x = std::numeric_limits<float>::max() * 0.5f;
    if (scenario == 10)
        physical[0].x = 0.0f;
    if (scenario == 0)
    {
        states[0].position.w = 4;
        states[0].velocity = glm::vec4(1, -2, 3, 7);
    }
    if (scenario == 11)
    {
        states.assign(1024, State{glm::vec4(42, -7, 3, 0), glm::vec4(0)});
        physical.assign(states.size(), glm::vec4(1e20f, 0, 0, 0));
        controls.assign(states.size(), Control{0, 0, 0, 1});
    }
    const std::vector<glm::vec4> derived(states.size(), glm::vec4(0.75f, 0, 0, 0));
    const auto count = static_cast<GLuint>(states.size());
    std::vector<GLuint> buffers;
    buffers.push_back(buffer(0, states.size() * sizeof(State), states.data()));
    buffers.push_back(buffer(2, physical.size() * sizeof(glm::vec4), physical.data()));
    buffers.push_back(buffer(3, controls.size() * sizeof(Control), controls.data()));
    const GLuint nodeCapacity = 1 + depth * count;
    const GLuint aggregate = buffer(6, nodeCapacity * sizeof(glm::vec4), nullptr);
    const GLuint topology = buffer(7, nodeCapacity * sizeof(Node), nullptr);
    const GLuint links = buffer(8, states.size() * sizeof(glm::uvec2), nullptr);
    const GLuint metadata = buffer(9, sizeof(glm::uvec4), nullptr);
    buffers.insert(buffers.end(), {aggregate, topology, links, metadata});
    buffers.push_back(buffer(12, derived.size() * sizeof(glm::vec4), derived.data()));
    const GLuint acceleration = buffer(13, states.size() * sizeof(glm::vec4), nullptr);
    const GLuint collision = buffer(15, states.size() * sizeof(std::uint32_t), nullptr);
    buffers.push_back(acceleration);
    buffers.push_back(collision);
    const GLuint builder = program(shaderPath(shaderDirectory, "build_octree.comp"));
    buildTree(builder, count, nodeCapacity);
    const auto treeStatus = read<glm::uvec4>(metadata, 1)[0];
    require(treeStatus.y == 0 && treeStatus.x > 0 && treeStatus.x <= nodeCapacity,
            "Octree build failed");
    const auto aggregates = read<glm::vec4>(aggregate, treeStatus.x);
    glm::dvec3 weighted(0.0);
    double mass = 0.0;
    bool positive = true;
    bool hasObject = false;
    for (GLuint i = 0; i < count; ++i)
    {
        if (controls[i].initializing) continue;
        hasObject = true;
        positive = positive && physical[i].x > 0;
        mass += physical[i].x;
        weighted += double(physical[i].x) * glm::dvec3(states[i].position);
    }
    const auto nodes = read<Node>(topology, treeStatus.x);
    if (hasObject && positive && mass <= std::numeric_limits<float>::max())
    {
        require(nodes[0].data.z == 1u, "Root aggregate unexpectedly unusable");
        require(std::abs(double(aggregates[0].w) - mass) < mass * 1e-5,
                "Root mass mismatch");
        const double comError = glm::length(glm::dvec3(aggregates[0]) - weighted / mass);
        const double comTolerance = 1e-6 * std::max(1.0, double(nodes[0].centerHalf.w));
        require(comError < comTolerance,
                "Root center of mass mismatch: error=" + std::to_string(comError) +
                    ", tolerance=" + std::to_string(comTolerance));
    }
    else
    {
        require(nodes[0].data.z == 0u, "Invalid root aggregate was marked usable");
    }
    const auto next = read<glm::uvec2>(links, count);
    std::vector<unsigned> inbound(treeStatus.x, 0);
    for (GLuint parent = 0; parent < treeStatus.x; ++parent)
    {
        for (GLuint octant = 0; octant < 8; ++octant)
        {
            const GLuint child = octant < 4 ? nodes[parent].children0[octant]
                                            : nodes[parent].children1[octant - 4];
            if (child == 0xffffffffu) continue;
            require(child < treeStatus.x && nodes[child].data.y == nodes[parent].data.y + 1,
                    "Invalid octree child index or depth");
            ++inbound[child];
        }
    }
    require(inbound[0] == 0, "Root has an incoming edge");
    for (GLuint node = 1; node < treeStatus.x; ++node)
        require(inbound[node] == 1, "Octree node does not have exactly one parent");
    std::vector<unsigned> membership(count, 0);
    for (const Node &node : nodes)
    {
        if (node.data.y != depth) continue;
        std::uint32_t index = node.data.x;
        std::uint32_t previous = 0;
        for (GLuint iterations = 0; index != 0xffffffffu; ++iterations)
        {
            require(index < count && iterations < count, "Invalid tree leaf list");
            require(iterations == 0 || previous < index, "Leaf order is not deterministic");
            ++membership[index];
            previous = index;
            index = next[index].x;
        }
    }
    for (GLuint i = 0; i < count; ++i)
        require(membership[i] == (controls[i].initializing ? 0u : 1u),
                "Object tree membership mismatch");
    for (GLuint index = 0; index < treeStatus.x; ++index)
    {
        const Node &node = nodes[index];
        double expectedMass = 0.0;
        glm::dvec3 expectedWeighted(0.0);
        bool usable = true;
        bool hasMass = false;
        if (node.data.y == depth)
        {
            for (GLuint object = node.data.x; object != 0xffffffffu; object = next[object].x)
            {
                const double objectMass = physical[object].x;
                usable = usable && objectMass > 0;
                expectedMass += objectMass;
                expectedWeighted += objectMass * glm::dvec3(states[object].position);
                hasMass = true;
            }
        }
        else
        {
            for (GLuint octant = 0; octant < 8; ++octant)
            {
                const GLuint child = octant < 4 ? node.children0[octant]
                                                : node.children1[octant - 4];
                if (child == 0xffffffffu) continue;
                usable = usable && nodes[child].data.z != 0;
                const double childMass = aggregates[child].w;
                expectedMass += childMass;
                expectedWeighted += childMass * glm::dvec3(aggregates[child]);
                hasMass = true;
            }
        }
        usable = usable && hasMass && expectedMass > 0 &&
                 expectedMass <= std::numeric_limits<float>::max();
        require((node.data.z != 0) == usable, "Node aggregate validity mismatch");
        if (usable)
        {
            const glm::dvec3 expectedCom = expectedWeighted / expectedMass;
            const double massError = std::abs(double(aggregates[index].w) - expectedMass);
            const double comError = glm::length(glm::dvec3(aggregates[index]) - expectedCom);
            require(massError <= 1e-6 * expectedMass &&
                        comError <= 1e-6 * std::max(1.0, glm::length(expectedCom)),
                    "Node aggregate mass or center-of-mass mismatch");
        }
    }
    const GLuint nbody = program(shaderPath(shaderDirectory, "nbody.comp"));
    glUseProgram(nbody);
    glUniform1ui(glGetUniformLocation(nbody, "u_objectCount"), count);
    glUniform1d(glGetUniformLocation(nbody, "u_gravitationalConstant"), gravity);
    glUniform1f(glGetUniformLocation(nbody, "u_epsilon"), epsilon);
    glUniform1f(glGetUniformLocation(nbody, "u_theta"), 0.0f);
    glDispatchCompute((count + 63) / 64, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    const GLuint collisionProgram = program(shaderPath(shaderDirectory, "collision.comp"));
    glUseProgram(collisionProgram);
    glUniform1ui(glGetUniformLocation(collisionProgram, "u_objectCount"), count);
    glDispatchCompute((count + 63) / 64, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    const auto actualAcceleration = read<glm::vec4>(acceleration, count);
    const auto actualCollision = read<std::uint32_t>(collision, count);
    std::vector<glm::vec3> expectedAccelerations(count);
    for (GLuint i = 0; i < count; ++i)
    {
        glm::vec3 expected(0);
        std::uint32_t overlaps = 0;
        if (!controls[i].initializing)
        {
            for (GLuint j = 0; j < count; ++j)
            {
                if (i == j || controls[j].initializing)
                    continue;
                const glm::vec3 displacement = glm::vec3(states[j].position - states[i].position);
                const float distance = glm::length(displacement);
                if (distance <= 0.0f)
                    continue;
                const float meters = distance * 1000.0f;
                const float denominator = meters * meters + epsilon * epsilon;
                const double force = gravity * double(physical[i].x) * double(physical[j].x) /
                                     double(denominator);
                expected += displacement / distance * float(force / double(physical[i].x));
                if (derived[i].x + derived[j].x > distance)
                    ++overlaps;
            }
        }
        const double error = glm::length(glm::dvec3(actualAcceleration[i]) - glm::dvec3(expected));
        expectedAccelerations[i] = expected;
        if (std::isnan(expected.x) || std::isnan(expected.y) || std::isnan(expected.z))
        {
            for (int component = 0; component < 3; ++component)
                require(std::isnan(expected[component]) ==
                            std::isnan(actualAcceleration[i][component]),
                        "Zero-mass target changed the direct shader's NaN semantics");
        }
        else
        {
            require(error <= 1e-5 + 1e-4 * glm::length(glm::dvec3(expected)),
                    "Acceleration mismatch for scenario " + std::to_string(scenario) +
                        ", object " + std::to_string(i));
        }
        require(actualCollision[i] == overlaps,
                "Collision mismatch for object " + std::to_string(i));
    }
    if (scenario == 5 || scenario == 6 || scenario == 8 || scenario == 9)
    {
        glUseProgram(nbody);
        glUniform1f(glGetUniformLocation(nbody, "u_theta"), 0.5f);
        glDispatchCompute((count + 63) / 64, 1, 1);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        const auto approximation = read<glm::vec4>(acceleration, count);
        std::vector<double> relative;
        double sumSquaredAbsolute = 0.0;
        double sumSquaredRelative = 0.0;
        double maxAbsolute = 0.0;
        for (GLuint i = 0; i < count; ++i)
        {
            const double absolute = glm::length(glm::dvec3(approximation[i]) -
                                                glm::dvec3(expectedAccelerations[i]));
            const double reference = glm::length(glm::dvec3(expectedAccelerations[i]));
            const double relativeError = absolute / std::max(reference, 1e-30);
            relative.push_back(relativeError);
            sumSquaredRelative += relativeError * relativeError;
            sumSquaredAbsolute += absolute * absolute;
            maxAbsolute = std::max(maxAbsolute, absolute);
        }
        std::sort(relative.begin(), relative.end());
        std::cout << "theta=0.5 distribution=" << ((scenario == 5 || scenario == 8) ? "uniform" : "clustered")
                  << " N=" << count << " absolute_RMS="
                  << std::sqrt(sumSquaredAbsolute / count) << " absolute_max=" << maxAbsolute
                  << " relative_RMS=" << std::sqrt(sumSquaredRelative / count)
                  << " relative_p50=" << relative[count / 2]
                  << " relative_p95=" << relative[(count * 95) / 100]
                  << " relative_max=" << relative.back() << '\n';
        require(relative[(count * 95) / 100] < 0.05, "Theta error exceeds 5% at p95");

        const GLuint nextState = buffer(1, states.size() * sizeof(State), nullptr);
        buffers.push_back(nextState);
        const GLuint integration = program(shaderPath(shaderDirectory, "integrate.comp"));
        const auto directPath = std::filesystem::path(shaderDirectory).parent_path() /
                                "tests" / "direct_force.comp";
        const GLuint directProgram = program(directPath.string());
        glUseProgram(directProgram);
        glUniform1ui(glGetUniformLocation(directProgram, "u_objectCount"), count);
        glUniform1d(glGetUniformLocation(directProgram, "u_gravitationalConstant"), gravity);
        glUniform1f(glGetUniformLocation(directProgram, "u_epsilon"), epsilon);
        glUseProgram(integration);
        glUniform1ui(glGetUniformLocation(integration, "u_objectCount"), count);
        glUniform1ui(glGetUniformLocation(integration, "u_paused"), 0);
        glUniform1f(glGetUniformLocation(integration, "u_dt"), 1.0f / 96.0f);
        GLuint queries[9]{};
        glGenQueries(9, queries);
        std::array<double, 6> gpuMilliseconds{};
        double wallMilliseconds = 0.0;
        constexpr int warmup = 3;
        constexpr int iterations = 10;
        for (int step = 0; step < warmup + iterations; ++step)
        {
            glFinish();
            const auto start = std::chrono::steady_clock::now();
            glQueryCounter(queries[0], GL_TIMESTAMP);
            buildTree(builder, count, nodeCapacity, queries[6]);
            glQueryCounter(queries[1], GL_TIMESTAMP);
            const auto status = read<glm::uvec4>(metadata, 1)[0];
            require(status.y == 0u, "Benchmark octree build failed");
            glQueryCounter(queries[2], GL_TIMESTAMP);
            glUseProgram(nbody);
            glDispatchCompute((count + 63) / 64, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            glQueryCounter(queries[3], GL_TIMESTAMP);
            glUseProgram(collisionProgram);
            glDispatchCompute((count + 63) / 64, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            glQueryCounter(queries[4], GL_TIMESTAMP);
            glUseProgram(integration);
            glDispatchCompute((count + 63) / 64, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            glQueryCounter(queries[5], GL_TIMESTAMP);
            glFinish();
            const auto end = std::chrono::steady_clock::now();
            glQueryCounter(queries[7], GL_TIMESTAMP);
            glUseProgram(directProgram);
            glDispatchCompute((count + 63) / 64, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            glQueryCounter(queries[8], GL_TIMESTAMP);
            std::array<GLuint64, 9> time{};
            for (std::size_t i = 0; i < time.size(); ++i)
                glGetQueryObjectui64v(queries[i], GL_QUERY_RESULT, &time[i]);
            if (step >= warmup)
            {
                gpuMilliseconds[0] += double(time[6] - time[0]) / 1e6;
                gpuMilliseconds[1] += double(time[1] - time[6]) / 1e6;
                gpuMilliseconds[2] += double(time[3] - time[2]) / 1e6;
                gpuMilliseconds[3] += double(time[4] - time[3]) / 1e6;
                gpuMilliseconds[4] += double(time[5] - time[4]) / 1e6;
                gpuMilliseconds[5] += double(time[8] - time[7]) / 1e6;
                wallMilliseconds += std::chrono::duration<double, std::milli>(end - start).count();
            }
        }
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        const auto directGpu = read<glm::vec4>(acceleration, count);
        for (GLuint i = 0; i < count; ++i)
        {
            const double error = glm::length(glm::dvec3(directGpu[i]) -
                                             glm::dvec3(expectedAccelerations[i]));
            require(error <= 1e-5 + 1e-4 * glm::length(glm::dvec3(expectedAccelerations[i])),
                    "Test-only direct shader disagrees with CPU oracle");
        }
        for (int repeat = 0; repeat < 3; ++repeat)
        {
            buildTree(builder, count, nodeCapacity);
            glUseProgram(nbody);
            glDispatchCompute((count + 63) / 64, 1, 1);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            const auto repeated = read<glm::vec4>(acceleration, count);
            for (GLuint i = 0; i < count; ++i)
                require(repeated[i] == approximation[i],
                        "Tree force is not deterministic across rebuilds");
        }
        std::cout << "benchmark fixed_snapshot N=" << count
                  << " nodes=" << treeStatus.x << " node_capacity=" << nodeCapacity
                  << " build_group_ms=" << gpuMilliseconds[0] / iterations
                  << " aggregation_ms=" << gpuMilliseconds[1] / iterations
                  << " traversal_ms=" << gpuMilliseconds[2] / iterations
                  << " collision_ms=" << gpuMilliseconds[3] / iterations
                  << " integration_ms=" << gpuMilliseconds[4] / iterations
                  << " direct_force_ms=" << gpuMilliseconds[5] / iterations
                  << " wall_ms=" << wallMilliseconds / iterations << '\n';
        glDeleteQueries(9, queries);
        glDeleteProgram(integration);
        glDeleteProgram(directProgram);
    }
    if (scenario == 0)
    {
        const GLuint nextState = buffer(1, states.size() * sizeof(State), nullptr);
        buffers.push_back(nextState);
        const GLuint integration = program(shaderPath(shaderDirectory, "integrate.comp"));
        const Node sentinel{glm::vec4(123.0f), glm::uvec4(456), glm::uvec4(789),
                            glm::uvec4(1011)};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, topology);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Node), sizeof(Node), &sentinel);
        const glm::vec4 aggregateSentinel(123.0f);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, aggregate);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4), sizeof(glm::vec4),
                        &aggregateSentinel);
        buildTree(builder, count, 1);
        glUseProgram(nbody);
        glDispatchCompute((count + 63) / 64, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glUseProgram(collisionProgram);
        glDispatchCompute((count + 63) / 64, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glUseProgram(integration);
        glUniform1ui(glGetUniformLocation(integration, "u_objectCount"), count);
        glUniform1ui(glGetUniformLocation(integration, "u_paused"), 0);
        glUniform1f(glGetUniformLocation(integration, "u_dt"), 1.0f / 96.0f);
        glDispatchCompute((count + 63) / 64, 1, 1);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        require(read<glm::uvec4>(metadata, 1)[0].y == 1u, "Overflow was not reported");
        const Node untouched = read<Node>(topology, 2)[1];
        require(untouched.centerHalf == sentinel.centerHalf &&
                    untouched.children0 == sentinel.children0 &&
                    untouched.children1 == sentinel.children1 && untouched.data == sentinel.data &&
                    read<glm::vec4>(aggregate, 2)[1] == aggregateSentinel,
                "Overflow wrote beyond the declared node capacity");
        const auto guardedState = read<State>(nextState, count);
        const auto guardedAcceleration = read<glm::vec4>(acceleration, count);
        const auto guardedCollision = read<std::uint32_t>(collision, count);
        for (GLuint i = 0; i < count; ++i)
        {
            require(guardedState[i].position == states[i].position &&
                        guardedState[i].velocity == states[i].velocity,
                    "Integration advanced after overflow");
            require(guardedAcceleration[i] == glm::vec4(0) && guardedCollision[i] == 0,
                    "Force or collision ignored overflow");
        }
        std::vector<State> invalidStates = states;
        invalidStates[0].position.x = std::numeric_limits<float>::quiet_NaN();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        invalidStates.size() * sizeof(State), invalidStates.data());
        buildTree(builder, count, nodeCapacity);
        require(read<glm::uvec4>(metadata, 1)[0].y == 2u,
                "Non-finite position was not rejected");
        invalidStates = states;
        invalidStates[0].position.x = std::numeric_limits<float>::max();
        invalidStates[1].position.x = -std::numeric_limits<float>::max();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        invalidStates.size() * sizeof(State), invalidStates.data());
        buildTree(builder, count, nodeCapacity);
        require(read<glm::uvec4>(metadata, 1)[0].y == 2u,
                "Unrepresentable root bounds were not rejected");
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, states.size() * sizeof(State), states.data());
        auto invalidPhysical = physical;
        invalidPhysical[0].x = std::numeric_limits<float>::infinity();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        invalidPhysical.size() * sizeof(glm::vec4), invalidPhysical.data());
        buildTree(builder, count, nodeCapacity);
        require(read<glm::uvec4>(metadata, 1)[0].y == 2u,
                "Non-finite mass was not rejected");
        glDeleteProgram(integration);
    }
    glDeleteProgram(nbody);
    glDeleteProgram(builder);
    glDeleteProgram(collisionProgram);
    glDeleteBuffers(static_cast<GLsizei>(buffers.size()), buffers.data());
}

void compareTrajectory(const std::string &shaderDirectory, const std::string &baselinePath,
                       float theta)
{
    constexpr GLuint count = 202;
    constexpr int steps = 960;
    constexpr int interval = 96;
    constexpr double dt = 1.0 / 96.0;
    std::vector<State> states;
    std::vector<glm::vec4> physical, derived;
    std::vector<Control> controls;
    const auto append = [&](glm::vec3 position, glm::vec3 velocity, float mass, float density)
    {
        states.push_back({glm::vec4(position, 0), glm::vec4(velocity, 0)});
        physical.emplace_back(mass, density, 0, 0);
        const float radius = static_cast<float>(
            std::pow((3 * mass / density) / (4 * 3.14159265359), 1.0f / 3.0f) / 100000);
        derived.emplace_back(radius, 0, 0, 0);
        controls.push_back({0, 0, 0, static_cast<std::uint32_t>(states.size())});
    };
    append({3844, 0, 0}, {0, 0, 228}, 7.34767309e22f, 3344);
    append({0, 0, 0}, {0, 0, 0}, 5.97219e24f, 5515);
    std::mt19937 generator(123);
    std::uniform_real_distribution<float> radiusDist(4000.0f, 12000.0f);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * glm::pi<float>());
    std::uniform_real_distribution<float> massDist(1.0e20f, 3.0e23f);
    std::uniform_real_distribution<float> speedDist(120.0f, 260.0f);
    std::uniform_real_distribution<float> tiltDist(-0.15f, 0.15f);
    for (GLuint i = 0; i < 200; ++i)
    {
        const float radius = radiusDist(generator);
        const float angle = angleDist(generator);
        const float tilt = tiltDist(generator);
        const float speed = speedDist(generator);
        const glm::vec3 position(radius * std::cos(angle), radius * tilt,
                                 radius * std::sin(angle));
        const glm::vec3 velocity(-std::sin(angle) * speed, 0, std::cos(angle) * speed);
        append(position, velocity, massDist(generator), 3344);
    }

    std::ifstream input(baselinePath);
    require(input.good(), "Cannot read baseline CSV: " + baselinePath);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)) &&
                line == "time,step,object_id,x,y,z",
            "Unexpected baseline CSV header");
    std::array<std::vector<glm::vec3>, 11> baseline;
    std::array<std::vector<bool>, 11> seen;
    for (std::size_t sample = 0; sample < baseline.size(); ++sample)
    {
        baseline[sample].resize(count);
        seen[sample].resize(count, false);
    }
    while (std::getline(input, line))
    {
        std::istringstream row(line);
        std::string timeText, stepText, idText, xText, yText, zText;
        require(bool(std::getline(row, timeText, ',')) && bool(std::getline(row, stepText, ',')) &&
                    bool(std::getline(row, idText, ',')) && bool(std::getline(row, xText, ',')) &&
                    bool(std::getline(row, yText, ',')) && bool(std::getline(row, zText, ',')),
                "Malformed baseline CSV row");
        const int step = std::stoi(stepText);
        const unsigned id = std::stoul(idText);
        const int sample = step == 0 ? 0 : (step + 1) % interval == 0 ? (step + 1) / interval : -1;
        if (sample < 0 || sample >= static_cast<int>(baseline.size())) continue;
        require(id < count && !seen[sample][id], "Unexpected or duplicate baseline object");
        baseline[sample][id] = glm::vec3(std::stof(xText), std::stof(yText), std::stof(zText));
        seen[sample][id] = true;
    }
    for (const auto &sample : seen)
        require(std::all_of(sample.begin(), sample.end(), [](bool found) { return found; }),
                "Baseline CSV does not contain all expected samples");

    auto reference = states;
    for (int step = 1; step <= steps; ++step)
    {
        for (GLuint i = 0; i < count; ++i)
        {
            glm::vec3 position(reference[i].position);
            glm::vec3 velocity(reference[i].velocity);
            for (GLuint j = 0; j < count; ++j)
            {
                if (i == j) continue;
                const glm::vec3 displacement = glm::vec3(reference[j].position) - position;
                const float squared = displacement.x * displacement.x +
                                      displacement.y * displacement.y +
                                      displacement.z * displacement.z;
                const float distance = std::sqrt(squared);
                if (distance <= 0.0f) continue;
                const float meters = distance * 1000.0f;
                const double force = gravity * double(physical[i].x) * double(physical[j].x) /
                                     double(meters * meters);
                const glm::vec3 acceleration = displacement / distance *
                                                float(force / double(physical[i].x));
                for (int component = 0; component < 3; ++component)
                    velocity[component] = static_cast<float>(
                        double(velocity[component]) + double(acceleration[component]) * dt);
                if (derived[i].x + derived[j].x > std::pow(squared, 1.0f / 2.0f))
                    velocity *= -0.2f;
            }
            for (int component = 0; component < 3; ++component)
                position[component] = static_cast<float>(
                    double(position[component]) + double(velocity[component]) * dt);
            reference[i].position = glm::vec4(position, 0);
            reference[i].velocity = glm::vec4(velocity, 0);
        }
        if (step != 1 && step % interval != 0) continue;
        const int sample = step == 1 ? 0 : step / interval;
        double maximum = 0.0;
        for (GLuint i = 0; i < count; ++i)
            maximum = std::max(maximum, glm::length(glm::dvec3(reference[i].position) -
                                                    glm::dvec3(baseline[sample][i])));
        std::cout << "existing_baseline reconstructed_step=" << step
                  << " max_position_error=" << maximum << '\n';
        if (step == 1)
            require(maximum < 0.02, "Existing baseline does not match reconstructed first step");
    }

    const GLuint capacity = 1 + depth * count;
    std::vector<GLuint> buffers;
    GLuint current = buffer(0, count * sizeof(State), states.data());
    GLuint next = buffer(1, count * sizeof(State), states.data());
    buffers.push_back(current);
    buffers.push_back(next);
    buffers.push_back(buffer(2, count * sizeof(glm::vec4), physical.data()));
    buffers.push_back(buffer(3, count * sizeof(Control), controls.data()));
    buffers.push_back(buffer(6, capacity * sizeof(glm::vec4), nullptr));
    buffers.push_back(buffer(7, capacity * sizeof(Node), nullptr));
    buffers.push_back(buffer(8, count * sizeof(glm::uvec2), nullptr));
    const GLuint metadata = buffer(9, sizeof(glm::uvec4), nullptr);
    buffers.push_back(metadata);
    buffers.push_back(buffer(12, count * sizeof(glm::vec4), derived.data()));
    buffers.push_back(buffer(13, count * sizeof(glm::vec4), nullptr));
    buffers.push_back(buffer(15, count * sizeof(std::uint32_t), nullptr));
    const GLuint builder = program(shaderPath(shaderDirectory, "build_octree.comp"));
    const GLuint nbody = program(shaderPath(shaderDirectory, "nbody.comp"));
    const GLuint collision = program(shaderPath(shaderDirectory, "collision.comp"));
    const GLuint integration = program(shaderPath(shaderDirectory, "integrate.comp"));
    glUseProgram(nbody);
    glUniform1ui(glGetUniformLocation(nbody, "u_objectCount"), count);
    glUniform1d(glGetUniformLocation(nbody, "u_gravitationalConstant"), gravity);
    glUniform1f(glGetUniformLocation(nbody, "u_epsilon"), epsilon);
    glUniform1f(glGetUniformLocation(nbody, "u_theta"), theta);
    glUseProgram(collision);
    glUniform1ui(glGetUniformLocation(collision, "u_objectCount"), count);
    glUseProgram(integration);
    glUniform1ui(glGetUniformLocation(integration, "u_objectCount"), count);
    glUniform1ui(glGetUniformLocation(integration, "u_paused"), 0);
    glUniform1f(glGetUniformLocation(integration, "u_dt"), static_cast<float>(dt));
    const GLuint groups = (count + 63) / 64;
    for (int step = 1; step <= steps; ++step)
    {
        buildTree(builder, count, capacity);
        require(read<glm::uvec4>(metadata, 1)[0].y == 0u, "Trajectory tree build failed");
        glUseProgram(nbody);
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glUseProgram(collision);
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glUseProgram(integration);
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        std::swap(current, next);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, current);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, next);
        if (step % interval != 0) continue;
        const auto actual = read<State>(current, count);
        double sumSquared = 0.0;
        double maxError = 0.0;
        for (GLuint i = 0; i < count; ++i)
        {
            const double error = glm::length(glm::dvec3(actual[i].position) -
                                             glm::dvec3(baseline[step / interval][i]));
            sumSquared += error * error;
            maxError = std::max(maxError, error);
        }
        std::cout << "existing_baseline theta=" << theta << " step=" << step
                  << " RMS_position_error="
                  << std::sqrt(sumSquared / count) << " max_position_error=" << maxError << '\n';
    }
    glDeleteProgram(builder);
    glDeleteProgram(nbody);
    glDeleteProgram(collision);
    glDeleteProgram(integration);
    glDeleteBuffers(static_cast<GLsizei>(buffers.size()), buffers.data());
}
}

int main(int argc, char **argv)
{
    try
    {
        require(argc == 2 || argc == 3, "Expected shader directory and optional baseline CSV");
        require(glfwInit() == GLFW_TRUE, "GLFW initialization failed");
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        GLFWwindow *window = glfwCreateWindow(64, 64, "mass aggregation test", nullptr, nullptr);
        require(window != nullptr, "OpenGL 4.4 context creation failed");
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        require(glewInit() == GLEW_OK, "GLEW initialization failed");
        std::cout << "OpenGL: " << glGetString(GL_VERSION) << ", "
                  << glGetString(GL_RENDERER) << '\n';
        GLint bindings = 0, blocks = 0;
        GLint64 maxBlock = 0;
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &bindings);
        glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &blocks);
        glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlock);
        std::cout << "SSBO limits bindings=" << bindings << " compute_blocks=" << blocks
                  << " block_bytes=" << maxBlock << '\n';
        for (int scenario = 0; scenario <= 11; ++scenario)
            testDirect(argv[1], scenario);
        if (argc == 3)
        {
            compareTrajectory(argv[1], argv[2], 0.0f);
            compareTrajectory(argv[1], argv[2], 0.5f);
        }
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cout << "Direct force and collision contract passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
