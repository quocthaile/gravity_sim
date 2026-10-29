#include "models/diagnostic/diagnostic_triangle_module.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

namespace grav_models
{

namespace
{

const char *kDiagTriangleVert = R"glsl(#version 440 core
layout(location = 0) in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos, 1.0);
}
)glsl";

const char *kDiagTriangleFrag = R"glsl(#version 440 core
uniform vec4 uColor;
out vec4 FragColor;

void main()
{
    FragColor = uColor;
}
)glsl";

} // namespace

DiagnosticTriangleModule::DiagnosticTriangleModule()
{
    presets_ = {
        {
            "diag_orange_triangle_3d_test",
            "Core-Profile Orange Triangle (3D_test.cpp)",
            "src/legacy/3D_test.cpp:14-60",
            "Minimal OpenGL 4.4 Core graphics pipeline & context sanity diagnostic.",
            1,
        },
    };

    solver_.id = "diagnostic_ndc_triangle_v1";
    solver_.displayName = "NDC Static Triangle Rasterizer";
    solver_.backendName = "OpenGL 4.4 Core VAO/VBO";
    solver_.status = grav_sim::SolverValidationStatus::DiagnosticOnly;
    solver_.notes =
        "Verifies GL context, VAO/VBO binding, shader compilation, and swap-chain health.";
}

DiagnosticTriangleModule::~DiagnosticTriangleModule() { ReleaseResources(); }

void DiagnosticTriangleModule::OnEnter(const grav_sim::ModuleServices &services,
                                       std::size_t presetIndex)
{
    services_ = services;
    activePresetIndex_ = presetIndex < presets_.size() ? presetIndex : 0;

    try
    {
        program_ = services_.shaderCache->GetOrCreateGraphicsFromSource(
            "diag:orange_triangle", kDiagTriangleVert, kDiagTriangleFrag);
        locTint_ = glGetUniformLocation(program_, "uColor");

        constexpr float kVertices[9] = {
            -0.5f, -0.5f, 0.0f, // Bottom left
            0.5f,  -0.5f, 0.0f, // Bottom right
            0.0f,  0.5f,  0.0f, // Top center
        };

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        if (services_.resourceCensus != nullptr)
        {
            vaoCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::VertexArray, vao_, 0, 0,
                "scene:diagnostic_triangle", "DiagTriangleVAO", services_.sceneGeneration);
            vboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::GeometryBuffer, vbo_, sizeof(kVertices),
                sizeof(kVertices), "scene:diagnostic_triangle", "DiagTriangleVBO",
                services_.sceneGeneration);
        }

        // Configure flat zero-source grid in case user manually toggles G while in diagnostic mode.
        if (services_.sharedGrid != nullptr)
        {
            services_.sharedGrid->Configure(grav_render::GridConfig::GravityCompatibility());
            services_.sharedGrid->SetCpuSources({}, 1);
        }

        triangleColor_ = glm::vec4(1.0f, 0.5f, 0.2f, 1.0f);
        frameCounter_ = 0;
        resourcesActive_ = true;
    }
    catch (...)
    {
        ReleaseResources();
        throw;
    }
}

void DiagnosticTriangleModule::HandleInput(const grav_sim::ModuleInputState &input,
                                           double wallDeltaSeconds)
{
    (void)input;
    (void)wallDeltaSeconds;
}

void DiagnosticTriangleModule::Update(const grav_sim::StepContext &step)
{
    if (!step.paused || step.singleStepRequested)
    {
        ++frameCounter_;
    }
}

void DiagnosticTriangleModule::Render(const grav_sim::RenderContext &render)
{
    if (!resourcesActive_ || render.framebufferWidth <= 0 || render.framebufferHeight <= 0)
    {
        return;
    }

    if (services_.sharedGrid != nullptr && render.gridVisible && services_.sharedGrid->IsDirty())
    {
        grav_telemetry::ScopedGpuPassQuery deformQuery(
            services_.telemetry, grav_telemetry::GpuPassId::GridDeform, render.sceneGeneration);
        services_.sharedGrid->UpdateIfDirty(true);
    }

    grav_telemetry::ScopedGpuPassQuery drawQuery(
        services_.telemetry, grav_telemetry::GpuPassId::SceneDraw, render.sceneGeneration);

    if (services_.sharedGrid != nullptr && render.gridVisible)
    {
        const float aspect = static_cast<float>(render.framebufferWidth) /
                             static_cast<float>(std::max(render.framebufferHeight, 1));
        const glm::mat4 view =
            glm::lookAt(glm::vec3(0.0f, 2500.0f, 9000.0f), glm::vec3(0.0f, -800.0f, 0.0f),
                        glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 1.0f, 150000.0f);
        services_.sharedGrid->Draw(view, proj, true, true);
    }

    glDisable(GL_DEPTH_TEST);
    glUseProgram(program_);
    glUniform4fv(locTint_, 1, glm::value_ptr(triangleColor_));
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

void DiagnosticTriangleModule::OnResize(int framebufferWidth, int framebufferHeight)
{
    services_.framebufferWidth = framebufferWidth;
    services_.framebufferHeight = framebufferHeight;
}

void DiagnosticTriangleModule::ResetCamera()
{
    triangleColor_ = glm::vec4(1.0f, 0.5f, 0.2f, 1.0f);
    ++cameraRevision_;
}

void DiagnosticTriangleModule::ReleaseResources()
{
    if (services_.resourceCensus != nullptr)
    {
        if (vaoCensusId_ != 0)
        {
            services_.resourceCensus->Unregister(vaoCensusId_);
        }
        if (vboCensusId_ != 0)
        {
            services_.resourceCensus->Unregister(vboCensusId_);
        }
    }

    if (vao_ != 0)
    {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (vbo_ != 0)
    {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    resourcesActive_ = false;
}

grav_sim::ModuleRuntimeStats DiagnosticTriangleModule::RuntimeStats() const
{
    grav_sim::ModuleRuntimeStats stats{};
    stats.primaryCount = 3;
    stats.primaryLabel = "Vertices";
    stats.outputWidth = services_.framebufferWidth;
    stats.outputHeight = services_.framebufferHeight;
    stats.stepCount = frameCounter_;
    stats.simulationOrAffineParam = static_cast<double>(frameCounter_);
    stats.paramLabel = "Frames";
    stats.gridOverlayDiagnosticOnly = true;
    return stats;
}

void DiagnosticTriangleModule::DrawInteractiveControls(bool configLocked)
{
    ImGui::TextWrapped("Graphics Context Diagnostic adapted from src/legacy/3D_test.cpp.");
    ImGui::BeginDisabled(configLocked);
    ImGui::ColorEdit4("Triangle Color", glm::value_ptr(triangleColor_));
    ImGui::EndDisabled();
}

std::string DiagnosticTriangleModule::ControlsHelpText() const
{
    return "Ctrl+F12: Diagnostic Triangle | C: Reset orange color | G: Toggle optional flat grid | "
           "F1..F6: Return to simulation modes";
}

} // namespace grav_models

