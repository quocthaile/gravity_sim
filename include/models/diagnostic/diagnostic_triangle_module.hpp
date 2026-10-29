#pragma once

#include "models/simulation_module.hpp"
#include <GL/glew.h>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace grav_models
{

class DiagnosticTriangleModule final : public grav_sim::ISimulationModule
{
  public:
    DiagnosticTriangleModule();
    ~DiagnosticTriangleModule() override;

    grav_sim::ModeId Id() const override { return grav_sim::ModeId::Diagnostic_Triangle; }
    const std::vector<grav_sim::PresetDescriptor> &Presets() const override { return presets_; }
    std::size_t ActivePresetIndex() const override { return activePresetIndex_; }
    const grav_sim::PresetDescriptor &ActivePreset() const override
    {
        return presets_[activePresetIndex_];
    }
    const grav_sim::SolverDescriptor &ActiveSolver() const override { return solver_; }

    bool DefaultGridVisible() const override { return false; }

    void OnEnter(const grav_sim::ModuleServices &services, std::size_t presetIndex) override;
    void HandleInput(const grav_sim::ModuleInputState &input, double wallDeltaSeconds) override;
    void Update(const grav_sim::StepContext &step) override;
    void Render(const grav_sim::RenderContext &render) override;
    void OnResize(int framebufferWidth, int framebufferHeight) override;
    void ResetCamera() override;
    std::uint64_t CameraRevision() const override { return cameraRevision_; }

    void BeginExit() override {}
    void ReleaseResources() override;

    grav_sim::ModuleRuntimeStats RuntimeStats() const override;
    void DrawInteractiveControls(bool configLocked) override;
    std::string ControlsHelpText() const override;

  private:
    grav_sim::ModuleServices services_{};
    std::vector<grav_sim::PresetDescriptor> presets_;
    std::size_t activePresetIndex_ = 0;
    grav_sim::SolverDescriptor solver_;

    GLuint program_ = 0;
    GLint locTint_ = -1;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    std::uint64_t vaoCensusId_ = 0;
    std::uint64_t vboCensusId_ = 0;

    glm::vec4 triangleColor_{1.0f, 0.5f, 0.2f, 1.0f};
    std::uint64_t frameCounter_ = 0;
    std::uint64_t cameraRevision_ = 1;
    bool resourcesActive_ = false;
};

} // namespace grav_models

