#pragma once

#include "app/camera_controllers.hpp"
#include "gravity_sim_3Dgrid_function.hpp"
#include "models/simulation_module.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace grav_models
{

class GravityGpuModule final : public grav_sim::ISimulationModule
{
  public:
    GravityGpuModule();
    ~GravityGpuModule() override;

    grav_sim::ModeId Id() const override { return grav_sim::ModeId::F1_GravityGpu; }
    const std::vector<grav_sim::PresetDescriptor> &Presets() const override { return presets_; }
    std::size_t ActivePresetIndex() const override { return activePresetIndex_; }
    const grav_sim::PresetDescriptor &ActivePreset() const override
    {
        return presets_[activePresetIndex_];
    }
    const grav_sim::SolverDescriptor &ActiveSolver() const override { return solver_; }

    void OnEnter(const grav_sim::ModuleServices &services, std::size_t presetIndex) override;
    void HandleInput(const grav_sim::ModuleInputState &input, double wallDeltaSeconds) override;
    void Update(const grav_sim::StepContext &step) override;
    void Render(const grav_sim::RenderContext &render) override;
    void OnResize(int framebufferWidth, int framebufferHeight) override;
    void ResetCamera() override;
    std::uint64_t CameraRevision() const override { return camera_.Revision(); }

    void BeginExit() override;
    void ReleaseResources() override;

    grav_sim::ModuleRuntimeStats RuntimeStats() const override;
    void DrawInteractiveControls(bool configLocked) override;
    std::string ControlsHelpText() const override;

    void SetCustomCsvPath(const std::string &csvPath) { customCsvPath_ = csvPath; }
    const std::string &CustomCsvPath() const { return customCsvPath_; }
    void SetTrajectoryExportEnabled(bool enabled) { enableTrajectoryExport_ = enabled; }
    bool IsTrajectoryExportEnabled() const { return enableTrajectoryExport_; }
    void SetTrajectoryExportPath(const std::string &path) { trajectoryExportPath_ = path; }
    void SetTrajectorySampleIntervalSteps(int interval) { trajectorySampleIntervalSteps_ = interval; }
    std::size_t RecordedTrajectorySampleCount() const { return trajectorySamples_.size(); }

  private:
    void LoadPresetObjects(std::size_t presetIndex);

    grav_sim::ModuleServices services_{};
    std::vector<grav_sim::PresetDescriptor> presets_;
    std::size_t activePresetIndex_ = 0;
    grav_sim::SolverDescriptor solver_;

    grav_app::FreeFlyCamera camera_;
    bool cameraInitialized_ = false;

    GLuint objectRenderProgram_ = 0;
    GLuint nbodyComputeProgram_ = 0;
    GLuint integrateComputeProgram_ = 0;
    ShaderUniformLocations renderUniforms_{};
    ComputeUniformLocations computeUniforms_{};

    std::uint64_t sphereVaoCensusId_ = 0;
    std::uint64_t sphereVboCensusId_ = 0;

    std::size_t objectCount_ = 0;
    float softeningEpsilon_ = 10.0f;
    float spawnMassKg_ = 1.0e20f;
    std::uint64_t stepCount_ = 0;
    bool holdPauseActive_ = false;
    bool initializingObjectDirty_ = false;

    std::string customCsvPath_ = "shaders/presets/sample_four_body.csv";
    char csvPathBuffer_[512] = "shaders/presets/sample_four_body.csv";
    std::string lastStatusMessage_;

    bool enableTrajectoryExport_ = false;
    int trajectorySampleIntervalSteps_ = 96;
    std::string trajectoryExportPath_ = "numerical_error/f1_trajectory_v2.csv";
    std::vector<TrajectorySampleV2> trajectorySamples_;
    bool resourcesActive_ = false;
};

} // namespace grav_models
