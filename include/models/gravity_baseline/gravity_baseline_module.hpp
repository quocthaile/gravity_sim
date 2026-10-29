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

// Standalone historical CPU baseline physics step matching src/gravity_sim_3Dgrid_baseline.cpp:423-470:
// - Sequential in-place outer/inner pair loop (body i updates position before body i+1 evaluates forces)
// - Skips bodies with initializing == true in force & collision evaluation
// - Distance converted to meters via distance * 1000.0f with NO softening epsilon
// - Collision velocity multiplier (-0.2f on overlap) applied per neighbor inside the inner loop
void StepBaselineBodiesInPlace(std::vector<Object> &bodies, double stepDt = 1.0 / 96.0);

class GravityBaselineModule final : public grav_sim::ISimulationModule
{
  public:
    GravityBaselineModule();
    ~GravityBaselineModule() override;

    grav_sim::ModeId Id() const override { return grav_sim::ModeId::F6_GravityBaselineCpu; }
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
    const std::vector<Object> &Bodies() const { return bodies_; }

  private:
    struct CpuSphereInstance
    {
        glm::vec4 posRadius{0.0f};
        glm::vec4 color{1.0f};
    };

    void LoadPresetBodies(std::size_t presetIndex);
    void SyncGridAndInstances();
    void EnsureInstanceCapacity(std::size_t requiredCount);
    void RecordCpuTrajectorySample(std::uint64_t sceneGeneration);

    grav_sim::ModuleServices services_{};
    std::vector<grav_sim::PresetDescriptor> presets_;
    std::size_t activePresetIndex_ = 0;
    grav_sim::SolverDescriptor solver_;

    grav_app::FreeFlyCamera camera_;
    bool cameraInitialized_ = false;

    std::vector<Object> bodies_;
    std::vector<grav_render::GridSource> gridSources_;
    std::vector<CpuSphereInstance> instanceScratch_;

    GLuint instancedProgram_ = 0;
    GLint locView_ = -1;
    GLint locProjection_ = -1;

    GLuint instanceVao_ = 0;
    GLuint instanceVbo_ = 0;
    std::size_t instanceCapacity_ = 0;

    std::uint64_t sphereVaoCensusId_ = 0;
    std::uint64_t sphereVboCensusId_ = 0;
    std::uint64_t instanceVaoCensusId_ = 0;
    std::uint64_t instanceVboCensusId_ = 0;

    float spawnMassKg_ = 1.0e20f;
    std::uint64_t stepCount_ = 0;
    std::uint64_t sourceRevision_ = 1;
    bool holdPauseActive_ = false;
    bool stateDirty_ = false;

    std::string customCsvPath_ = "shaders/presets/sample_four_body.csv";
    char csvPathBuffer_[512] = "shaders/presets/sample_four_body.csv";
    std::string lastStatusMessage_;

    bool enableTrajectoryExport_ = false;
    int trajectorySampleIntervalSteps_ = 96;
    std::string trajectoryExportPath_ = "numerical_error/f6_baseline_trajectory_v2.csv";
    std::vector<TrajectorySampleV2> trajectorySamples_;

    bool resourcesActive_ = false;
};

} // namespace grav_models

