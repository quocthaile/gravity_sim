#pragma once

#include "app/camera_controllers.hpp"
#include "models/black_hole/black_hole_core.hpp"
#include "models/simulation_module.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace grav_models
{

class BlackHoleCpuModule final : public grav_sim::ISimulationModule
{
  public:
    BlackHoleCpuModule();
    ~BlackHoleCpuModule() override;

    grav_sim::ModeId Id() const override { return grav_sim::ModeId::F5_BlackHoleCpu; }
    const std::vector<grav_sim::PresetDescriptor> &Presets() const override { return presets_; }
    std::size_t ActivePresetIndex() const override { return activePresetIndex_; }
    const grav_sim::PresetDescriptor &ActivePreset() const override
    {
        return presets_[activePresetIndex_];
    }
    const grav_sim::SolverDescriptor &ActiveSolver() const override
    {
        return solvers_[activeSolverIndex_];
    }
    std::vector<grav_sim::SolverDescriptor> AvailableSolvers() const override { return solvers_; }
    bool SetActiveSolverById(const std::string &solverId) override;

    void OnEnter(const grav_sim::ModuleServices &services, std::size_t presetIndex) override;
    void HandleInput(const grav_sim::ModuleInputState &input, double wallDeltaSeconds) override;
    void Update(const grav_sim::StepContext &step) override;
    void Render(const grav_sim::RenderContext &render) override;
    void OnResize(int framebufferWidth, int framebufferHeight) override;
    void ResetCamera() override;
    std::uint64_t CameraRevision() const override { return camera_.Revision(); }

    void BeginExit() override;
    bool IsReadyToRelease() const override { return worker_.IsIdle(); }
    void ReleaseResources() override;

    grav_sim::ModuleRuntimeStats RuntimeStats() const override;
    void DrawInteractiveControls(bool configLocked) override;
    std::string ControlsHelpText() const override;

    void SetRenderResolution(int width, int height);
    bool HasValidUploadedImage() const { return hasUploadedImage_; }

  private:
    void ApplyPresetDefaults(std::size_t presetIndex);
    void EnsureTexture(int width, int height);
    void SyncSharedBlackHoleAndGrid();
    void SubmitCpuGeodesicJob(std::uint64_t sceneGeneration);

    grav_sim::ModuleServices services_{};
    std::vector<grav_sim::PresetDescriptor> presets_;
    std::size_t activePresetIndex_ = 0;

    std::vector<grav_sim::SolverDescriptor> solvers_;
    std::size_t activeSolverIndex_ = 0;

    grav_app::OrbitPanCamera camera_;
    bool cameraConfigured_ = false;

    CpuRayWorker worker_;
    std::uint64_t nextJobId_ = 1;

    GLuint texture_ = 0;
    int texW_ = 0;
    int texH_ = 0;
    std::uint64_t texCensusId_ = 0;

    int renderWidth_ = 200;
    int renderHeight_ = 150;
    int maxSteps_ = 12000;
    float dLambdaMeters_ = 1.0e7f;
    bool diskEnabled_ = true;
    bool redSilhouetteStyle_ = false;
    std::vector<BlackHoleSceneObject> sceneObjects_;

    std::uint64_t lastSubmittedCameraRev_ = 0;
    std::uint64_t lastBhRevision_ = 0;
    bool imageDirty_ = true;
    bool hasUploadedImage_ = false;
    double imageAgeMs_ = 0.0;
    std::uint64_t completedImages_ = 0;
    bool resourcesActive_ = false;
};

} // namespace grav_models

