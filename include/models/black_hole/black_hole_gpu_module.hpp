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

class BlackHoleGpuModule final : public grav_sim::ISimulationModule
{
  public:
    BlackHoleGpuModule();
    ~BlackHoleGpuModule() override;

    grav_sim::ModeId Id() const override { return grav_sim::ModeId::F3_BlackHoleGpu; }
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
    void ReleaseResources() override;

    grav_sim::ModuleRuntimeStats RuntimeStats() const override;
    void DrawInteractiveControls(bool configLocked) override;
    std::string ControlsHelpText() const override;

    // Inspection & parity-test hooks (T08, T13)
    GLuint OutputTextureHandle() const { return outputTexture_; }
    int ActiveComputeWidth() const { return allocatedTexW_; }
    int ActiveComputeHeight() const { return allocatedTexH_; }
    void SetComputeResolution(int width, int height);
    BlackHoleSceneConfig BuildCurrentSceneConfig(int outW, int outH) const;

  private:
    void ConfigurePresetScene(std::size_t presetIndex);
    void EnsureOutputTexture(int width, int height);
    void SyncSharedBlackHoleAndGrid();
    void StepLegacyExperimentalObjectGravity();
    void DispatchRayCompute(int targetW, int targetH, bool movingFlag, std::uint64_t sceneGen);

    grav_sim::ModuleServices services_{};
    std::vector<grav_sim::PresetDescriptor> presets_;
    std::size_t activePresetIndex_ = 0;

    std::vector<grav_sim::SolverDescriptor> solvers_;
    std::size_t activeSolverIndex_ = 0;

    grav_app::OrbitPanCamera camera_;
    bool cameraConfigured_ = false;

    GLuint computeProgramCanonical_ = 0;
    GLuint computeProgramLegacy_ = 0;

    GLuint outputTexture_ = 0;
    int allocatedTexW_ = 0;
    int allocatedTexH_ = 0;
    std::uint64_t texCensusId_ = 0;

    GLuint cameraUbo_ = 0;
    GLuint diskUbo_ = 0;
    GLuint objectsUbo_ = 0;
    std::uint64_t camUboCensusId_ = 0;
    std::uint64_t diskUboCensusId_ = 0;
    std::uint64_t objUboCensusId_ = 0;

    int baseComputeWidth_ = 360;
    int baseComputeHeight_ = 270;
    bool enableMovingLod_ = true;

    bool diskEnabled_ = true;
    bool legacyGravityMotionEnabled_ = false;
    float dLambdaMeters_ = 1.0e7f;
    int maxStepsStatic_ = 30000;
    int maxStepsMoving_ = 12000;
    std::vector<BlackHoleSceneObject> sceneObjects_;

    std::uint64_t lastCameraRevision_ = 0;
    std::uint64_t lastBhRevision_ = 0;
    bool imageDirty_ = true;
    double imageAgeMs_ = 0.0;
    std::uint64_t dispatchCount_ = 0;
    bool resourcesActive_ = false;
};

} // namespace grav_models

