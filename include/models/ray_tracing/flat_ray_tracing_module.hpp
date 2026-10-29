#pragma once

#include "app/camera_controllers.hpp"
#include "models/black_hole/black_hole_core.hpp"
#include "models/simulation_module.hpp"
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace grav_models
{

class FlatRayTracingModule final : public grav_sim::ISimulationModule
{
  public:
    FlatRayTracingModule();
    ~FlatRayTracingModule() override;

    grav_sim::ModeId Id() const override { return grav_sim::ModeId::F4_FlatRayTracingCpu; }
    const std::vector<grav_sim::PresetDescriptor> &Presets() const override { return presets_; }
    std::size_t ActivePresetIndex() const override { return activePresetIndex_; }
    const grav_sim::PresetDescriptor &ActivePreset() const override
    {
        return presets_[activePresetIndex_];
    }
    const grav_sim::SolverDescriptor &ActiveSolver() const override { return activeSolver_; }
    bool DefaultGridVisible() const override { return false; }

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
    struct FlatSphere
    {
        glm::vec3 center{0.0f};
        float radius = 1.0f;
        glm::vec3 color{1.0f};
        float reflectivity = 0.0f;
        float emission = 0.0f;
        glm::vec3 velocity{0.0f};
    };

    void LoadScenePreset(std::size_t presetIndex);
    void EnsureTexture(int width, int height);
    void SubmitTraceJob(std::uint64_t sceneGeneration);
    static glm::vec3 TraceFlatSceneRay(const glm::vec3 &origin, const glm::vec3 &dir,
                                       const std::vector<FlatSphere> &spheres,
                                       const std::vector<glm::vec3> &lights,
                                       const glm::vec3 &bgColor, bool recursiveMode, int depth);

    grav_sim::ModuleServices services_{};
    std::vector<grav_sim::PresetDescriptor> presets_;
    std::size_t activePresetIndex_ = 0;
    grav_sim::SolverDescriptor activeSolver_;

    grav_app::FreeFlyCamera camera_;
    bool cameraConfigured_ = false;

    CpuRayWorker worker_;
    std::uint64_t nextJobId_ = 1;

    GLuint texture_ = 0;
    int texW_ = 0;
    int texH_ = 0;
    std::uint64_t texCensusId_ = 0;

    int renderWidth_ = 320;
    int renderHeight_ = 240;

    std::vector<FlatSphere> spheres_;
    std::vector<glm::vec3> pointLights_;
    glm::vec3 backgroundColor_{0.0f, 0.0f, 0.1f};
    bool recursiveShadingMode_ = false;
    bool constantGravityBounceMode_ = false;

    std::uint64_t lastSubmittedCameraRev_ = 0;
    bool sceneDirty_ = true;
    bool hasUploadedImage_ = false;
    double imageAgeMs_ = 0.0;
    std::uint64_t completedImages_ = 0;
    bool resourcesActive_ = false;
};

} // namespace grav_models

