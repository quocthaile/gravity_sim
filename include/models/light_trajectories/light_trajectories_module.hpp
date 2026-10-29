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

// Fixed-capacity ring buffer separated from PlanarRayState so RK4 stages never copy trail history
// and long-running stress runs remain strictly memory-bounded (Section 9.2 & T14).
class BoundedTrailRing
{
  public:
    void ConfigureCapacity(std::size_t capacity);
    void Clear();
    void Push(const glm::vec2 &posMeters);

    std::size_t Size() const { return count_; }
    std::size_t Capacity() const { return buffer_.size(); }
    glm::vec2 AtChronological(std::size_t index) const;
    void ReleaseMemory();

  private:
    std::vector<glm::vec2> buffer_;
    std::size_t head_ = 0;
    std::size_t count_ = 0;
};

class LightTrajectoriesModule final : public grav_sim::ISimulationModule
{
  public:
    LightTrajectoriesModule();
    ~LightTrajectoriesModule() override;

    grav_sim::ModeId Id() const override { return grav_sim::ModeId::F2_LightTrajectories; }
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
    std::uint64_t CameraRevision() const override;

    void BeginExit() override;
    void ReleaseResources() override;

    grav_sim::ModuleRuntimeStats RuntimeStats() const override;
    void DrawInteractiveControls(bool configLocked) override;
    std::string ControlsHelpText() const override;

    // Inspection hooks for automated tests (T14)
    std::size_t RayCount() const { return rays_.size(); }
    std::size_t TrailCapacityPerRay() const { return trailCapacity_; }
    std::size_t TotalAllocatedTrailPoints() const;
    std::uint64_t LastFrameDrawCallCount() const { return lastFrameDrawCalls_; }
    double MaxNullRelativeError() const { return maxNullRelError_; }
    double MaxAngularMomentumRelativeError() const { return maxAngMomRelError_; }

  private:
    struct TrailVertex
    {
        glm::vec3 posWorld;
        glm::vec4 color;
    };

    void BuildPresetRays(std::size_t presetIndex);
    void SyncBlackHoleToGridAndHorizon();
    void RebuildHorizonGeometry();
    void UploadTrailGeometry();

    grav_sim::ModuleServices services_{};
    std::vector<grav_sim::PresetDescriptor> presets_;
    std::size_t activePresetIndex_ = 0;

    std::vector<grav_sim::SolverDescriptor> solvers_;
    std::size_t activeSolverIndex_ = 0;

    bool use2DOrthoCamera_ = false;
    grav_app::OrbitPanCamera orbitCamera_;
    grav_app::PlanarOrthoCamera orthoCamera_;
    bool camerasConfigured_ = false;

    GLuint trailProgram_ = 0;
    GLint locViewProj_ = -1;

    GLuint horizonVao_ = 0;
    GLuint horizonVbo_ = 0;
    std::uint64_t horizonVaoCensusId_ = 0;
    std::uint64_t horizonVboCensusId_ = 0;
    GLsizei horizonFanVertexCount_ = 0;
    GLsizei horizonRimFirst_ = 0;
    GLsizei horizonRimCount_ = 0;
    GLsizei photonSphereFirst_ = 0;
    GLsizei photonSphereCount_ = 0;

    GLuint trailVao_ = 0;
    GLuint trailVbo_ = 0;
    std::uint64_t trailVaoCensusId_ = 0;
    std::uint64_t trailVboCensusId_ = 0;
    std::size_t trailVboCapacityVertices_ = 0;
    GLsizei uploadedLineVertexCount_ = 0;
    GLsizei uploadedHeadFirst_ = 0;
    GLsizei uploadedHeadCount_ = 0;

    std::vector<PlanarRayState> rays_;
    std::vector<BoundedTrailRing> trails_;
    std::vector<TrailVertex> cpuScratchVertices_;

    std::size_t trailCapacity_ = 1024;
    int substepsPerFrame_ = 4;
    float dLambdaAffine_ = 1.0f;
    std::uint64_t maxAffineSteps_ = 25000;
    std::uint64_t stepCount_ = 0;
    double cumulativeAffine_ = 0.0;
    std::uint64_t lastObservedBhRevision_ = 0;
    bool trailGeometryDirty_ = true;

    double maxNullRelError_ = 0.0;
    double maxAngMomRelError_ = 0.0;
    std::uint64_t lastFrameDrawCalls_ = 0;
    bool resourcesActive_ = false;
};

} // namespace grav_models

