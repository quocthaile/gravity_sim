#pragma once

#include "gpu_memory_manager.hpp"
#include "render/fullscreen_presenter.hpp"
#include "render/resource_census.hpp"
#include "render/shader_cache.hpp"
#include "render/shared_grid.hpp"
#include "telemetry/telemetry_service.hpp"
#include <GLFW/glfw3.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace grav_models
{
struct BlackHoleParameters;
}

namespace grav_sim
{

enum class ModeId : std::uint8_t
{
    F1_GravityGpu = 0,
    F2_LightTrajectories,
    F3_BlackHoleGpu,
    F4_FlatRayTracingCpu,
    F5_BlackHoleCpu,
    F6_GravityBaselineCpu,
    Diagnostic_Triangle,
    Count,
};

inline const char *ModeShortCode(ModeId mode)
{
    switch (mode)
    {
    case ModeId::F1_GravityGpu:
        return "F1";
    case ModeId::F2_LightTrajectories:
        return "F2";
    case ModeId::F3_BlackHoleGpu:
        return "F3";
    case ModeId::F4_FlatRayTracingCpu:
        return "F4";
    case ModeId::F5_BlackHoleCpu:
        return "F5";
    case ModeId::F6_GravityBaselineCpu:
        return "F6";
    case ModeId::Diagnostic_Triangle:
        return "Ctrl+F12";
    case ModeId::Count:
        break;
    }
    return "Unknown";
}

inline const char *ModeDisplayName(ModeId mode)
{
    switch (mode)
    {
    case ModeId::F1_GravityGpu:
        return "Gravity N-Body (GPU Compute)";
    case ModeId::F2_LightTrajectories:
        return "Black Hole Light Trajectories (Planar Geodesics)";
    case ModeId::F3_BlackHoleGpu:
        return "Black Hole Lensing Image (GPU Compute)";
    case ModeId::F4_FlatRayTracingCpu:
        return "Flat-Space Ray Tracing (CPU)";
    case ModeId::F5_BlackHoleCpu:
        return "Black Hole Lensing Image (CPU)";
    case ModeId::F6_GravityBaselineCpu:
        return "Historical CPU Gravity Adapter";
    case ModeId::Diagnostic_Triangle:
        return "Graphics Context Diagnostic (Triangle)";
    case ModeId::Count:
        break;
    }
    return "Unknown";
}

inline const char *ModeProvenanceSummary(ModeId mode)
{
    switch (mode)
    {
    case ModeId::F1_GravityGpu:
        return "gravity_sim: src/gravity_sim_3Dgrid.cpp + nbody.comp + integrate.comp";
    case ModeId::F2_LightTrajectories:
        return "black_hole: 2D_lensing.cpp adapted to OpenGL 4.4 Core + SharedGridService";
    case ModeId::F3_BlackHoleGpu:
        return "black_hole: black_hole.cpp + geodesic.comp (std140 fixed + RK4 canonical)";
    case ModeId::F4_FlatRayTracingCpu:
        return "black_hole: ray_tracing.cpp + Gravity_Sim.zip flat ray-tracing variants";
    case ModeId::F5_BlackHoleCpu:
        return "black_hole: CPU-geodesic.cpp (geometric silhouette & 3D geodesics)";
    case ModeId::F6_GravityBaselineCpu:
        return "gravity_sim: src/gravity_sim_3Dgrid_baseline.cpp namespaced physics adapter";
    case ModeId::Diagnostic_Triangle:
        return "gravity_sim: src/legacy/3D_test.cpp core-profile diagnostic";
    case ModeId::Count:
        break;
    }
    return "Unknown";
}

enum class SolverValidationStatus : std::uint8_t
{
    SourceCompatibleLegacy = 0,
    CorrectedImplementation,
    NumericalValidated,
    DiagnosticOnly,
};

inline const char *SolverStatusLabel(SolverValidationStatus status)
{
    switch (status)
    {
    case SolverValidationStatus::SourceCompatibleLegacy:
        return "source_compatible_legacy";
    case SolverValidationStatus::CorrectedImplementation:
        return "corrected_implementation";
    case SolverValidationStatus::NumericalValidated:
        return "numerical_validated";
    case SolverValidationStatus::DiagnosticOnly:
        return "diagnostic_only";
    }
    return "unknown";
}

struct PresetDescriptor
{
    std::string id;
    std::string displayName;
    std::string provenance;
    std::string description;
    std::uint32_t revision = 1;
};

struct SolverDescriptor
{
    std::string id;
    std::string displayName;
    std::string backendName;
    SolverValidationStatus status = SolverValidationStatus::SourceCompatibleLegacy;
    std::string notes;
};

struct ModuleRuntimeStats
{
    std::size_t primaryCount = 0;
    const char *primaryLabel = "Objects";
    int outputWidth = 0;
    int outputHeight = 0;
    std::uint64_t stepCount = 0;
    double simulationOrAffineParam = 0.0;
    const char *paramLabel = "Nominal Time (s)";
    std::uint32_t seed = 0;
    bool hasSeed = false;
    bool usesSharedBlackHole = false;
    bool gridOverlayDiagnosticOnly = false;
    bool hasAsyncWorker = false;
    bool workerBusy = false;
    float workerProgress = 1.0f;
    std::uint32_t workerThreadCount = 1;
    std::string extraStatus;
};

struct ModuleServices
{
    GLFWwindow *window = nullptr;
    GpuMemoryManager *gpuMemoryManager = nullptr;
    grav_render::ShaderCache *shaderCache = nullptr;
    grav_render::ResourceCensus *resourceCensus = nullptr;
    grav_render::FullscreenPresenter *fullscreenPresenter = nullptr;
    grav_render::SharedGridService *sharedGrid = nullptr;
    grav_telemetry::TelemetryService *telemetry = nullptr;
    grav_models::BlackHoleParameters *sharedBlackHole = nullptr;
    std::uint64_t sceneGeneration = 1;
    int framebufferWidth = 1920;
    int framebufferHeight = 1080;
};

struct ModuleInputState
{
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool moveUp = false;
    bool moveDown = false;
    bool shiftHeld = false;
    bool holdPauseK = false;

    bool arrowUp = false;
    bool arrowDown = false;
    bool arrowLeft = false;
    bool arrowRight = false;

    bool mouseLeftDown = false;
    bool mouseRightDown = false;
    bool mouseMiddleDown = false;
    bool mouseLeftJustPressed = false;
    bool mouseLeftJustReleased = false;
    bool mouseRightJustPressed = false;
    bool mouseRightJustReleased = false;
    bool mouseMiddleJustPressed = false;
    bool mouseMiddleJustReleased = false;

    double mouseX = 0.0;
    double mouseY = 0.0;
    double mouseDeltaX = 0.0;
    double mouseDeltaY = 0.0;
    double scrollDeltaY = 0.0;

    bool resetCameraRequested = false;
    bool focusLostThisFrame = false;
};

struct StepContext
{
    double wallDeltaSeconds = 0.0;
    bool paused = false;
    bool singleStepRequested = false;
    bool gridVisible = true;
    bool forceRecomputeStaticImage = false;
    bool benchmarkConfigLocked = false;
    std::uint64_t sceneGeneration = 1;
};

struct RenderContext
{
    int framebufferWidth = 1920;
    int framebufferHeight = 1080;
    bool gridVisible = true;
    std::uint64_t sceneGeneration = 1;
};

class ISimulationModule
{
  public:
    virtual ~ISimulationModule() = default;

    virtual ModeId Id() const = 0;
    virtual const std::vector<PresetDescriptor> &Presets() const = 0;
    virtual std::size_t ActivePresetIndex() const = 0;
    virtual const PresetDescriptor &ActivePreset() const = 0;
    virtual const SolverDescriptor &ActiveSolver() const = 0;
    virtual std::vector<SolverDescriptor> AvailableSolvers() const { return {ActiveSolver()}; }
    virtual bool SetActiveSolverById(const std::string &solverId)
    {
        (void)solverId;
        return false;
    }

    virtual bool DefaultGridVisible() const { return true; }

    virtual void OnEnter(const ModuleServices &services, std::size_t presetIndex) = 0;
    virtual void HandleInput(const ModuleInputState &input, double wallDeltaSeconds) = 0;
    virtual void Update(const StepContext &step) = 0;
    virtual void Render(const RenderContext &render) = 0;
    virtual void OnResize(int framebufferWidth, int framebufferHeight) = 0;
    virtual void ResetCamera() = 0;
    virtual std::uint64_t CameraRevision() const = 0;

    virtual void BeginExit() = 0;
    virtual bool IsReadyToRelease() const { return true; }
    virtual void ReleaseResources() = 0;

    virtual ModuleRuntimeStats RuntimeStats() const = 0;
    virtual void DrawInteractiveControls(bool configLocked) { (void)configLocked; }
    virtual std::string ControlsHelpText() const = 0;
};

} // namespace grav_sim

