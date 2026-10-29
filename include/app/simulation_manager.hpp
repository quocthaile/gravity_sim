#pragma once

#include "models/black_hole/black_hole_core.hpp"
#include "models/simulation_module.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace grav_app
{

enum class LifecycleState : std::uint8_t
{
    Uninitialized = 0,
    Running,
    Quiescing,
    Releasing,
    Loading,
    Warmup,
    Error,
};

const char *LifecycleStateName(LifecycleState state);

class SimulationManager
{
  public:
    using ModuleFactoryFn = std::function<std::unique_ptr<grav_sim::ISimulationModule>(grav_sim::ModeId)>;

    SimulationManager();
    ~SimulationManager();

    SimulationManager(const SimulationManager &) = delete;
    SimulationManager &operator=(const SimulationManager &) = delete;

    void Initialize(const grav_sim::ModuleServices &services,
                    grav_sim::ModeId initialMode = grav_sim::ModeId::F1_GravityGpu,
                    std::size_t initialPresetIndex = 0);

    void Shutdown();

    // Non-blocking transition requests; multiple calls before/during transition are coalesced (T05).
    void RequestMode(grav_sim::ModeId targetMode);
    void RequestModeAndPreset(grav_sim::ModeId targetMode, std::size_t presetIndex);
    void RequestPresetIndex(std::size_t presetIndex);
    void RequestCyclePreset(int delta);
    void RequestResetRuntime();

    // Advances the non-blocking state machine by one frame tick.
    void TickLifecycle();

    void HandleInput(const grav_sim::ModuleInputState &input, double wallDeltaSeconds);
    void UpdateActive(const grav_sim::StepContext &step);
    void RenderActive(const grav_sim::RenderContext &render);
    void OnResize(int framebufferWidth, int framebufferHeight);
    void ResetActiveCamera();

    LifecycleState State() const { return state_; }
    bool IsTransitioning() const
    {
        return state_ == LifecycleState::Quiescing || state_ == LifecycleState::Releasing ||
               state_ == LifecycleState::Loading || state_ == LifecycleState::Warmup ||
               pending_.active;
    }

    grav_sim::ModeId ActiveMode() const { return activeMode_; }
    grav_sim::ISimulationModule *ActiveModule() { return activeModule_.get(); }
    const grav_sim::ISimulationModule *ActiveModule() const { return activeModule_.get(); }

    std::uint64_t SceneGeneration() const { return sceneGeneration_; }
    std::uint64_t CompletedLoadCount() const { return completedLoadCount_; }
    double LastTransitionDurationMs() const { return lastTransitionDurationMs_; }

    bool IsGridVisible() const { return gridVisible_; }
    void SetGridVisible(bool visible) { gridVisible_ = visible; }
    void ToggleGridVisible() { gridVisible_ = !gridVisible_; }

    void SetWarmupFrameTarget(int frames) { warmupFrameTarget_ = (frames < 0) ? 0 : frames; }
    int WarmupFrameTarget() const { return warmupFrameTarget_; }

    const std::string &LastErrorMessage() const { return lastErrorMessage_; }
    grav_sim::ModeId FailedMode() const { return failedMode_; }

    // Test & fault-injection hooks (T10)
    void SetCustomModuleFactory(ModuleFactoryFn factory) { customFactory_ = std::move(factory); }
    void InjectNextLoadFailure(const std::string &errorMessage);

    static std::unique_ptr<grav_sim::ISimulationModule> DefaultCreateModule(grav_sim::ModeId mode);

  private:
    struct PendingRequest
    {
        bool active = false;
        grav_sim::ModeId targetMode = grav_sim::ModeId::F1_GravityGpu;
        std::size_t targetPresetIndex = 0;
        bool useRememberedPreset = true;
        bool forceReloadSameMode = false;
    };

    void BeginQuiescing();
    void PerformRelease();
    void PerformLoad();
    static void ResetCommonGlBindings();

    grav_sim::ModuleServices services_{};
    LifecycleState state_ = LifecycleState::Uninitialized;

    std::unique_ptr<grav_sim::ISimulationModule> activeModule_;
    grav_sim::ModeId activeMode_ = grav_sim::ModeId::F1_GravityGpu;
    std::array<std::size_t, static_cast<std::size_t>(grav_sim::ModeId::Count)> rememberedPresets_{};

    PendingRequest pending_{};
    bool gridVisible_ = true;

    std::uint64_t sceneGeneration_ = 1;
    std::uint64_t completedLoadCount_ = 0;
    int warmupFrameTarget_ = 2;
    int warmupFramesRemaining_ = 0;

    std::chrono::steady_clock::time_point transitionStartWall_{};
    std::chrono::steady_clock::time_point quiesceStartWall_{};
    double lastTransitionDurationMs_ = 0.0;

    std::string lastErrorMessage_;
    grav_sim::ModeId failedMode_ = grav_sim::ModeId::F1_GravityGpu;

    ModuleFactoryFn customFactory_;
    bool injectFailureNextLoad_ = false;
    std::string injectedFailureMessage_;
};

} // namespace grav_app

