#pragma once

#include "app/simulation_manager.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "gravity_sim_3Dgrid_function.hpp"
#include "models/black_hole/black_hole_core.hpp"
#include "render/fullscreen_presenter.hpp"
#include "render/resource_census.hpp"
#include "render/shader_cache.hpp"
#include "render/shared_grid.hpp"
#include "telemetry/telemetry_service.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <cstdint>
#include <string>

namespace grav_app
{

struct ApplicationOptions
{
    int width = 1920;
    int height = 1080;
    bool visible = true;
    bool vsync = true;
    grav_sim::ModeId initialMode = grav_sim::ModeId::F1_GravityGpu;
    std::size_t initialPresetIndex = 0;
    int maxFrames = 0; // 0 = run until user quits; >0 = exit cleanly after N frames (smoke/test mode)
    std::string explicitAssetRoot;
    const char *argv0 = nullptr;
};

class Application
{
  public:
    Application() = default;
    ~Application();

    Application(const Application &) = delete;
    Application &operator=(const Application &) = delete;

    bool Initialize(const ApplicationOptions &options = {});
    int Run();
    void StepFrame(double fixedDeltaOverrideSec = -1.0);
    void Shutdown();

    // Programmatic event injection for automated acceptance tests (T05, T08, T12)
    void HandleKeyAction(int key, int scancode, int action, int mods, bool bypassImGuiCapture = false);
    void HandleFramebufferResizeEvent(int width, int height);

    GLFWwindow *Window() const { return window_; }
    bool IsMinimized() const { return minimized_; }
    bool IsPendingQuit() const { return pendingQuit_; }
    bool IsPaused() const { return paused_; }
    void SetPaused(bool paused) { paused_ = paused; }

    SimulationManager &SimManager() { return simManager_; }
    const SimulationManager &SimManager() const { return simManager_; }
    grav_render::SharedGridService &SharedGrid() { return sharedGrid_; }
    grav_render::ResourceCensus &Census() { return resourceCensus_; }
    grav_render::ShaderCache &Shaders() { return shaderCache_; }
    grav_telemetry::TelemetryService &Telemetry() { return telemetry_; }
    grav_bench::BenchmarkRunner &Benchmark() { return benchmark_; }
    grav_models::BlackHoleParameters &SharedBlackHole() { return sharedBlackHole_; }

    void StartBenchmarkRun();
    void AbortBenchmarkRun(const std::string &reason);

  private:
    static void GlfwKeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void GlfwCharCallback(GLFWwindow *window, unsigned int codepoint);
    static void GlfwMouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    static void GlfwCursorPosCallback(GLFWwindow *window, double xpos, double ypos);
    static void GlfwScrollCallback(GLFWwindow *window, double xoffset, double yoffset);
    static void GlfwFramebufferSizeCallback(GLFWwindow *window, int width, int height);
    static void GlfwWindowFocusCallback(GLFWwindow *window, int focused);

    void ClearHeldInputState();
    void ToggleFullscreen();
    void ApplyVsync(bool enabled);
    grav_bench::WorkloadFingerprint CaptureWorkloadFingerprint() const;

    void RenderImGuiInterface();
    void DrawCompactHudOverlay();
    void DrawControlPanelWindow();
    void DrawInteractiveTab();
    void DrawBenchmarkTab();
    void DrawDiagnosticsTab();
    void DrawHelpModal();
    void DrawErrorBanner();

    ApplicationOptions options_{};
    GLFWwindow *window_ = nullptr;
    bool initialized_ = false;
    bool imguiInitialized_ = false;

    int framebufferWidth_ = 1920;
    int framebufferHeight_ = 1080;
    bool minimized_ = false;
    bool pendingResize_ = false;

    bool fullscreen_ = false;
    int windowedPosX_ = 100;
    int windowedPosY_ = 100;
    int windowedWidth_ = 1600;
    int windowedHeight_ = 900;

    bool vsyncEnabled_ = true;
    bool savedUserVsync_ = true;
    bool savedUserHudVisible_ = true;
    bool benchmarkOverridesApplied_ = false;

    bool pendingQuit_ = false;
    bool paused_ = false;
    bool singleStepRequested_ = false;
    bool hudVisible_ = true;
    bool controlPanelVisible_ = true;
    bool showHelpModal_ = false;

    grav_sim::ModuleInputState liveInput_{};
    bool hasPrevCursor_ = false;
    double prevCursorX_ = 0.0;
    double prevCursorY_ = 0.0;

    std::chrono::steady_clock::time_point prevFrameWall_{};
    bool hasPrevFrameWall_ = false;
    std::uint64_t totalPresentedFrames_ = 0;

    grav_render::ResourceCensus resourceCensus_;
    grav_render::ShaderCache shaderCache_;
    grav_render::FullscreenPresenter fullscreenPresenter_;
    grav_render::SharedGridService sharedGrid_;
    grav_telemetry::TelemetryService telemetry_;
    grav_models::BlackHoleParameters sharedBlackHole_;
    SimulationManager simManager_;
    grav_bench::BenchmarkRunner benchmark_;
};

} // namespace grav_app

