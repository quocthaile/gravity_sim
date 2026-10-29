#include "app/application.hpp"

#include "render/asset_paths.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <sstream>

namespace grav_app
{

namespace
{

ImVec4 SolverStatusBadgeColor(grav_sim::SolverValidationStatus status)
{
    switch (status)
    {
    case grav_sim::SolverValidationStatus::NumericalValidated:
        return ImVec4(0.22f, 0.85f, 0.45f, 1.0f);
    case grav_sim::SolverValidationStatus::CorrectedImplementation:
        return ImVec4(0.25f, 0.75f, 0.95f, 1.0f);
    case grav_sim::SolverValidationStatus::SourceCompatibleLegacy:
        return ImVec4(0.95f, 0.72f, 0.22f, 1.0f);
    case grav_sim::SolverValidationStatus::DiagnosticOnly:
        return ImVec4(0.75f, 0.75f, 0.80f, 1.0f);
    }
    return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
}

std::string CompilerSummaryString()
{
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    std::ostringstream oss;
    oss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
    return oss.str();
#elif defined(_MSC_VER)
    std::ostringstream oss;
    oss << "MSVC " << _MSC_FULL_VER;
    return oss.str();
#else
    return "Unknown C++20 Compiler";
#endif
}

std::string BuildTypeString()
{
#if defined(NDEBUG)
    return "Release";
#else
    return "Debug";
#endif
}

void ApplyBenchmarkDarkTheme()
{
    ImGui::StyleColorsDark();
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.09f, 0.12f, 0.93f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.14f, 0.22f, 0.34f, 1.00f);
    style.Colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.36f, 0.56f, 1.00f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.18f, 0.30f, 0.46f, 0.80f);
}

} // namespace

Application::~Application() { Shutdown(); }

bool Application::Initialize(const ApplicationOptions &options)
{
    if (initialized_)
    {
        return true;
    }

    options_ = options;
    if (!options_.explicitAssetRoot.empty())
    {
        grav_render::SetAssetRootOverride(options_.explicitAssetRoot);
    }
    else
    {
        grav_render::ClearAssetRootOverride();
    }

    window_ = StartGLU(options_.width, options_.height,
                       "GravitySim - Unified Gravity & Black Hole Simulator", options_.visible);
    if (window_ == nullptr)
    {
        std::cerr << "[Application] Failed to initialize GLFW / OpenGL 4.4 Core window.\n";
        return false;
    }

    GLint majorVer = 0;
    GLint minorVer = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &majorVer);
    glGetIntegerv(GL_MINOR_VERSION, &minorVer);
    if (majorVer < 4 || (majorVer == 4 && minorVer < 4))
    {
        std::cerr << "[Application] OpenGL 4.4 Core Profile is required, but context reported "
                  << majorVer << "." << minorVer << ".\n";
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
        return false;
    }

    glfwGetFramebufferSize(window_, &framebufferWidth_, &framebufferHeight_);
    if (framebufferWidth_ <= 0 || framebufferHeight_ <= 0)
    {
        framebufferWidth_ = std::max(options_.width, 1);
        framebufferHeight_ = std::max(options_.height, 1);
    }
    glViewport(0, 0, framebufferWidth_, framebufferHeight_);

    vsyncEnabled_ = options_.vsync;
    ApplyVsync(vsyncEnabled_);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr; // Keep workspace directory clean across test & benchmark runs
    ApplyBenchmarkDarkTheme();

    ImGui_ImplGlfw_InitForOpenGL(window_, false);
    ImGui_ImplOpenGL3_Init("#version 440");
    imguiInitialized_ = true;

    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, &Application::GlfwKeyCallback);
    glfwSetCharCallback(window_, &Application::GlfwCharCallback);
    glfwSetMouseButtonCallback(window_, &Application::GlfwMouseButtonCallback);
    glfwSetCursorPosCallback(window_, &Application::GlfwCursorPosCallback);
    glfwSetScrollCallback(window_, &Application::GlfwScrollCallback);
    glfwSetFramebufferSizeCallback(window_, &Application::GlfwFramebufferSizeCallback);
    glfwSetWindowFocusCallback(window_, &Application::GlfwWindowFocusCallback);

    const char *glVendorCStr = reinterpret_cast<const char *>(glGetString(GL_VENDOR));
    const char *glRendererCStr = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    const char *glVersionCStr = reinterpret_cast<const char *>(glGetString(GL_VERSION));
    const char *glslVersionCStr =
        reinterpret_cast<const char *>(glGetString(GL_SHADING_LANGUAGE_VERSION));

    const std::string glRenderer = glRendererCStr ? glRendererCStr : "Unknown Renderer";

    shaderCache_.SetCensus(&resourceCensus_);
    fullscreenPresenter_.Initialize(shaderCache_, resourceCensus_);
    sharedGrid_.Initialize(gpuMemoryManager, shaderCache_, resourceCensus_,
                           grav_render::GridConfig::GravityCompatibility());
    telemetry_.Initialize(glRenderer);

    grav_bench::DeviceEnvironmentInfo envInfo{};
    envInfo.glVendor = glVendorCStr ? glVendorCStr : "Unknown";
    envInfo.glRenderer = glRenderer;
    envInfo.glVersion = glVersionCStr ? glVersionCStr : "Unknown";
    envInfo.glslVersion = glslVersionCStr ? glslVersionCStr : "Unknown";
    envInfo.dxgiAdapter = telemetry_.SystemMetrics().dxgiAvailable
                              ? telemetry_.SystemMetrics().dxgiAdapterName
                              : "N/A";
    envInfo.osName = "Windows x64";
    envInfo.compilerInfo = CompilerSummaryString();
    envInfo.buildType = BuildTypeString();
    benchmark_.SetEnvironmentInfo(envInfo);

    sharedBlackHole_.SetMassKg(grav_models::kSagittariusAMassKg);

    grav_sim::ModuleServices services{};
    services.window = window_;
    services.gpuMemoryManager = &gpuMemoryManager;
    services.shaderCache = &shaderCache_;
    services.resourceCensus = &resourceCensus_;
    services.fullscreenPresenter = &fullscreenPresenter_;
    services.sharedGrid = &sharedGrid_;
    services.telemetry = &telemetry_;
    services.sharedBlackHole = &sharedBlackHole_;
    services.sceneGeneration = 1;
    services.framebufferWidth = framebufferWidth_;
    services.framebufferHeight = framebufferHeight_;

    simManager_.Initialize(services, options_.initialMode, options_.initialPresetIndex);

    prevFrameWall_ = std::chrono::steady_clock::now();
    hasPrevFrameWall_ = true;
    initialized_ = true;
    return true;
}

int Application::Run()
{
    if (!initialized_)
    {
        return 1;
    }

    while (!pendingQuit_ && (window_ != nullptr && !glfwWindowShouldClose(window_)))
    {
        StepFrame();

        if (options_.maxFrames > 0 &&
            totalPresentedFrames_ >= static_cast<std::uint64_t>(options_.maxFrames))
        {
            pendingQuit_ = true;
        }
    }

    return 0;
}

void Application::StepFrame(double fixedDeltaOverrideSec)
{
    if (!initialized_ || window_ == nullptr)
    {
        return;
    }

    glfwPollEvents();
    if (pendingQuit_ || glfwWindowShouldClose(window_))
    {
        return;
    }

    const auto nowWall = std::chrono::steady_clock::now();
    double wallDeltaSec = 1.0 / 60.0;
    if (fixedDeltaOverrideSec >= 0.0)
    {
        wallDeltaSec = fixedDeltaOverrideSec;
    }
    else if (hasPrevFrameWall_)
    {
        wallDeltaSec = std::chrono::duration<double>(nowWall - prevFrameWall_).count();
    }
    prevFrameWall_ = nowWall;
    hasPrevFrameWall_ = true;
    const double clampedInputDt = std::clamp(wallDeltaSec, 0.0, 0.1);

    // Handle minimized 0x0 window safely without allocating 0x0 targets or crashing (T08).
    if (minimized_ || framebufferWidth_ <= 0 || framebufferHeight_ <= 0)
    {
        if (benchmark_.IsActive())
        {
            AbortBenchmarkRun("Aborted: window minimized during benchmark run");
        }
        liveInput_.mouseDeltaX = 0.0;
        liveInput_.mouseDeltaY = 0.0;
        liveInput_.scrollDeltaY = 0.0;
        liveInput_.mouseLeftJustPressed = false;
        liveInput_.mouseLeftJustReleased = false;
        liveInput_.mouseRightJustPressed = false;
        liveInput_.mouseRightJustReleased = false;
        liveInput_.mouseMiddleJustPressed = false;
        liveInput_.mouseMiddleJustReleased = false;
        liveInput_.resetCameraRequested = false;
        liveInput_.focusLostThisFrame = false;
        glfwWaitEventsTimeout(0.016);
        return;
    }

    if (pendingResize_)
    {
        pendingResize_ = false;
        glViewport(0, 0, framebufferWidth_, framebufferHeight_);
        simManager_.OnResize(framebufferWidth_, framebufferHeight_);
    }

    // 1. Advance non-blocking lifecycle transitions
    simManager_.TickLifecycle();

    telemetry_.BeginFrame(wallDeltaSec, simManager_.SceneGeneration());

    // Restore VSync/HUD if benchmark completed or aborted on previous tick
    if (!benchmark_.IsActive() && benchmarkOverridesApplied_)
    {
        ApplyVsync(savedUserVsync_);
        hudVisible_ = savedUserHudVisible_;
        benchmarkOverridesApplied_ = false;
    }

    // 2. Route scene input & update active module
    const auto updateStart = std::chrono::steady_clock::now();
    simManager_.HandleInput(liveInput_, clampedInputDt);

    grav_sim::StepContext stepCtx{};
    stepCtx.wallDeltaSeconds = clampedInputDt;
    stepCtx.paused = paused_;
    stepCtx.singleStepRequested = singleStepRequested_;
    stepCtx.gridVisible = simManager_.IsGridVisible();
    stepCtx.forceRecomputeStaticImage = benchmark_.ShouldForceStaticRecompute();
    stepCtx.benchmarkConfigLocked = benchmark_.IsConfigLocked();
    stepCtx.sceneGeneration = simManager_.SceneGeneration();

    simManager_.UpdateActive(stepCtx);
    singleStepRequested_ = false;

    const auto updateEnd = std::chrono::steady_clock::now();
    telemetry_.AddCpuUpdateMs(
        std::chrono::duration<double, std::milli>(updateEnd - updateStart).count());

    // Clear per-frame edge-triggered input fields
    liveInput_.mouseDeltaX = 0.0;
    liveInput_.mouseDeltaY = 0.0;
    liveInput_.scrollDeltaY = 0.0;
    liveInput_.mouseLeftJustPressed = false;
    liveInput_.mouseLeftJustReleased = false;
    liveInput_.mouseRightJustPressed = false;
    liveInput_.mouseRightJustReleased = false;
    liveInput_.mouseMiddleJustPressed = false;
    liveInput_.mouseMiddleJustReleased = false;
    liveInput_.resetCameraRequested = false;
    liveInput_.focusLostThisFrame = false;

    // 3. Render active module scene
    const auto renderStart = std::chrono::steady_clock::now();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, framebufferWidth_, framebufferHeight_);
    glClearColor(0.02f, 0.025f, 0.04f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    grav_sim::RenderContext renderCtx{};
    renderCtx.framebufferWidth = framebufferWidth_;
    renderCtx.framebufferHeight = framebufferHeight_;
    renderCtx.gridVisible = simManager_.IsGridVisible();
    renderCtx.sceneGeneration = simManager_.SceneGeneration();

    simManager_.RenderActive(renderCtx);
    const auto renderEnd = std::chrono::steady_clock::now();
    telemetry_.AddCpuRenderSubmitMs(
        std::chrono::duration<double, std::milli>(renderEnd - renderStart).count());

    // 4. Render Dear ImGui HUD & Control Panels
    const auto uiStart = std::chrono::steady_clock::now();
    {
        grav_telemetry::ScopedGpuPassQuery uiGpuQuery(
            &telemetry_, grav_telemetry::GpuPassId::UiDraw, simManager_.SceneGeneration());
        RenderImGuiInterface();
    }
    const auto uiEnd = std::chrono::steady_clock::now();
    telemetry_.AddCpuUiMs(std::chrono::duration<double, std::milli>(uiEnd - uiStart).count());

    telemetry_.EndFrame();

    // 5. Feed BenchmarkRunner after steady-state frame submission
    if (benchmark_.IsActive())
    {
        if (simManager_.State() != LifecycleState::Running)
        {
            AbortBenchmarkRun("Aborted: simulation left steady Running state during benchmark");
        }
        else
        {
            benchmark_.OnFrameSample(CaptureWorkloadFingerprint(), wallDeltaSec * 1000.0,
                                     telemetry_, &resourceCensus_, &gpuMemoryManager);
            if (!benchmark_.IsActive() && benchmarkOverridesApplied_)
            {
                ApplyVsync(savedUserVsync_);
                hudVisible_ = savedUserHudVisible_;
                benchmarkOverridesApplied_ = false;
            }
        }
    }

    // 6. Single authoritative swap per frame
    glfwSwapBuffers(window_);
    ++totalPresentedFrames_;
}

void Application::Shutdown()
{
    if (!initialized_ && window_ == nullptr)
    {
        return;
    }

    if (benchmark_.IsActive())
    {
        benchmark_.Abort("Aborted: application shutdown");
    }

    simManager_.Shutdown();
    sharedGrid_.Shutdown();
    fullscreenPresenter_.Shutdown();
    shaderCache_.Shutdown();
    telemetry_.Shutdown();
    gpuMemoryManager.Shutdown();

    if (imguiInitialized_)
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        imguiInitialized_ = false;
    }

    if (window_ != nullptr)
    {
        glfwSetWindowUserPointer(window_, nullptr);
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
    }

    initialized_ = false;
}

void Application::StartBenchmarkRun()
{
    if (simManager_.State() != LifecycleState::Running || simManager_.ActiveModule() == nullptr)
    {
        return;
    }

    savedUserVsync_ = vsyncEnabled_;
    savedUserHudVisible_ = hudVisible_;
    if (benchmark_.Config().disableVsyncDuringRun)
    {
        ApplyVsync(false);
    }
    if (benchmark_.Config().hideHudDuringRun)
    {
        hudVisible_ = false;
    }
    benchmarkOverridesApplied_ = true;

    benchmark_.Start(CaptureWorkloadFingerprint(), simManager_.LastTransitionDurationMs(),
                     &resourceCensus_, &gpuMemoryManager);
}

void Application::AbortBenchmarkRun(const std::string &reason)
{
    if (!benchmark_.IsActive())
    {
        return;
    }
    benchmark_.Abort(reason);
    if (benchmarkOverridesApplied_)
    {
        ApplyVsync(savedUserVsync_);
        hudVisible_ = savedUserHudVisible_;
        benchmarkOverridesApplied_ = false;
    }
}

void Application::ApplyVsync(bool enabled)
{
    vsyncEnabled_ = enabled;
    if (window_ != nullptr)
    {
        glfwSwapInterval(enabled ? 1 : 0);
    }
}

void Application::ToggleFullscreen()
{
    if (window_ == nullptr)
    {
        return;
    }

    if (!fullscreen_)
    {
        glfwGetWindowPos(window_, &windowedPosX_, &windowedPosY_);
        glfwGetWindowSize(window_, &windowedWidth_, &windowedHeight_);

        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = monitor ? glfwGetVideoMode(monitor) : nullptr;
        if (monitor != nullptr && mode != nullptr)
        {
            glfwSetWindowMonitor(window_, monitor, 0, 0, mode->width, mode->height,
                                 mode->refreshRate);
            fullscreen_ = true;
            ApplyVsync(vsyncEnabled_);
        }
    }
    else
    {
        glfwSetWindowMonitor(window_, nullptr, windowedPosX_, windowedPosY_,
                             std::max(windowedWidth_, 640), std::max(windowedHeight_, 480), 0);
        fullscreen_ = false;
        ApplyVsync(vsyncEnabled_);
    }
}

grav_bench::WorkloadFingerprint Application::CaptureWorkloadFingerprint() const
{
    grav_bench::WorkloadFingerprint fp{};
    fp.mode = simManager_.ActiveMode();
    fp.modeCode = grav_sim::ModeShortCode(fp.mode);
    fp.modeName = grav_sim::ModeDisplayName(fp.mode);
    fp.displayWidth = framebufferWidth_;
    fp.displayHeight = framebufferHeight_;
    fp.gridDivisions = sharedGrid_.Config().divisions;
    fp.gridVisible = simManager_.IsGridVisible();
    fp.hudVisible = hudVisible_;
    fp.vsyncEnabled = vsyncEnabled_;
    fp.blackHoleRevision = sharedBlackHole_.revision;

    if (const auto *mod = simManager_.ActiveModule())
    {
        const auto &preset = mod->ActivePreset();
        const auto &solver = mod->ActiveSolver();
        const auto stats = mod->RuntimeStats();

        fp.presetId = preset.id;
        fp.presetName = preset.displayName;
        fp.presetRevision = preset.revision;
        fp.solverId = solver.id;
        fp.solverName = solver.displayName;
        fp.solverStatus = grav_sim::SolverStatusLabel(solver.status);
        fp.backendName = solver.backendName;

        fp.outputWidth = stats.outputWidth > 0 ? stats.outputWidth : framebufferWidth_;
        fp.outputHeight = stats.outputHeight > 0 ? stats.outputHeight : framebufferHeight_;
        fp.primaryCount = stats.primaryCount;
        fp.primaryLabel = stats.primaryLabel ? stats.primaryLabel : "Items";
        fp.hasSeed = stats.hasSeed;
        fp.seed = stats.seed;
        fp.cameraRevision = mod->CameraRevision();
    }
    return fp;
}

void Application::ClearHeldInputState()
{
    liveInput_.moveForward = false;
    liveInput_.moveBackward = false;
    liveInput_.moveLeft = false;
    liveInput_.moveRight = false;
    liveInput_.moveUp = false;
    liveInput_.moveDown = false;
    liveInput_.shiftHeld = false;
    liveInput_.holdPauseK = false;
    liveInput_.arrowUp = false;
    liveInput_.arrowDown = false;
    liveInput_.arrowLeft = false;
    liveInput_.arrowRight = false;
    liveInput_.mouseLeftDown = false;
    liveInput_.mouseRightDown = false;
    liveInput_.mouseMiddleDown = false;
}

void Application::HandleKeyAction(int key, int scancode, int action, int mods,
                                  bool bypassImGuiCapture)
{
    (void)scancode;
    const bool wantKeyboard =
        (!bypassImGuiCapture && imguiInitialized_) ? ImGui::GetIO().WantCaptureKeyboard : false;

    const bool pressed = (action == GLFW_PRESS);
    const bool heldOrPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);

    // Continuous movement keys update on PRESS and RELEASE (never trigger mode loads!)
    if (!wantKeyboard || action == GLFW_RELEASE)
    {
        const bool down = heldOrPressed && !wantKeyboard;
        switch (key)
        {
        case GLFW_KEY_W:
            liveInput_.moveForward = down;
            break;
        case GLFW_KEY_S:
            liveInput_.moveBackward = down;
            break;
        case GLFW_KEY_A:
            liveInput_.moveLeft = down;
            break;
        case GLFW_KEY_D:
            liveInput_.moveRight = down;
            break;
        case GLFW_KEY_SPACE:
            liveInput_.moveUp = down;
            break;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
            liveInput_.moveDown = down;
            break;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            liveInput_.shiftHeld = down;
            break;
        case GLFW_KEY_K:
            liveInput_.holdPauseK = down;
            break;
        case GLFW_KEY_UP:
            liveInput_.arrowUp = down;
            break;
        case GLFW_KEY_DOWN:
            liveInput_.arrowDown = down;
            break;
        case GLFW_KEY_LEFT:
            liveInput_.arrowLeft = down;
            break;
        case GLFW_KEY_RIGHT:
            liveInput_.arrowRight = down;
            break;
        default:
            break;
        }
    }

    // Discrete global actions trigger ONLY on GLFW_PRESS (GLFW_REPEAT is strictly ignored: T05).
    if (!pressed)
    {
        return;
    }

    if (key == GLFW_KEY_ESCAPE)
    {
        if (showHelpModal_)
        {
            showHelpModal_ = false;
        }
        else
        {
            ClearHeldInputState();
        }
        return;
    }

    if (key == GLFW_KEY_F8)
    {
        showHelpModal_ = !showHelpModal_;
        return;
    }

    if (key == GLFW_KEY_F7)
    {
        if (!benchmark_.IsConfigLocked())
        {
            hudVisible_ = !hudVisible_;
        }
        return;
    }

    if (key == GLFW_KEY_F9)
    {
        if (benchmark_.IsActive())
        {
            AbortBenchmarkRun("Aborted by user (F9)");
        }
        else
        {
            StartBenchmarkRun();
        }
        return;
    }

    if (key == GLFW_KEY_F11)
    {
        if (benchmark_.IsActive())
        {
            AbortBenchmarkRun("Aborted: fullscreen toggle requested during benchmark");
        }
        ToggleFullscreen();
        return;
    }

    if (wantKeyboard)
    {
        return;
    }

    auto abortBenchIfActive = [&](const char *reason)
    {
        if (benchmark_.IsActive())
        {
            AbortBenchmarkRun(reason);
        }
    };

    switch (key)
    {
    case GLFW_KEY_Q:
        pendingQuit_ = true;
        break;
    case GLFW_KEY_F1:
        abortBenchIfActive("Aborted: switched to F1 during benchmark run");
        simManager_.RequestMode(grav_sim::ModeId::F1_GravityGpu);
        break;
    case GLFW_KEY_F2:
        abortBenchIfActive("Aborted: switched to F2 during benchmark run");
        simManager_.RequestMode(grav_sim::ModeId::F2_LightTrajectories);
        break;
    case GLFW_KEY_F3:
        abortBenchIfActive("Aborted: switched to F3 during benchmark run");
        simManager_.RequestMode(grav_sim::ModeId::F3_BlackHoleGpu);
        break;
    case GLFW_KEY_F4:
        abortBenchIfActive("Aborted: switched to F4 during benchmark run");
        simManager_.RequestMode(grav_sim::ModeId::F4_FlatRayTracingCpu);
        break;
    case GLFW_KEY_F5:
        abortBenchIfActive("Aborted: switched to F5 during benchmark run");
        simManager_.RequestMode(grav_sim::ModeId::F5_BlackHoleCpu);
        break;
    case GLFW_KEY_F6:
        abortBenchIfActive("Aborted: switched to F6 during benchmark run");
        simManager_.RequestMode(grav_sim::ModeId::F6_GravityBaselineCpu);
        break;
    case GLFW_KEY_F12:
        if ((mods & GLFW_MOD_CONTROL) != 0)
        {
            abortBenchIfActive("Aborted: switched to Diagnostic Triangle during benchmark run");
            simManager_.RequestMode(grav_sim::ModeId::Diagnostic_Triangle);
        }
        break;
    case GLFW_KEY_TAB:
        abortBenchIfActive("Aborted: cycled preset (Tab) during benchmark run");
        simManager_.RequestCyclePreset((mods & GLFW_MOD_SHIFT) != 0 ? -1 : 1);
        break;
    case GLFW_KEY_G:
        abortBenchIfActive("Aborted: toggled grid visibility (G) during benchmark run");
        simManager_.ToggleGridVisible();
        break;
    case GLFW_KEY_P:
        abortBenchIfActive("Aborted: toggled pause (P) during benchmark run");
        paused_ = !paused_;
        break;
    case GLFW_KEY_N:
        if (paused_)
        {
            abortBenchIfActive("Aborted: single-step (N) during benchmark run");
            singleStepRequested_ = true;
        }
        break;
    case GLFW_KEY_R:
        abortBenchIfActive("Aborted: reset preset runtime (R) during benchmark run");
        simManager_.RequestResetRuntime();
        break;
    case GLFW_KEY_C:
        abortBenchIfActive("Aborted: reset camera (C) during benchmark run");
        liveInput_.resetCameraRequested = true;
        break;
    default:
        break;
    }
}

void Application::HandleFramebufferResizeEvent(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        minimized_ = true;
        return;
    }
    minimized_ = false;
    framebufferWidth_ = width;
    framebufferHeight_ = height;
    pendingResize_ = true;
}

void Application::GlfwKeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (auto *app = static_cast<Application *>(glfwGetWindowUserPointer(window)))
    {
        app->HandleKeyAction(key, scancode, action, mods, false);
    }
}

void Application::GlfwCharCallback(GLFWwindow *window, unsigned int codepoint)
{
    ImGui_ImplGlfw_CharCallback(window, codepoint);
}

void Application::GlfwMouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    auto *app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    if (app == nullptr)
    {
        return;
    }

    const bool wantMouse =
        app->imguiInitialized_ ? ImGui::GetIO().WantCaptureMouse : false;

    if (action == GLFW_PRESS && !wantMouse)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            app->liveInput_.mouseLeftDown = true;
            app->liveInput_.mouseLeftJustPressed = true;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            app->liveInput_.mouseRightDown = true;
            app->liveInput_.mouseRightJustPressed = true;
        }
        else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            app->liveInput_.mouseMiddleDown = true;
            app->liveInput_.mouseMiddleJustPressed = true;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (app->liveInput_.mouseLeftDown)
            {
                app->liveInput_.mouseLeftJustReleased = true;
            }
            app->liveInput_.mouseLeftDown = false;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (app->liveInput_.mouseRightDown)
            {
                app->liveInput_.mouseRightJustReleased = true;
            }
            app->liveInput_.mouseRightDown = false;
        }
        else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            if (app->liveInput_.mouseMiddleDown)
            {
                app->liveInput_.mouseMiddleJustReleased = true;
            }
            app->liveInput_.mouseMiddleDown = false;
        }
    }
}

void Application::GlfwCursorPosCallback(GLFWwindow *window, double xpos, double ypos)
{
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    auto *app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    if (app == nullptr)
    {
        return;
    }

    if (!app->hasPrevCursor_)
    {
        app->prevCursorX_ = xpos;
        app->prevCursorY_ = ypos;
        app->hasPrevCursor_ = true;
    }

    const double dx = xpos - app->prevCursorX_;
    const double dy = ypos - app->prevCursorY_;
    app->prevCursorX_ = xpos;
    app->prevCursorY_ = ypos;

    app->liveInput_.mouseX = xpos;
    app->liveInput_.mouseY = ypos;

    const bool wantMouse =
        app->imguiInitialized_ ? ImGui::GetIO().WantCaptureMouse : false;
    if (!wantMouse || app->liveInput_.mouseLeftDown || app->liveInput_.mouseRightDown ||
        app->liveInput_.mouseMiddleDown)
    {
        app->liveInput_.mouseDeltaX += dx;
        app->liveInput_.mouseDeltaY += dy;
    }
}

void Application::GlfwScrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    auto *app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    if (app == nullptr)
    {
        return;
    }

    const bool wantMouse =
        app->imguiInitialized_ ? ImGui::GetIO().WantCaptureMouse : false;
    if (!wantMouse)
    {
        app->liveInput_.scrollDeltaY += yoffset;
    }
}

void Application::GlfwFramebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    if (auto *app = static_cast<Application *>(glfwGetWindowUserPointer(window)))
    {
        app->HandleFramebufferResizeEvent(width, height);
    }
}

void Application::GlfwWindowFocusCallback(GLFWwindow *window, int focused)
{
    ImGui_ImplGlfw_WindowFocusCallback(window, focused);
    auto *app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    if (app != nullptr && focused == GLFW_FALSE)
    {
        app->ClearHeldInputState();
        app->hasPrevCursor_ = false;
        app->liveInput_.focusLostThisFrame = true;
    }
}

void Application::RenderImGuiInterface()
{
    if (!imguiInitialized_)
    {
        return;
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (simManager_.State() == LifecycleState::Error)
    {
        DrawErrorBanner();
    }

    if (hudVisible_)
    {
        DrawCompactHudOverlay();
    }

    if (controlPanelVisible_)
    {
        DrawControlPanelWindow();
    }

    if (showHelpModal_)
    {
        DrawHelpModal();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::DrawErrorBanner()
{
    ImGui::SetNextWindowPos(ImVec2(framebufferWidth_ * 0.5f, 24.0f), ImGuiCond_Always,
                            ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.95f);
    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoCollapse;
    if (ImGui::Begin("Module Initialization Error", nullptr, flags))
    {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                           "Failed to load [%s] %s:",
                           grav_sim::ModeShortCode(simManager_.FailedMode()),
                           grav_sim::ModeDisplayName(simManager_.FailedMode()));
        ImGui::TextWrapped("%s", simManager_.LastErrorMessage().c_str());
        ImGui::Separator();
        ImGui::Text("Select a valid mode to recover:");
        for (std::uint8_t m = 0; m < static_cast<std::uint8_t>(grav_sim::ModeId::Count); ++m)
        {
            const auto mode = static_cast<grav_sim::ModeId>(m);
            if (m > 0)
            {
                ImGui::SameLine();
            }
            if (ImGui::Button(grav_sim::ModeShortCode(mode)))
            {
                simManager_.RequestMode(mode);
            }
        }
    }
    ImGui::End();
}

void Application::DrawCompactHudOverlay()
{
    ImGui::SetNextWindowPos(ImVec2(14.0f, 14.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(360.0f, 0.0f), ImVec2(480.0f, 800.0f));
    ImGui::SetNextWindowBgAlpha(0.84f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing;

    if (!ImGui::Begin("GravitySim Telemetry HUD [F7]", nullptr, flags))
    {
        ImGui::End();
        return;
    }

    const grav_sim::ModeId mode = simManager_.ActiveMode();
    ImGui::TextColored(ImVec4(0.45f, 0.80f, 1.0f, 1.0f), "[%s] %s",
                       grav_sim::ModeShortCode(mode), grav_sim::ModeDisplayName(mode));

    ImGui::SameLine();
    if (paused_)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.2f, 1.0f), "[PAUSED]");
    }
    else
    {
        ImGui::TextColored(ImVec4(0.35f, 0.90f, 0.45f, 1.0f), "[%s]",
                           LifecycleStateName(simManager_.State()));
    }

    if (const auto *mod = simManager_.ActiveModule())
    {
        const auto &preset = mod->ActivePreset();
        const auto &solver = mod->ActiveSolver();
        const auto stats = mod->RuntimeStats();

        ImGui::Text("Preset: %s", preset.displayName.c_str());
        ImGui::Text("Solver: %s", solver.id.c_str());
        ImGui::SameLine();
        ImGui::TextColored(SolverStatusBadgeColor(solver.status), "[%s]",
                           grav_sim::SolverStatusLabel(solver.status));
        ImGui::TextDisabled("Backend: %s", solver.backendName.c_str());

        ImGui::Separator();
        ImGui::Text("Presentation FPS: %.1f  (%.2f ms)", telemetry_.PresentationFps(),
                    telemetry_.LastFrameMs());

        if (telemetry_.ValidFrameSamples() > 1)
        {
            const auto &hist = telemetry_.FrameMsHistory();
            ImGui::PlotLines("##frame_ms_spark", hist.data(), static_cast<int>(hist.size()),
                             static_cast<int>(telemetry_.FrameHistoryOffset()),
                             "Frame Time (ms)", 0.0f, 50.0f, ImVec2(340.0f, 38.0f));
        }

        if (mode == grav_sim::ModeId::F3_BlackHoleGpu ||
            mode == grav_sim::ModeId::F4_FlatRayTracingCpu ||
            mode == grav_sim::ModeId::F5_BlackHoleCpu)
        {
            ImGui::Text("Image Evaluations/s: %.2f  |  Image Age: %.1f ms",
                        telemetry_.ImageEvaluationsPerSecond(), telemetry_.CurrentImageAgeMs());
        }
        else
        {
            ImGui::Text("Model Steps/s: %.1f  |  %s: %.3f", telemetry_.ModelStepsPerSecond(),
                        stats.paramLabel ? stats.paramLabel : "Param",
                        stats.simulationOrAffineParam);
        }

        ImGui::Text("%s: %zu  |  Display: %dx%d  |  Output: %dx%d",
                    stats.primaryLabel ? stats.primaryLabel : "Count", stats.primaryCount,
                    framebufferWidth_, framebufferHeight_,
                    stats.outputWidth > 0 ? stats.outputWidth : framebufferWidth_,
                    stats.outputHeight > 0 ? stats.outputHeight : framebufferHeight_);

        ImGui::Text("Grid [G]: %s (%d div, %zu src)%s  |  VSync: %s",
                    simManager_.IsGridVisible() ? "ON" : "OFF", sharedGrid_.Config().divisions,
                    sharedGrid_.ActiveSourceCount(),
                    (simManager_.IsGridVisible() && stats.gridOverlayDiagnosticOnly)
                        ? " [Diagnostic Overlay]"
                        : "",
                    vsyncEnabled_ ? "ON" : "OFF");

        if (stats.hasAsyncWorker && stats.workerBusy)
        {
            char overlay[64];
            std::snprintf(overlay, sizeof(overlay), "CPU Worker (%u threads): %.0f%%",
                          stats.workerThreadCount, stats.workerProgress * 100.0f);
            ImGui::ProgressBar(stats.workerProgress, ImVec2(340.0f, 16.0f), overlay);
        }
    }

    ImGui::Separator();
    const auto &sys = telemetry_.SystemMetrics();
    const auto censusSnap = resourceCensus_.Snapshot(&gpuMemoryManager);

    ImGui::Text("Proc CPU: %s  |  RAM WS: %.1f MB (Priv: %.1f MB)",
                sys.cpuAvailable ? (std::to_string(static_cast<int>(std::round(sys.processCpuPercent))) + "%").c_str()
                                 : "N/A",
                static_cast<double>(sys.workingSetBytes) / (1024.0 * 1024.0),
                static_cast<double>(sys.privateUsageBytes) / (1024.0 * 1024.0));

    if (sys.dxgiAvailable)
    {
        ImGui::Text("App GPU Alloc: %.2f MB  |  DXGI VRAM: %.0f / %.0f MB",
                    static_cast<double>(censusSnap.totalAllocatedBytes) / (1024.0 * 1024.0),
                    static_cast<double>(sys.dxgiUsageBytes) / (1024.0 * 1024.0),
                    static_cast<double>(sys.dxgiBudgetBytes) / (1024.0 * 1024.0));
    }
    else
    {
        ImGui::Text("App GPU Alloc: %.2f MB  |  DXGI VRAM: N/A",
                    static_cast<double>(censusSnap.totalAllocatedBytes) / (1024.0 * 1024.0));
    }

    if (benchmark_.IsActive())
    {
        ImGui::Separator();
        char benchBuf[96];
        std::snprintf(benchBuf, sizeof(benchBuf), "BENCHMARK [%s] %.0f%% (F9 Abort)",
                      grav_bench::BenchmarkStateName(benchmark_.State()),
                      benchmark_.OverallProgress() * 100.0f);
        ImGui::ProgressBar(benchmark_.OverallProgress(), ImVec2(340.0f, 18.0f), benchBuf);
    }

    ImGui::TextDisabled("F1..F6: Modes | Tab: Preset | P: Pause | F8: Help | F9: Benchmark");
    ImGui::End();
}

void Application::DrawControlPanelWindow()
{
    ImGui::SetNextWindowPos(ImVec2(static_cast<float>(std::max(framebufferWidth_ - 475, 20)), 14.0f),
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(460.0f, 640.0f), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Simulation & Benchmark Console", &controlPanelVisible_))
    {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("MainConsoleTabs"))
    {
        if (ImGui::BeginTabItem("Interactive"))
        {
            DrawInteractiveTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Benchmark"))
        {
            DrawBenchmarkTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Diagnostics"))
        {
            DrawDiagnosticsTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void Application::DrawInteractiveTab()
{
    const bool locked = benchmark_.IsConfigLocked();
    if (locked)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f),
                           "Benchmark active: configuration locked to prevent mixed samples.");
    }

    ImGui::BeginDisabled(locked);

    ImGui::Text("Simulation Mode (F1-F6, Ctrl+F12):");
    for (std::uint8_t m = 0; m < static_cast<std::uint8_t>(grav_sim::ModeId::Count); ++m)
    {
        const auto modeId = static_cast<grav_sim::ModeId>(m);
        if (m > 0 && m != 4)
        {
            ImGui::SameLine();
        }
        const bool selected = (simManager_.ActiveMode() == modeId);
        if (selected)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.24f, 0.48f, 0.74f, 1.0f));
        }
        std::string btnLabel = std::string(grav_sim::ModeShortCode(modeId));
        if (ImGui::Button(btnLabel.c_str()))
        {
            simManager_.RequestMode(modeId);
        }
        if (selected)
        {
            ImGui::PopStyleColor();
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("%s\nProvenance: %s", grav_sim::ModeDisplayName(modeId),
                              grav_sim::ModeProvenanceSummary(modeId));
        }
    }

    ImGui::Separator();

    if (auto *mod = simManager_.ActiveModule())
    {
        const auto &presets = mod->Presets();
        const std::size_t activeIdx = mod->ActivePresetIndex();
        if (!presets.empty())
        {
            if (ImGui::BeginCombo("Preset [Tab]", presets[activeIdx].displayName.c_str()))
            {
                for (std::size_t i = 0; i < presets.size(); ++i)
                {
                    const bool isSel = (i == activeIdx);
                    if (ImGui::Selectable(presets[i].displayName.c_str(), isSel))
                    {
                        simManager_.RequestPresetIndex(i);
                    }
                    if (isSel)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::TextDisabled("Provenance: %s", presets[activeIdx].provenance.c_str());
            ImGui::TextWrapped("%s", presets[activeIdx].description.c_str());
        }

        const auto solvers = mod->AvailableSolvers();
        const auto &curSolver = mod->ActiveSolver();
        if (solvers.size() > 1)
        {
            if (ImGui::BeginCombo("Solver Version", curSolver.displayName.c_str()))
            {
                for (const auto &s : solvers)
                {
                    const bool isSel = (s.id == curSolver.id);
                    std::string label =
                        s.displayName + " [" + grav_sim::SolverStatusLabel(s.status) + "]";
                    if (ImGui::Selectable(label.c_str(), isSel))
                    {
                        mod->SetActiveSolverById(s.id);
                    }
                }
                ImGui::EndCombo();
            }
        }
        else
        {
            ImGui::Text("Solver: %s", curSolver.displayName.c_str());
        }
        ImGui::TextColored(SolverStatusBadgeColor(curSolver.status), "Validation Status: %s",
                           grav_sim::SolverStatusLabel(curSolver.status));
        ImGui::TextWrapped("%s", curSolver.notes.c_str());

        ImGui::Separator();

        if (ImGui::Button(paused_ ? "Resume [P]" : "Pause [P]"))
        {
            paused_ = !paused_;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!paused_);
        if (ImGui::Button("Step [N]"))
        {
            singleStepRequested_ = true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Reset [R]"))
        {
            simManager_.RequestResetRuntime();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Cam [C]"))
        {
            simManager_.ResetActiveCamera();
        }

        bool gridVis = simManager_.IsGridVisible();
        if (ImGui::Checkbox("Shared 3D Grid [G]", &gridVis))
        {
            simManager_.SetGridVisible(gridVis);
        }
        ImGui::SameLine();
        bool vsync = vsyncEnabled_;
        if (ImGui::Checkbox("VSync", &vsync))
        {
            ApplyVsync(vsync);
        }
        ImGui::SameLine();
        if (ImGui::Button("Shortcuts [F8]"))
        {
            showHelpModal_ = true;
        }

        const auto stats = mod->RuntimeStats();
        if (stats.usesSharedBlackHole)
        {
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Shared Black Hole Parameters (F2 / F3 / F5)",
                                        ImGuiTreeNodeFlags_DefaultOpen))
            {
                float massMultiple = static_cast<float>(sharedBlackHole_.massKg /
                                                        grav_models::kSagittariusAMassKg);
                if (ImGui::SliderFloat("Mass (x SagA*)", &massMultiple, 0.20f, 3.00f, "%.3f x"))
                {
                    sharedBlackHole_.SetMassKg(static_cast<double>(massMultiple) *
                                               grav_models::kSagittariusAMassKg);
                }
                if (ImGui::Button("Reset Sagittarius A* Mass (8.54e36 kg)"))
                {
                    sharedBlackHole_.SetMassKg(grav_models::kSagittariusAMassKg);
                }
                ImGui::Text("Mass: %.4e kg  |  rs = 2GM/c^2: %.4e m (%.2f Gm)",
                            sharedBlackHole_.massKg, sharedBlackHole_.schwarzschildRadiusMeters,
                            sharedBlackHole_.schwarzschildRadiusMeters * 1.0e-9);
                ImGui::TextDisabled("Revision #%llu (synchronized across F2, F3, F5 & SharedGrid)",
                                    static_cast<unsigned long long>(sharedBlackHole_.revision));
            }
        }

        ImGui::EndDisabled(); // unlock before module draws its own lock-aware widgets

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.65f, 0.85f, 1.0f, 1.0f), "Mode-Specific Controls:");
        mod->DrawInteractiveControls(locked);
    }
    else
    {
        ImGui::EndDisabled();
        ImGui::TextDisabled("No active simulation module.");
    }
}

void Application::DrawBenchmarkTab()
{
    auto &cfg = benchmark_.MutableConfig();
    const bool active = benchmark_.IsActive();

    ImGui::TextWrapped("Deterministic Workload Benchmark Runner (Section 13.2). "
                       "Warmup and mode-switching durations are excluded from steady-state "
                       "statistics. Any mode, preset, solver, resolution, or camera change "
                       "during an active run aborts the run to prevent mixed-workload metrics.");
    ImGui::Separator();

    ImGui::BeginDisabled(active);
    ImGui::SliderInt("Warmup Frames", &cfg.warmupFrames, 0, 300);
    ImGui::SliderInt("Measured Frames", &cfg.measuredFrames, 30, 1200);
    ImGui::Checkbox("Force VSync OFF During Run", &cfg.disableVsyncDuringRun);
    ImGui::Checkbox("Force Recompute Static Ray Image Every Frame",
                    &cfg.forceRecomputeEachIteration);
    ImGui::Checkbox("Hide HUD Overlay During Run", &cfg.hideHudDuringRun);
    ImGui::Checkbox("Auto-Export CSV + JSON Manifest on Completion", &cfg.autoExportOnComplete);
    ImGui::EndDisabled();

    ImGui::Separator();
    if (!active)
    {
        if (ImGui::Button("Start Benchmark Run [F9]", ImVec2(220.0f, 30.0f)))
        {
            StartBenchmarkRun();
        }
    }
    else
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.22f, 0.22f, 1.0f));
        if (ImGui::Button("Abort Benchmark & Unlock [F9]", ImVec2(240.0f, 30.0f)))
        {
            AbortBenchmarkRun("Aborted by user from Benchmark Console");
        }
        ImGui::PopStyleColor();
    }

    ImGui::Text("Run State: %s  |  Warmup: %d/%d  |  Measured: %d/%d",
                grav_bench::BenchmarkStateName(benchmark_.State()),
                benchmark_.WarmupFramesCompleted(), cfg.warmupFrames,
                benchmark_.MeasuredFramesCompleted(), cfg.measuredFrames);
    ImGui::ProgressBar(benchmark_.OverallProgress(), ImVec2(-1.0f, 18.0f));

    if (benchmark_.State() == grav_bench::BenchmarkState::Aborted)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.40f, 0.40f, 1.0f), "Run Aborted: %s",
                           benchmark_.AbortReason().c_str());
    }

    if (benchmark_.State() == grav_bench::BenchmarkState::Completed)
    {
        const auto &sum = benchmark_.Summary();
        const auto &fp = benchmark_.LockedFingerprint();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.35f, 0.92f, 0.50f, 1.0f),
                           "Completed Summary: [%s] %s (%s)", fp.modeCode.c_str(),
                           fp.presetId.c_str(), fp.solverId.c_str());
        ImGui::Text("Presentation FPS (from mean): %.2f FPS (%zu samples)", sum.presentationFps,
                    sum.sampleCount);
        ImGui::Text("Frame ms — Mean: %.3f | Median: %.3f | P95: %.3f | P99: %.3f",
                    sum.meanFrameMs, sum.medianFrameMs, sum.p95FrameMs, sum.p99FrameMs);
        ImGui::Text("Frame ms — Min: %.3f | Max: %.3f | Load/Switch: %.2f ms", sum.minFrameMs,
                    sum.maxFrameMs, sum.lastModeLoadOrSwitchMs);
        ImGui::Text("Throughput — Steps/s: %.1f | Image Evals/s: %.2f",
                    sum.meanModelStepsPerSec, sum.meanImageEvaluationsPerSec);
        ImGui::Text("Run Peak App GPU Allocation: %.2f MB",
                    static_cast<double>(sum.runPeakAllocatedBytes) / (1024.0 * 1024.0));
    }

    if (benchmark_.State() == grav_bench::BenchmarkState::Completed ||
        benchmark_.State() == grav_bench::BenchmarkState::Aborted)
    {
        if (ImGui::Button("Export CSV & JSON Manifest Now"))
        {
            benchmark_.ExportResults();
        }
        if (!benchmark_.LastManifestPath().empty())
        {
            ImGui::TextWrapped("CSV: %s", benchmark_.LastCsvPath().c_str());
            ImGui::TextWrapped("Manifest: %s", benchmark_.LastManifestPath().c_str());
        }
    }
}

void Application::DrawDiagnosticsTab()
{
    const auto &pt = telemetry_.PassTimings();
    const auto &sys = telemetry_.SystemMetrics();
    const auto snap = resourceCensus_.Snapshot(&gpuMemoryManager);

    if (ImGui::CollapsingHeader("CPU & Delayed GPU Pass Timings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("CPU Update Submit: %.3f ms  |  CPU Render Submit: %.3f ms", pt.cpuUpdateMs,
                    pt.cpuRenderSubmitMs);
        ImGui::Text("CPU Quiesce Fence Poll: %.3f ms  |  CPU UI Submit: %.3f ms",
                    pt.cpuFenceWaitMs, pt.cpuUiMs);
        ImGui::Text("Last Mode Load/Switch Time: %.2f ms  |  Scene Generation: #%llu",
                    simManager_.LastTransitionDurationMs(),
                    static_cast<unsigned long long>(simManager_.SceneGeneration()));
        ImGui::Separator();
        ImGui::TextDisabled("Delayed GPU GL_TIME_ELAPSED Ring (Generation-Tagged, Non-Blocking):");
        for (std::size_t i = 0; i < grav_telemetry::TelemetryService::kPassCount; ++i)
        {
            const auto pass = static_cast<grav_telemetry::GpuPassId>(i);
            if (pt.gpuPassValid[i])
            {
                ImGui::Text("  %-20s : %.3f ms", grav_telemetry::GpuPassName(pass),
                            pt.gpuPassMs[i]);
            }
            else
            {
                ImGui::TextDisabled("  %-20s : inactive", grav_telemetry::GpuPassName(pass));
            }
        }
    }

    if (ImGui::CollapsingHeader("OS Process & Device Providers", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float hz = static_cast<float>(telemetry_.OsSamplingHz());
        if (ImGui::SliderFloat("OS/DXGI Sampling Rate (Hz)", &hz, 0.5f, 10.0f, "%.1f Hz"))
        {
            telemetry_.SetOsSamplingHz(static_cast<double>(hz));
        }

        const auto &env = benchmark_.EnvironmentInfo();
        ImGui::Text("GL Renderer: %s", env.glRenderer.c_str());
        ImGui::Text("GL Version : %s", env.glVersion.c_str());
        ImGui::Text("Process CPU: %s (%u logical CPUs)",
                    sys.cpuAvailable ? (std::to_string(sys.processCpuPercent) + "%").c_str() : "N/A",
                    sys.logicalCpuCount);
        ImGui::Text("Process RAM: Working Set %.2f MB | Private %.2f MB",
                    static_cast<double>(sys.workingSetBytes) / (1024.0 * 1024.0),
                    static_cast<double>(sys.privateUsageBytes) / (1024.0 * 1024.0));
        if (sys.dxgiAvailable)
        {
            ImGui::Text("DXGI Adapter: %s", sys.dxgiAdapterName.c_str());
            ImGui::Text("DXGI Local Video Memory: %.2f MB / Budget %.2f MB",
                        static_cast<double>(sys.dxgiUsageBytes) / (1024.0 * 1024.0),
                        static_cast<double>(sys.dxgiBudgetBytes) / (1024.0 * 1024.0));
        }
        else
        {
            ImGui::TextDisabled("DXGI Local Video Memory: N/A");
        }
        ImGui::TextDisabled("Device GPU Utilization / Temp / Power: N/A (vendor SDK optional)");
    }

    if (ImGui::CollapsingHeader("Application GPU Resource Census", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const auto gridH = sharedGrid_.Handles();
        ImGui::Text("Live Objects: %zu (SSBO:%zu Tex:%zu UBO:%zu Geom:%zu VAO:%zu Prog:%zu)",
                    snap.totalLiveObjects, snap.ssboCount, snap.textureCount, snap.uboCount,
                    snap.geometryBufferCount, snap.vertexArrayCount, snap.shaderProgramCount);
        ImGui::Text("Allocated — Total: %.2f KB | Shared: %.2f KB | Active Scene: %.2f KB",
                    static_cast<double>(snap.totalAllocatedBytes) / 1024.0,
                    static_cast<double>(snap.sharedAllocatedBytes) / 1024.0,
                    static_cast<double>(snap.sceneAllocatedBytes) / 1024.0);
        ImGui::Text("Peaks — App Peak: %.2f KB | Scene Peak: %.2f KB | Run Peak: %.2f KB",
                    static_cast<double>(snap.appPeakAllocatedBytes) / 1024.0,
                    static_cast<double>(snap.scenePeakAllocatedBytes) / 1024.0,
                    static_cast<double>(snap.runPeakAllocatedBytes) / 1024.0);
        ImGui::Text("SharedGrid Handles — Base:%u Deformed:%u Src:%u VAO:%u EBO:%u",
                    gridH.baseGridBuffer, gridH.deformedGridBuffer, gridH.gridSourcesBuffer,
                    gridH.gridVAO, gridH.gridEBO);
        ImGui::Text("SharedGrid Dispatches — Deform:%llu | Materialize:%llu | Cached Shaders:%zu",
                    static_cast<unsigned long long>(sharedGrid_.ComputeDispatchCount()),
                    static_cast<unsigned long long>(sharedGrid_.MaterializationDispatchCount()),
                    shaderCache_.CachedProgramCount());

        if (ImGui::TreeNode("Detailed Live Resource Ledger"))
        {
            for (const auto &r : snap.entries)
            {
                ImGui::Text("[%s] %s #%u (%s) — %.1f KB (gen %llu)", r.ownerTag.c_str(),
                            grav_render::ResourceCategoryName(r.category), r.glHandle,
                            r.label.c_str(), static_cast<double>(r.allocatedBytes) / 1024.0,
                            static_cast<unsigned long long>(r.sceneGeneration));
            }
            ImGui::TreePop();
        }
    }
}

void Application::DrawHelpModal()
{
    ImGui::SetNextWindowPos(ImVec2(framebufferWidth_ * 0.5f, framebufferHeight_ * 0.5f),
                            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(580.0f, 440.0f), ImGuiCond_Appearing);

    if (ImGui::Begin("Keyboard Shortcuts & Controls Reference [F8]", &showHelpModal_))
    {
        ImGui::TextColored(ImVec4(0.45f, 0.82f, 1.0f, 1.0f), "Global Application Shortcuts");
        ImGui::BulletText("F1 : Gravity N-Body Simulation (GPU Compute)");
        ImGui::BulletText("F2 : Black Hole Light Trajectories (Planar Geodesics, 2D/3D)");
        ImGui::BulletText("F3 : Black Hole Lensing Image (GPU Compute, RK4 & Legacy)");
        ImGui::BulletText("F4 : Flat-Space Ray Tracing (CPU Multi-Thread Worker)");
        ImGui::BulletText("F5 : Black Hole Lensing Image (CPU Geodesics & Silhouette)");
        ImGui::BulletText("F6 : Historical CPU Gravity Baseline Adapter");
        ImGui::BulletText("Ctrl + F12 : Graphics Context Diagnostic (Orange Triangle)");
        ImGui::Separator();
        ImGui::BulletText("Tab / Shift+Tab : Cycle presets within active mode");
        ImGui::BulletText("G : Toggle shared 3D deformed spacetime grid");
        ImGui::BulletText("P : Pause / Resume simulation (camera & UI remain active)");
        ImGui::BulletText("N : Advance 1 single simulation step while paused");
        ImGui::BulletText("R : Reset current preset runtime state");
        ImGui::BulletText("C : Reset active mode camera");
        ImGui::BulletText("F7 : Toggle Telemetry HUD overlay");
        ImGui::BulletText("F8 : Toggle this Help window");
        ImGui::BulletText("F9 : Start / Abort Benchmark run");
        ImGui::BulletText("F11 : Toggle Fullscreen / Windowed");
        ImGui::BulletText("Esc : Close Help window / release held input");
        ImGui::BulletText("Q : Quit application cleanly");

        if (const auto *mod = simManager_.ActiveModule())
        {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.45f, 0.82f, 1.0f, 1.0f), "Active Mode Controls (%s):",
                               grav_sim::ModeShortCode(simManager_.ActiveMode()));
            ImGui::TextWrapped("%s", mod->ControlsHelpText().c_str());
        }
    }
    ImGui::End();
}

} // namespace grav_app
