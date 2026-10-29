#include "app/simulation_manager.hpp"

#include "models/black_hole/black_hole_cpu_module.hpp"
#include "models/black_hole/black_hole_gpu_module.hpp"
#include "models/diagnostic/diagnostic_triangle_module.hpp"
#include "models/gravity/gravity_gpu_module.hpp"
#include "models/gravity_baseline/gravity_baseline_module.hpp"
#include "models/light_trajectories/light_trajectories_module.hpp"
#include "models/ray_tracing/flat_ray_tracing_module.hpp"
#include <GL/glew.h>
#include <algorithm>
#include <stdexcept>

namespace grav_app
{

const char *LifecycleStateName(LifecycleState state)
{
    switch (state)
    {
    case LifecycleState::Uninitialized:
        return "Uninitialized";
    case LifecycleState::Running:
        return "Running";
    case LifecycleState::Quiescing:
        return "Quiescing";
    case LifecycleState::Releasing:
        return "Releasing";
    case LifecycleState::Loading:
        return "Loading";
    case LifecycleState::Warmup:
        return "Warmup";
    case LifecycleState::Error:
        return "Error";
    }
    return "Unknown";
}

SimulationManager::SimulationManager() = default;

SimulationManager::~SimulationManager() { Shutdown(); }

std::unique_ptr<grav_sim::ISimulationModule>
SimulationManager::DefaultCreateModule(grav_sim::ModeId mode)
{
    switch (mode)
    {
    case grav_sim::ModeId::F1_GravityGpu:
        return std::make_unique<grav_models::GravityGpuModule>();
    case grav_sim::ModeId::F2_LightTrajectories:
        return std::make_unique<grav_models::LightTrajectoriesModule>();
    case grav_sim::ModeId::F3_BlackHoleGpu:
        return std::make_unique<grav_models::BlackHoleGpuModule>();
    case grav_sim::ModeId::F4_FlatRayTracingCpu:
        return std::make_unique<grav_models::FlatRayTracingModule>();
    case grav_sim::ModeId::F5_BlackHoleCpu:
        return std::make_unique<grav_models::BlackHoleCpuModule>();
    case grav_sim::ModeId::F6_GravityBaselineCpu:
        return std::make_unique<grav_models::GravityBaselineModule>();
    case grav_sim::ModeId::Diagnostic_Triangle:
        return std::make_unique<grav_models::DiagnosticTriangleModule>();
    case grav_sim::ModeId::Count:
        break;
    }
    throw std::runtime_error("Unsupported ModeId requested in DefaultCreateModule");
}

void SimulationManager::Initialize(const grav_sim::ModuleServices &services,
                                   grav_sim::ModeId initialMode, std::size_t initialPresetIndex)
{
    services_ = services;
    sceneGeneration_ = (services.sceneGeneration > 0) ? services.sceneGeneration : 1;
    services_.sceneGeneration = sceneGeneration_;

    pending_.active = true;
    pending_.targetMode = initialMode;
    pending_.targetPresetIndex = initialPresetIndex;
    pending_.useRememberedPreset = false;
    pending_.forceReloadSameMode = true;

    transitionStartWall_ = std::chrono::steady_clock::now();
    state_ = LifecycleState::Loading;
    PerformLoad();
}

void SimulationManager::Shutdown()
{
    if (activeModule_)
    {
        activeModule_->BeginExit();
        // Wait for CPU workers to reach cancellation checkpoint safely.
        const auto waitStart = std::chrono::steady_clock::now();
        while (!activeModule_->IsReadyToRelease())
        {
            const double waitedSec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - waitStart).count();
            if (waitedSec > 5.0)
            {
                break;
            }
        }
        if (services_.gpuMemoryManager != nullptr)
        {
            services_.gpuMemoryManager->WaitForGpuCompletion();
        }
        activeModule_->ReleaseResources();
        activeModule_.reset();
        ResetCommonGlBindings();
    }
    pending_.active = false;
    state_ = LifecycleState::Uninitialized;
}

void SimulationManager::RequestMode(grav_sim::ModeId targetMode)
{
    if (targetMode >= grav_sim::ModeId::Count)
    {
        return;
    }

    // Pressing the already-active F-key in steady Running state with no pending switch is a no-op (T05).
    if (!pending_.active && state_ == LifecycleState::Running && activeModule_ != nullptr &&
        activeMode_ == targetMode)
    {
        return;
    }

    // If a transition is pending back to the currently running mode before quiescing started, cancel it.
    if (state_ == LifecycleState::Running && activeModule_ != nullptr && activeMode_ == targetMode)
    {
        pending_.active = false;
        return;
    }

    pending_.active = true;
    pending_.targetMode = targetMode;
    pending_.useRememberedPreset = true;
    pending_.forceReloadSameMode = false;
}

void SimulationManager::RequestModeAndPreset(grav_sim::ModeId targetMode, std::size_t presetIndex)
{
    if (targetMode >= grav_sim::ModeId::Count)
    {
        return;
    }

    pending_.active = true;
    pending_.targetMode = targetMode;
    pending_.targetPresetIndex = presetIndex;
    pending_.useRememberedPreset = false;
    pending_.forceReloadSameMode = true;
}

void SimulationManager::RequestPresetIndex(std::size_t presetIndex)
{
    const grav_sim::ModeId baseMode = pending_.active ? pending_.targetMode : activeMode_;
    RequestModeAndPreset(baseMode, presetIndex);
}

void SimulationManager::RequestCyclePreset(int delta)
{
    if (!activeModule_ || activeModule_->Presets().empty())
    {
        return;
    }
    const std::size_t count = activeModule_->Presets().size();
    const std::size_t current =
        (pending_.active && !pending_.useRememberedPreset && pending_.targetMode == activeMode_)
            ? (pending_.targetPresetIndex % count)
            : activeModule_->ActivePresetIndex();
    const int wrapped =
        (static_cast<int>(current) + (delta % static_cast<int>(count)) + static_cast<int>(count)) %
        static_cast<int>(count);
    RequestModeAndPreset(activeMode_, static_cast<std::size_t>(wrapped));
}

void SimulationManager::RequestResetRuntime()
{
    if (activeModule_)
    {
        RequestModeAndPreset(activeMode_, activeModule_->ActivePresetIndex());
    }
    else if (state_ == LifecycleState::Error)
    {
        RequestMode(failedMode_);
    }
}

void SimulationManager::InjectNextLoadFailure(const std::string &errorMessage)
{
    injectFailureNextLoad_ = true;
    injectedFailureMessage_ =
        errorMessage.empty() ? "Injected module initialization failure" : errorMessage;
}

void SimulationManager::BeginQuiescing()
{
    transitionStartWall_ = std::chrono::steady_clock::now();
    quiesceStartWall_ = transitionStartWall_;

    if (activeModule_)
    {
        activeModule_->BeginExit();
    }
    if (services_.gpuMemoryManager != nullptr)
    {
        services_.gpuMemoryManager->CreateFreshFence();
    }
    state_ = LifecycleState::Quiescing;
}

void SimulationManager::PerformRelease()
{
    state_ = LifecycleState::Releasing;
    if (activeModule_)
    {
        activeModule_->ReleaseResources();
        activeModule_.reset();
    }

    ResetCommonGlBindings();

    ++sceneGeneration_;
    services_.sceneGeneration = sceneGeneration_;

    if (services_.gpuMemoryManager != nullptr)
    {
        services_.gpuMemoryManager->ResetScenePeakAllocation();
    }
    if (services_.resourceCensus != nullptr)
    {
        services_.resourceCensus->ResetScenePeak(services_.gpuMemoryManager);
    }

    state_ = LifecycleState::Loading;
}

void SimulationManager::PerformLoad()
{
    const PendingRequest req = pending_;
    pending_.active = false;

    const grav_sim::ModeId targetMode = req.targetMode;
    const auto modeIdx = static_cast<std::size_t>(targetMode);
    std::size_t targetPreset =
        req.useRememberedPreset ? rememberedPresets_[modeIdx] : req.targetPresetIndex;

    const bool modeChanging = (activeMode_ != targetMode) || (completedLoadCount_ == 0);

    std::unique_ptr<grav_sim::ISimulationModule> candidate;
    try
    {
        candidate = customFactory_ ? customFactory_(targetMode) : DefaultCreateModule(targetMode);
        if (!candidate)
        {
            throw std::runtime_error("Module factory returned null module instance");
        }

        if (!candidate->Presets().empty() && targetPreset >= candidate->Presets().size())
        {
            targetPreset = 0;
        }

        if (injectFailureNextLoad_)
        {
            const std::string msg = injectedFailureMessage_;
            injectFailureNextLoad_ = false;
            injectedFailureMessage_.clear();
            // Exercise partial-init cleanup path even when fault is injected after module creation.
            throw std::runtime_error(msg);
        }

        services_.sceneGeneration = sceneGeneration_;
        candidate->OnEnter(services_, targetPreset);

        activeModule_ = std::move(candidate);
        activeMode_ = targetMode;
        rememberedPresets_[modeIdx] = activeModule_->ActivePresetIndex();
        if (modeChanging)
        {
            gridVisible_ = activeModule_->DefaultGridVisible();
        }

        lastErrorMessage_.clear();
        ++completedLoadCount_;

        const auto now = std::chrono::steady_clock::now();
        lastTransitionDurationMs_ =
            std::chrono::duration<double, std::milli>(now - transitionStartWall_).count();

        warmupFramesRemaining_ = warmupFrameTarget_;
        if (warmupFramesRemaining_ > 0)
        {
            state_ = LifecycleState::Warmup;
        }
        else
        {
            if (services_.telemetry != nullptr)
            {
                services_.telemetry->OnSceneGenerationChanged(sceneGeneration_);
            }
            state_ = LifecycleState::Running;
        }
    }
    catch (const std::exception &ex)
    {
        if (candidate)
        {
            candidate->BeginExit();
            candidate->ReleaseResources();
            candidate.reset();
        }
        ResetCommonGlBindings();
        failedMode_ = targetMode;
        lastErrorMessage_ = ex.what();
        state_ = LifecycleState::Error;
    }
    catch (...)
    {
        if (candidate)
        {
            candidate->BeginExit();
            candidate->ReleaseResources();
            candidate.reset();
        }
        ResetCommonGlBindings();
        failedMode_ = targetMode;
        lastErrorMessage_ = "Unknown exception during module initialization";
        state_ = LifecycleState::Error;
    }
}

void SimulationManager::TickLifecycle()
{
    if (state_ == LifecycleState::Running)
    {
        if (pending_.active)
        {
            BeginQuiescing();
        }
        else
        {
            return;
        }
    }

    if (state_ == LifecycleState::Error)
    {
        if (pending_.active)
        {
            transitionStartWall_ = std::chrono::steady_clock::now();
            PerformRelease();
            PerformLoad();
        }
        return;
    }

    if (state_ == LifecycleState::Quiescing)
    {
        const auto pollStart = std::chrono::steady_clock::now();
        FencePollStatus fenceStatus = FencePollStatus::NoFence;
        if (services_.gpuMemoryManager != nullptr)
        {
            fenceStatus = services_.gpuMemoryManager->PollGpuCompletion();
        }
        const bool moduleReady = (!activeModule_) || activeModule_->IsReadyToRelease();
        const auto pollEnd = std::chrono::steady_clock::now();

        if (services_.telemetry != nullptr)
        {
            services_.telemetry->AddCpuFenceWaitMs(
                std::chrono::duration<double, std::milli>(pollEnd - pollStart).count());
        }

        const double quiesceElapsedSec =
            std::chrono::duration<double>(pollEnd - quiesceStartWall_).count();

        if (fenceStatus == FencePollStatus::Pending && quiesceElapsedSec > 3.0 &&
            services_.gpuMemoryManager != nullptr)
        {
            // Safety fallback if a driver fence fails to signal promptly.
            services_.gpuMemoryManager->WaitForGpuCompletion();
            fenceStatus = FencePollStatus::Signaled;
        }

        if ((fenceStatus == FencePollStatus::Signaled || fenceStatus == FencePollStatus::NoFence ||
             fenceStatus == FencePollStatus::Failed) &&
            moduleReady)
        {
            PerformRelease();
            PerformLoad();
        }
        return;
    }

    if (state_ == LifecycleState::Releasing)
    {
        PerformRelease();
        PerformLoad();
        return;
    }

    if (state_ == LifecycleState::Loading)
    {
        PerformLoad();
        return;
    }

    if (state_ == LifecycleState::Warmup)
    {
        if (pending_.active)
        {
            BeginQuiescing();
            return;
        }

        if (warmupFramesRemaining_ > 0)
        {
            --warmupFramesRemaining_;
        }
        if (warmupFramesRemaining_ <= 0)
        {
            if (services_.telemetry != nullptr)
            {
                services_.telemetry->OnSceneGenerationChanged(sceneGeneration_);
            }
            state_ = LifecycleState::Running;
        }
    }
}

void SimulationManager::HandleInput(const grav_sim::ModuleInputState &input,
                                    double wallDeltaSeconds)
{
    if ((state_ == LifecycleState::Running || state_ == LifecycleState::Warmup) && activeModule_)
    {
        if (input.resetCameraRequested)
        {
            activeModule_->ResetCamera();
        }
        activeModule_->HandleInput(input, wallDeltaSeconds);
    }
}

void SimulationManager::UpdateActive(const grav_sim::StepContext &step)
{
    if ((state_ == LifecycleState::Running || state_ == LifecycleState::Warmup) && activeModule_)
    {
        grav_sim::StepContext localStep = step;
        localStep.sceneGeneration = sceneGeneration_;
        localStep.gridVisible = gridVisible_;
        activeModule_->Update(localStep);
    }
}

void SimulationManager::RenderActive(const grav_sim::RenderContext &render)
{
    if ((state_ == LifecycleState::Running || state_ == LifecycleState::Warmup) && activeModule_)
    {
        grav_sim::RenderContext localRender = render;
        localRender.sceneGeneration = sceneGeneration_;
        localRender.gridVisible = gridVisible_;
        activeModule_->Render(localRender);
    }
}

void SimulationManager::OnResize(int framebufferWidth, int framebufferHeight)
{
    if (framebufferWidth <= 0 || framebufferHeight <= 0)
    {
        return;
    }
    services_.framebufferWidth = framebufferWidth;
    services_.framebufferHeight = framebufferHeight;
    if (activeModule_)
    {
        activeModule_->OnResize(framebufferWidth, framebufferHeight);
    }
}

void SimulationManager::ResetActiveCamera()
{
    if (activeModule_)
    {
        activeModule_->ResetCamera();
    }
}

void SimulationManager::ResetCommonGlBindings()
{
    glUseProgram(0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    for (GLuint uboSlot = 1; uboSlot <= 3; ++uboSlot)
    {
        glBindBufferBase(GL_UNIFORM_BUFFER, uboSlot, 0);
    }
}

} // namespace grav_app

