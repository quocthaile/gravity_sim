#include "models/gravity/gravity_gpu_module.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <imgui.h>
#include <stdexcept>

namespace grav_models
{

GravityGpuModule::GravityGpuModule()
{
    presets_ = {
        {
            "f1_default_2_plus_200_seed123",
            "Default: Earth-Moon + 200 Random Orbiters (seed 123)",
            "src/gravity_sim_3Dgrid.cpp + CreateObjects({}, 200)",
            "Authoritative GPU snapshot N-body preset matching baseline commit a5f9f93.",
            1,
        },
        {
            "f1_two_body_earth_moon",
            "Two-Body: Earth-Moon (randomObjectCount = 0)",
            "src/gravity_sim_3Dgrid_function.cpp: CreateObjects({}, 0)",
            "Deterministic 2-body configuration without random orbiter cloud.",
            1,
        },
        {
            "f1_legacy_three_body_ic",
            "Legacy Initial Conditions: Three-Body (gravity_sim.cpp)",
            "src/legacy/gravity_sim.cpp:227-232 (initial conditions only)",
            "Three-body initial state evaluated under the GPU snapshot N-body solver.",
            1,
        },
        {
            "f1_custom_csv_loader",
            "CSV Object File Loader",
            "src/gravity_sim_3Dgrid_function.cpp: CreateObjects(filePath, 0)",
            "Loads validated initial positions, velocities, masses, and densities from CSV.",
            1,
        },
    };

    solver_.id = "gpu_snapshot_euler_v1";
    solver_.displayName = "GPU Snapshot Pairwise N-Body (dv/96, dx/94)";
    solver_.backendName = "OpenGL 4.4 Compute (SSBO Persistent Coherent)";
    solver_.status = grav_sim::SolverValidationStatus::SourceCompatibleLegacy;
    solver_.notes =
        "Preserves exact source semantics: softening epsilon=10 m, dv += a/96, dx += v/94.";

    std::snprintf(csvPathBuffer_, sizeof(csvPathBuffer_), "%s", customCsvPath_.c_str());
}

GravityGpuModule::~GravityGpuModule() { ReleaseResources(); }

void GravityGpuModule::LoadPresetObjects(std::size_t presetIndex)
{
    std::vector<Object>().swap(objs);
    if (presetIndex == 0)
    {
        objs = CreateObjects({}, 200);
    }
    else if (presetIndex == 1)
    {
        objs = CreateObjects({}, 0);
    }
    else if (presetIndex == 2)
    {
        objs = CreateLegacyThreeBodyObjects();
    }
    else
    {
        objs = CreateObjects(customCsvPath_, 0);
    }
}

void GravityGpuModule::OnEnter(const grav_sim::ModuleServices &services, std::size_t presetIndex)
{
    services_ = services;
    activePresetIndex_ = presetIndex < presets_.size() ? presetIndex : 0;

    if (!cameraInitialized_)
    {
        camera_.ConfigureDefaults(glm::vec3(0.0f, 1000.0f, 5000.0f), -90.0f, 0.0f, 2500.0f,
                                  900.0f, 45.0f, 0.1f, 750000.0f);
        cameraInitialized_ = true;
    }

    try
    {
        objectRenderProgram_ = services_.shaderCache->GetOrCreateGraphics(
            "f1:objects_render", "shaders/objects.vert", "shaders/objects.frag");
        nbodyComputeProgram_ =
            services_.shaderCache->GetOrCreateCompute("f1:nbody_compute", "shaders/nbody.comp");
        integrateComputeProgram_ = services_.shaderCache->GetOrCreateCompute(
            "f1:integrate_compute", "shaders/integrate.comp");

        ShaderPrograms progs{};
        progs.objectRendering = objectRenderProgram_;
        progs.nbodyCompute = nbodyComputeProgram_;
        progs.integrateCompute = integrateComputeProgram_;
        InitializeComputeUniformLocations(progs, computeUniforms_);
        InitializeObjectRenderingUniforms(objectRenderProgram_, renderUniforms_);

        CreateUnitSphere();
        if (services_.resourceCensus != nullptr && GetUnitSphereVao() != 0)
        {
            const std::size_t sphereBytes =
                static_cast<std::size_t>(GetUnitSphereVertexCount()) * 3 * sizeof(float);
            sphereVaoCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::VertexArray, GetUnitSphereVao(), 0, 0,
                "scene:f1_gravity_gpu", "UnitSphereVAO", services_.sceneGeneration);
            sphereVboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::GeometryBuffer, GetUnitSphereVbo(), sphereBytes,
                sphereBytes, "scene:f1_gravity_gpu", "UnitSphereVBO", services_.sceneGeneration);
        }

        LoadPresetObjects(activePresetIndex_);
        objectCount_ = objs.size();
        InitializeObjectComputeBuffers(objectCount_);
        objectGpuState(0, objectCount_);

        stepCount_ = 0;
        holdPauseActive_ = false;
        initializingObjectDirty_ = false;
        trajectorySamples_.clear();
        lastStatusMessage_.clear();

        if (services_.sharedGrid != nullptr)
        {
            services_.sharedGrid->Configure(grav_render::GridConfig::GravityCompatibility());
            grav_telemetry::ScopedGpuPassQuery query(
                services_.telemetry, grav_telemetry::GpuPassId::GridMaterialize,
                services_.sceneGeneration);
            services_.sharedGrid->MaterializeFromGpuObjects(objectCount_, stepCount_ + 1);
        }

        if (enableTrajectoryExport_)
        {
            CaptureGpuTrajectorySampleV2(trajectorySamples_, services_.sceneGeneration, stepCount_,
                                         0.0);
        }

        resourcesActive_ = true;
    }
    catch (...)
    {
        ReleaseResources();
        throw;
    }
}

void GravityGpuModule::HandleInput(const grav_sim::ModuleInputState &input,
                                   double wallDeltaSeconds)
{
    if (!resourcesActive_)
    {
        return;
    }

    holdPauseActive_ = input.holdPauseK;
    if (input.focusLostThisFrame)
    {
        holdPauseActive_ = false;
        if (!objs.empty() && objs.back().initializing)
        {
            objs.back().initializing = false;
            objs.back().launched = true;
            SynchronizeObjectGpuProperties(objs.size() - 1);
            initializingObjectDirty_ = true;
        }
    }

    const bool creatingObject = (!objs.empty() && objs.back().initializing);
    // When holding right-click during left-click object spawn, right-click grows object mass
    // rather than rotating the camera.
    camera_.HandleInput(input, wallDeltaSeconds, !creatingObject);

    if (input.mouseLeftJustPressed)
    {
        objs.emplace_back(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), spawnMassKg_);
        objs.back().initializing = true;
        initializingObjectDirty_ = true;
    }

    if (!objs.empty() && objs.back().initializing)
    {
        Object &spawning = objs.back();
        if (input.mouseRightDown)
        {
            const float dtClamped = static_cast<float>(std::clamp(wallDeltaSeconds, 0.0, 0.1));
            spawning.mass *= (1.0f + 1.0f * dtClamped);
            spawning.rs = static_cast<float>((2.0 * kGravitationalConstant * spawning.mass) /
                                             (static_cast<double>(kSpeedOfLight) * kSpeedOfLight));
            spawning.radius =
                std::pow((3.0f * spawning.mass / spawning.density) / (4.0f * 3.14159265359f),
                         1.0f / 3.0f) /
                100000.0f;
            initializingObjectDirty_ = true;
        }

        bool posMoved = false;
        const float moveStep = 25.0f;
        if (input.arrowUp)
        {
            if (!input.shiftHeld)
                spawning.position.y += moveStep;
            else
                spawning.position.z += moveStep;
            posMoved = true;
        }
        if (input.arrowDown)
        {
            if (!input.shiftHeld)
                spawning.position.y -= moveStep;
            else
                spawning.position.z -= moveStep;
            posMoved = true;
        }
        if (input.arrowRight)
        {
            spawning.position.x += moveStep;
            posMoved = true;
        }
        if (input.arrowLeft)
        {
            spawning.position.x -= moveStep;
            posMoved = true;
        }

        if (posMoved && objectCount_ == objs.size())
        {
            UploadInitializingObjectState(objs.size() - 1);
            initializingObjectDirty_ = true;
        }
    }

    if (input.mouseLeftJustReleased && !objs.empty() && objs.back().initializing)
    {
        objs.back().initializing = false;
        objs.back().launched = true;
        if (objectCount_ == objs.size())
        {
            SynchronizeObjectGpuProperties(objs.size() - 1);
        }
        initializingObjectDirty_ = true;
    }
}

void GravityGpuModule::Update(const grav_sim::StepContext &step)
{
    if (!resourcesActive_)
    {
        return;
    }

    bool buffersMutated = false;
    if (objs.size() != objectCount_)
    {
        SynchronizeObjectStateBuffers(objectCount_, objs.size());
        buffersMutated = true;
    }
    if (!objs.empty() && objs.back().initializing && initializingObjectDirty_)
    {
        SynchronizeObjectGpuProperties(objs.size() - 1);
        UploadInitializingObjectState(objs.size() - 1);
        buffersMutated = true;
    }
    initializingObjectDirty_ = false;

    const bool effectivePaused = (step.paused || holdPauseActive_) && !step.singleStepRequested;

    if (!effectivePaused && objectCount_ > 0)
    {
        BindAllObjectComputeBuffers();
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);

        {
            grav_telemetry::ScopedGpuPassQuery nbodyQuery(
                services_.telemetry, grav_telemetry::GpuPassId::NBody, step.sceneGeneration);
            RunNBodyCompute(nbodyComputeProgram_, computeUniforms_, objectCount_,
                            softeningEpsilon_);
        }
        {
            grav_telemetry::ScopedGpuPassQuery intQuery(
                services_.telemetry, grav_telemetry::GpuPassId::Integrate, step.sceneGeneration);
            RunIntegration(integrateComputeProgram_, computeUniforms_, objectCount_, false);
        }
        gpuMemoryManager.SwapCurrentNext();

        ++stepCount_;
        buffersMutated = true;
        if (services_.telemetry != nullptr)
        {
            services_.telemetry->RecordModelSteps(1);
        }

        if (enableTrajectoryExport_ && trajectorySampleIntervalSteps_ > 0 &&
            (stepCount_ % static_cast<std::uint64_t>(trajectorySampleIntervalSteps_) == 0))
        {
            CaptureGpuTrajectorySampleV2(trajectorySamples_, step.sceneGeneration, stepCount_,
                                         static_cast<double>(stepCount_) * dt);
        }
    }

    if (buffersMutated && services_.sharedGrid != nullptr)
    {
        grav_telemetry::ScopedGpuPassQuery matQuery(services_.telemetry,
                                                    grav_telemetry::GpuPassId::GridMaterialize,
                                                    step.sceneGeneration);
        services_.sharedGrid->MaterializeFromGpuObjects(objectCount_, stepCount_ + 1);
    }
}

void GravityGpuModule::Render(const grav_sim::RenderContext &render)
{
    if (!resourcesActive_ || render.framebufferWidth <= 0 || render.framebufferHeight <= 0)
    {
        return;
    }

    const float aspect = static_cast<float>(render.framebufferWidth) /
                         static_cast<float>(std::max(render.framebufferHeight, 1));
    const glm::mat4 view = camera_.ViewMatrix();
    const glm::mat4 proj = camera_.ProjectionMatrix(aspect);

    if (services_.sharedGrid != nullptr && render.gridVisible && services_.sharedGrid->IsDirty())
    {
        grav_telemetry::ScopedGpuPassQuery deformQuery(
            services_.telemetry, grav_telemetry::GpuPassId::GridDeform, render.sceneGeneration);
        services_.sharedGrid->UpdateIfDirty(true);
    }

    grav_telemetry::ScopedGpuPassQuery drawQuery(
        services_.telemetry, grav_telemetry::GpuPassId::SceneDraw, render.sceneGeneration);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (services_.sharedGrid != nullptr)
    {
        services_.sharedGrid->Draw(view, proj, render.gridVisible, true);
    }

    if (!objs.empty())
    {
        glUseProgram(objectRenderProgram_);
        glUniformMatrix4fv(renderUniforms_.objectView, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(renderUniforms_.objectProjection, 1, GL_FALSE, glm::value_ptr(proj));
        DrawObjects(objectRenderProgram_, objs);
    }
}

void GravityGpuModule::OnResize(int framebufferWidth, int framebufferHeight)
{
    services_.framebufferWidth = framebufferWidth;
    services_.framebufferHeight = framebufferHeight;
}

void GravityGpuModule::ResetCamera() { camera_.Reset(); }

void GravityGpuModule::BeginExit()
{
    if (!resourcesActive_)
    {
        return;
    }

    RetireInitializingObjects();
    objectCount_ = objs.size();

    if (enableTrajectoryExport_ && !trajectorySamples_.empty())
    {
        SavePositionsV2(trajectoryExportPath_, trajectorySamples_);
    }
}

void GravityGpuModule::ReleaseResources()
{
    if (services_.resourceCensus != nullptr)
    {
        services_.resourceCensus->Unregister(sphereVaoCensusId_);
        services_.resourceCensus->Unregister(sphereVboCensusId_);
    }
    DestroyObjectRenderingResources();
    ReleaseObjectComputeBuffers(true);

    std::vector<TrajectorySampleV2>().swap(trajectorySamples_);
    objectCount_ = 0;
    stepCount_ = 0;
    objectRenderProgram_ = 0;
    nbodyComputeProgram_ = 0;
    integrateComputeProgram_ = 0;
    resourcesActive_ = false;
}

grav_sim::ModuleRuntimeStats GravityGpuModule::RuntimeStats() const
{
    grav_sim::ModuleRuntimeStats stats;
    stats.primaryCount = objectCount_;
    stats.primaryLabel = "Bodies";
    stats.outputWidth = services_.framebufferWidth;
    stats.outputHeight = services_.framebufferHeight;
    stats.stepCount = stepCount_;
    stats.simulationOrAffineParam = static_cast<double>(stepCount_) * dt;
    stats.paramLabel = "Nominal Step Time (step/96)";
    if (activePresetIndex_ == 0)
    {
        stats.hasSeed = true;
        stats.seed = 123;
    }
    stats.extraStatus = lastStatusMessage_;
    return stats;
}

void GravityGpuModule::DrawInteractiveControls(bool configLocked)
{
    ImGui::BeginDisabled(configLocked);
    ImGui::SliderFloat("Softening epsilon (m)", &softeningEpsilon_, 0.1f, 5000.0f, "%.1f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Spawn Mass (kg)", &spawnMassKg_, 1.0e18f, 1.0e26f, "%.2e",
                       ImGuiSliderFlags_Logarithmic);

    if (activePresetIndex_ == 3)
    {
        ImGui::SeparatorText("CSV Preset File");
        if (ImGui::InputText("CSV Path", csvPathBuffer_, sizeof(csvPathBuffer_)))
        {
            customCsvPath_ = csvPathBuffer_;
        }
        ImGui::TextDisabled("Press R (Reset Runtime) after changing CSV path.");
    }

    ImGui::SeparatorText("Trajectory CSV Export (Schema v2)");
    ImGui::Checkbox("Enable GPU Trajectory Readback", &enableTrajectoryExport_);
    ImGui::SameLine();
    ImGui::TextDisabled("(Disabled by default for zero readback overhead)");
    if (enableTrajectoryExport_)
    {
        ImGui::SliderInt("Sample Interval (steps)", &trajectorySampleIntervalSteps_, 1, 480);
        if (ImGui::Button("Capture & Save CSV v2 Now"))
        {
            CaptureGpuTrajectorySampleV2(trajectorySamples_, services_.sceneGeneration, stepCount_,
                                         static_cast<double>(stepCount_) * dt);
            if (SavePositionsV2(trajectoryExportPath_, trajectorySamples_))
            {
                lastStatusMessage_ = "Saved " + std::to_string(trajectorySamples_.size()) +
                                     " samples to " + trajectoryExportPath_;
            }
            else
            {
                lastStatusMessage_ = "Failed to write " + trajectoryExportPath_;
            }
        }
    }
    ImGui::EndDisabled();

    if (!lastStatusMessage_.empty())
    {
        ImGui::TextWrapped("%s", lastStatusMessage_.c_str());
    }
}

std::string GravityGpuModule::ControlsHelpText() const
{
    return "W/A/S/D, Space, Left Shift: Free-fly camera | Right-Drag: Look | Scroll: Zoom | "
           "Left-Click (hold): Spawn object, Right-Hold: Grow mass, Arrows (+Shift): Move spawn";
}

} // namespace grav_models

