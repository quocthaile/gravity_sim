#include "models/gravity_baseline/gravity_baseline_module.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <imgui.h>

namespace grav_models
{

namespace
{

const char *kCpuInstancedSphereVert = R"glsl(#version 440 core
layout(location = 0) in vec3 localPosition;
layout(location = 1) in vec4 instancePosRadius;
layout(location = 2) in vec4 instanceColor;

uniform mat4 view;
uniform mat4 projection;

out vec4 vertexColor;

void main()
{
    vec3 worldPosition = instancePosRadius.xyz + localPosition * instancePosRadius.w;
    vertexColor = instanceColor;
    gl_Position = projection * view * vec4(worldPosition, 1.0);
}
)glsl";

const char *kCpuInstancedSphereFrag = R"glsl(#version 440 core
in vec4 vertexColor;
out vec4 fragmentColor;

void main()
{
    fragmentColor = vertexColor;
}
)glsl";

} // namespace

void StepBaselineBodiesInPlace(std::vector<Object> &bodies, double stepDt)
{
    for (auto &obj : bodies)
    {
        for (auto &obj2 : bodies)
        {
            if (&obj2 != &obj && !obj.initializing && !obj2.initializing)
            {
                const float dx = obj2.GetPosition()[0] - obj.GetPosition()[0];
                const float dy = obj2.GetPosition()[1] - obj.GetPosition()[1];
                const float dz = obj2.GetPosition()[2] - obj.GetPosition()[2];
                float distance = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (distance > 0.0f)
                {
                    const float dirX = dx / distance;
                    const float dirY = dy / distance;
                    const float dirZ = dz / distance;
                    distance *= 1000.0f;
                    const double gForce =
                        (kGravitationalConstant * static_cast<double>(obj.mass) *
                         static_cast<double>(obj2.mass)) /
                        static_cast<double>(distance * distance);

                    const float acc1 = static_cast<float>(gForce / static_cast<double>(obj.mass));
                    obj.Accelerate(dirX * acc1, dirY * acc1, dirZ * acc1, stepDt);

                    // Exact historical baseline collision response inside the inner loop.
                    obj.velocity *= obj.CheckCollision(obj2);
                }
            }
        }

        if (obj.initializing)
        {
            obj.radius = static_cast<float>(
                std::pow(((3.0 * obj.mass / obj.density) / (4.0 * 3.14159265359)), (1.0 / 3.0)) /
                100000.0);
        }
        else
        {
            // Historical in-place integration: body i updates position before body i+1 evaluates forces.
            obj.UpdatePosition(stepDt);
        }
    }
}

GravityBaselineModule::GravityBaselineModule()
{
    presets_ = {
        {
            "f6_default_2_plus_200_seed123",
            "Baseline CPU: Earth-Moon + 200 Orbiters (seed 123)",
            "src/gravity_sim_3Dgrid_baseline.cpp: CreateObjects({}, 200)",
            "Historical CPU sequential in-place N-body solver rendered via core-profile adapter.",
            1,
        },
        {
            "f6_two_body_earth_moon",
            "Baseline CPU: Two-Body Earth-Moon (random = 0)",
            "src/gravity_sim_3Dgrid_baseline.cpp: CreateObjects({}, 0)",
            "Deterministic 2-body baseline configuration without softening.",
            1,
        },
        {
            "f6_legacy_three_body_ic",
            "Legacy Initial Conditions: Three-Body (CPU Baseline Solver)",
            "src/legacy/gravity_sim.cpp:227-232 ICs + baseline CPU in-place solver",
            "Three-body initial state evaluated under the historical CPU in-place solver.",
            1,
        },
        {
            "f6_custom_csv_loader",
            "Baseline CPU: CSV Object File Loader",
            "src/gravity_sim_3Dgrid_baseline.cpp: CreateObjects(filePath, 0)",
            "Loads initial conditions from CSV and advances them with the CPU baseline solver.",
            1,
        },
    };

    solver_.id = "cpu_inplace_euler_baseline_v1";
    solver_.displayName = "Historical CPU Sequential In-Place Pairwise Euler (dt = 1/96, eps = 0)";
    solver_.backendName = "Single-Thread CPU Reference + OpenGL 4.4 Core Instanced Adapter";
    solver_.status = grav_sim::SolverValidationStatus::SourceCompatibleLegacy;
    solver_.notes =
        "Physics matches src/gravity_sim_3Dgrid_baseline.cpp (in-place update, no softening); "
        "rendering uses shared Core-Profile instanced sphere + SharedGridService.";

    std::snprintf(csvPathBuffer_, sizeof(csvPathBuffer_), "%s", customCsvPath_.c_str());
}

GravityBaselineModule::~GravityBaselineModule() { ReleaseResources(); }

void GravityBaselineModule::LoadPresetBodies(std::size_t presetIndex)
{
    bodies_.clear();
    if (presetIndex == 0)
    {
        bodies_ = CreateObjects({}, 200);
    }
    else if (presetIndex == 1)
    {
        bodies_ = CreateObjects({}, 0);
    }
    else if (presetIndex == 2)
    {
        bodies_ = CreateLegacyThreeBodyObjects();
    }
    else
    {
        bodies_ = CreateObjects(customCsvPath_, 0);
    }
}

void GravityBaselineModule::EnsureInstanceCapacity(std::size_t requiredCount)
{
    const std::size_t targetCap = std::max<std::size_t>(requiredCount, 16);
    if (instanceVbo_ != 0 && targetCap <= instanceCapacity_)
    {
        return;
    }

    std::size_t newCap = std::max<std::size_t>(instanceCapacity_ * 2, 64);
    while (newCap < targetCap)
    {
        newCap *= 2;
    }

    if (instanceVao_ == 0)
    {
        glGenVertexArrays(1, &instanceVao_);
        if (services_.resourceCensus != nullptr)
        {
            instanceVaoCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::VertexArray, instanceVao_, 0, 0,
                "scene:f6_gravity_baseline", "CpuSphereInstanceVAO", services_.sceneGeneration);
        }
    }

    if (instanceVbo_ == 0)
    {
        glGenBuffers(1, &instanceVbo_);
        if (services_.resourceCensus != nullptr)
        {
            instanceVboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::GeometryBuffer, instanceVbo_, 0, 0,
                "scene:f6_gravity_baseline", "CpuSphereInstanceVBO", services_.sceneGeneration);
        }
    }

    instanceCapacity_ = newCap;
    const std::size_t allocBytes = instanceCapacity_ * sizeof(CpuSphereInstance);

    glBindVertexArray(instanceVao_);

    // Attribute 0: unit sphere local vertex position
    glBindBuffer(GL_ARRAY_BUFFER, GetUnitSphereVbo());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glVertexAttribDivisor(0, 0);

    // Attributes 1 & 2: per-instance world position+radius and color
    glBindBuffer(GL_ARRAY_BUFFER, instanceVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(allocBytes), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(CpuSphereInstance),
                          reinterpret_cast<const void *>(offsetof(CpuSphereInstance, posRadius)));
    glVertexAttribDivisor(1, 1);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(CpuSphereInstance),
                          reinterpret_cast<const void *>(offsetof(CpuSphereInstance, color)));
    glVertexAttribDivisor(2, 1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    if (services_.resourceCensus != nullptr && instanceVboCensusId_ != 0)
    {
        services_.resourceCensus->UpdateBytes(instanceVboCensusId_, instanceVbo_, allocBytes,
                                              bodies_.size() * sizeof(CpuSphereInstance));
    }
}

void GravityBaselineModule::SyncGridAndInstances()
{
    EnsureInstanceCapacity(bodies_.size());

    instanceScratch_.resize(bodies_.size());
    gridSources_.clear();
    gridSources_.reserve(bodies_.size());

    for (std::size_t i = 0; i < bodies_.size(); ++i)
    {
        const Object &b = bodies_[i];
        instanceScratch_[i].posRadius = glm::vec4(b.position, std::max(b.radius, 1.0f));
        instanceScratch_[i].color = b.color;

        if (!b.initializing)
        {
            grav_render::GridSource src{};
            src.positionRs = glm::vec4(b.position, b.rs);
            gridSources_.push_back(src);
        }
    }

    if (!instanceScratch_.empty())
    {
        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo_);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        static_cast<GLsizeiptr>(instanceScratch_.size() * sizeof(CpuSphereInstance)),
                        instanceScratch_.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    if (services_.resourceCensus != nullptr && instanceVboCensusId_ != 0)
    {
        services_.resourceCensus->UpdateBytes(
            instanceVboCensusId_, instanceVbo_, instanceCapacity_ * sizeof(CpuSphereInstance),
            instanceScratch_.size() * sizeof(CpuSphereInstance));
    }

    ++sourceRevision_;
    if (services_.sharedGrid != nullptr)
    {
        services_.sharedGrid->SetCpuSources(gridSources_, sourceRevision_);
    }
}

void GravityBaselineModule::RecordCpuTrajectorySample(std::uint64_t sceneGeneration)
{
    TrajectorySampleV2 sample{};
    sample.sceneGeneration = sceneGeneration;
    sample.simulationStep = stepCount_;
    sample.nominalTimeSeconds = static_cast<double>(stepCount_) * (1.0 / 96.0);
    sample.entries.reserve(bodies_.size());
    for (const auto &b : bodies_)
    {
        if (!b.initializing)
        {
            TrajectoryRecordEntryV2 entry{};
            entry.logicalObjectId = b.id;
            entry.position = b.position;
            sample.entries.push_back(entry);
        }
    }
    trajectorySamples_.push_back(std::move(sample));
}

void GravityBaselineModule::OnEnter(const grav_sim::ModuleServices &services,
                                    std::size_t presetIndex)
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
        instancedProgram_ = services_.shaderCache->GetOrCreateGraphicsFromSource(
            "f6:cpu_objects_instanced", kCpuInstancedSphereVert, kCpuInstancedSphereFrag);
        locView_ = glGetUniformLocation(instancedProgram_, "view");
        locProjection_ = glGetUniformLocation(instancedProgram_, "projection");

        CreateUnitSphere();
        if (services_.resourceCensus != nullptr && GetUnitSphereVao() != 0)
        {
            const std::size_t sphereBytes =
                static_cast<std::size_t>(GetUnitSphereVertexCount()) * 3 * sizeof(float);
            sphereVaoCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::VertexArray, GetUnitSphereVao(), 0, 0,
                "scene:f6_gravity_baseline", "UnitSphereVAO", services_.sceneGeneration);
            sphereVboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::GeometryBuffer, GetUnitSphereVbo(), sphereBytes,
                sphereBytes, "scene:f6_gravity_baseline", "UnitSphereVBO",
                services_.sceneGeneration);
        }

        LoadPresetBodies(activePresetIndex_);
        stepCount_ = 0;
        holdPauseActive_ = false;
        stateDirty_ = false;
        trajectorySamples_.clear();
        lastStatusMessage_.clear();

        if (services_.sharedGrid != nullptr)
        {
            services_.sharedGrid->Configure(grav_render::GridConfig::GravityCompatibility());
        }
        SyncGridAndInstances();

        if (enableTrajectoryExport_)
        {
            RecordCpuTrajectorySample(services_.sceneGeneration);
        }

        resourcesActive_ = true;
    }
    catch (...)
    {
        ReleaseResources();
        throw;
    }
}

void GravityBaselineModule::HandleInput(const grav_sim::ModuleInputState &input,
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
        if (!bodies_.empty() && bodies_.back().initializing)
        {
            bodies_.back().initializing = false;
            bodies_.back().launched = true;
            stateDirty_ = true;
        }
    }

    const bool creatingBody = (!bodies_.empty() && bodies_.back().initializing);
    camera_.HandleInput(input, wallDeltaSeconds, !creatingBody);

    if (input.mouseLeftJustPressed)
    {
        bodies_.emplace_back(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                             spawnMassKg_);
        bodies_.back().initializing = true;
        stateDirty_ = true;
    }

    if (!bodies_.empty() && bodies_.back().initializing)
    {
        Object &spawning = bodies_.back();
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
            stateDirty_ = true;
        }

        const float moveStep = 25.0f;
        if (input.arrowUp)
        {
            if (!input.shiftHeld)
                spawning.position.y += moveStep;
            else
                spawning.position.z += moveStep;
            stateDirty_ = true;
        }
        if (input.arrowDown)
        {
            if (!input.shiftHeld)
                spawning.position.y -= moveStep;
            else
                spawning.position.z -= moveStep;
            stateDirty_ = true;
        }
        if (input.arrowRight)
        {
            spawning.position.x += moveStep;
            stateDirty_ = true;
        }
        if (input.arrowLeft)
        {
            spawning.position.x -= moveStep;
            stateDirty_ = true;
        }
    }

    if (input.mouseLeftJustReleased && !bodies_.empty() && bodies_.back().initializing)
    {
        bodies_.back().initializing = false;
        bodies_.back().launched = true;
        stateDirty_ = true;
    }
}

void GravityBaselineModule::Update(const grav_sim::StepContext &step)
{
    if (!resourcesActive_)
    {
        return;
    }

    const bool effectivePaused = (step.paused || holdPauseActive_) && !step.singleStepRequested;
    if (!effectivePaused && !bodies_.empty())
    {
        StepBaselineBodiesInPlace(bodies_, 1.0 / 96.0);
        ++stepCount_;
        stateDirty_ = true;

        if (services_.telemetry != nullptr)
        {
            services_.telemetry->RecordModelSteps(1);
        }

        if (enableTrajectoryExport_ && trajectorySampleIntervalSteps_ > 0 &&
            (stepCount_ % static_cast<std::uint64_t>(trajectorySampleIntervalSteps_) == 0))
        {
            RecordCpuTrajectorySample(step.sceneGeneration);
        }
    }

    if (stateDirty_)
    {
        SyncGridAndInstances();
        stateDirty_ = false;
    }
}

void GravityBaselineModule::Render(const grav_sim::RenderContext &render)
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

    if (!bodies_.empty() && instanceVao_ != 0)
    {
        glUseProgram(instancedProgram_);
        glUniformMatrix4fv(locView_, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(locProjection_, 1, GL_FALSE, glm::value_ptr(proj));
        glBindVertexArray(instanceVao_);
        glDrawArraysInstanced(GL_TRIANGLES, 0, GetUnitSphereVertexCount(),
                              static_cast<GLsizei>(bodies_.size()));
        glBindVertexArray(0);
    }
}

void GravityBaselineModule::OnResize(int framebufferWidth, int framebufferHeight)
{
    services_.framebufferWidth = framebufferWidth;
    services_.framebufferHeight = framebufferHeight;
}

void GravityBaselineModule::ResetCamera() { camera_.Reset(); }

void GravityBaselineModule::BeginExit()
{
    holdPauseActive_ = false;
    while (!bodies_.empty() && bodies_.back().initializing)
    {
        bodies_.pop_back();
    }

    if (enableTrajectoryExport_ && !trajectorySamples_.empty())
    {
        SavePositionsV2(trajectoryExportPath_, trajectorySamples_);
    }
}

void GravityBaselineModule::ReleaseResources()
{
    if (services_.resourceCensus != nullptr)
    {
        if (instanceVaoCensusId_ != 0)
        {
            services_.resourceCensus->Unregister(instanceVaoCensusId_);
        }
        if (instanceVboCensusId_ != 0)
        {
            services_.resourceCensus->Unregister(instanceVboCensusId_);
        }
        if (sphereVaoCensusId_ != 0)
        {
            services_.resourceCensus->Unregister(sphereVaoCensusId_);
        }
        if (sphereVboCensusId_ != 0)
        {
            services_.resourceCensus->Unregister(sphereVboCensusId_);
        }
    }

    if (instanceVao_ != 0)
    {
        glDeleteVertexArrays(1, &instanceVao_);
        instanceVao_ = 0;
    }
    if (instanceVbo_ != 0)
    {
        glDeleteBuffers(1, &instanceVbo_);
        instanceVbo_ = 0;
    }
    instanceCapacity_ = 0;

    DestroyObjectRenderingResources();
    std::vector<Object>().swap(bodies_);
    std::vector<grav_render::GridSource>().swap(gridSources_);
    std::vector<CpuSphereInstance>().swap(instanceScratch_);
    trajectorySamples_.clear();
    resourcesActive_ = false;
}

grav_sim::ModuleRuntimeStats GravityBaselineModule::RuntimeStats() const
{
    grav_sim::ModuleRuntimeStats stats{};
    stats.primaryCount = bodies_.size();
    stats.primaryLabel = "Bodies (CPU)";
    stats.outputWidth = services_.framebufferWidth;
    stats.outputHeight = services_.framebufferHeight;
    stats.stepCount = stepCount_;
    stats.simulationOrAffineParam = static_cast<double>(stepCount_) * (1.0 / 96.0);
    stats.paramLabel = "Nominal Time (s)";
    stats.hasSeed = (activePresetIndex_ == 0);
    stats.seed = 123;
    stats.extraStatus = lastStatusMessage_;
    return stats;
}

void GravityBaselineModule::DrawInteractiveControls(bool configLocked)
{
    ImGui::TextWrapped("Historical CPU Adapter Note: Physics matches "
                       "src/gravity_sim_3Dgrid_baseline.cpp (in-place sequential Euler, dt=1/96, "
                       "no softening). Rendering uses the shared OpenGL 4.4 Core instanced sphere "
                       "+ SharedGridService rather than original per-object VAO/VBOs. "
                       "Standalone gravity_sim_3Dgrid_baseline executable remains untouched.");
    ImGui::Separator();

    ImGui::BeginDisabled(configLocked);
    ImGui::SliderFloat("Spawn Mass (kg)", &spawnMassKg_, 1.0e18f, 1.0e25f, "%.3e",
                       ImGuiSliderFlags_Logarithmic);

    if (activePresetIndex_ == 3)
    {
        if (ImGui::InputText("CSV File Path", csvPathBuffer_, sizeof(csvPathBuffer_)))
        {
            customCsvPath_ = csvPathBuffer_;
        }
        if (ImGui::Button("Reload CSV Preset"))
        {
            try
            {
                LoadPresetBodies(3);
                stepCount_ = 0;
                SyncGridAndInstances();
                lastStatusMessage_ =
                    "Loaded " + std::to_string(bodies_.size()) + " bodies from CSV.";
            }
            catch (const std::exception &ex)
            {
                lastStatusMessage_ = std::string("CSV Error: ") + ex.what();
            }
        }
    }

    ImGui::Separator();
    ImGui::Checkbox("Opt-In Trajectory CSV Export (schema v2)", &enableTrajectoryExport_);
    if (enableTrajectoryExport_)
    {
        ImGui::SliderInt("Sample Interval (steps)", &trajectorySampleIntervalSteps_, 1, 480);
        ImGui::Text("Buffered Samples: %zu", trajectorySamples_.size());
        if (ImGui::Button("Flush Trajectory CSV Now"))
        {
            if (SavePositionsV2(trajectoryExportPath_, trajectorySamples_))
            {
                lastStatusMessage_ = "Saved trajectory CSV to " + trajectoryExportPath_;
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

std::string GravityBaselineModule::ControlsHelpText() const
{
    return "WASD + Space/LeftCtrl: Fly camera | Right-Drag: Look | Scroll: FOV Zoom | "
           "Hold K: Temporary Pause | Left-Click Hold: Spawn body (Right-Click grows mass, "
           "Arrow keys/+Shift move)";
}

} // namespace grav_models

