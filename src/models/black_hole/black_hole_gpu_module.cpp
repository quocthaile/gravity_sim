#include "models/black_hole/black_hole_gpu_module.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <imgui.h>

namespace grav_models
{
namespace
{
constexpr float kWorldMetersPerGm = 1.0e9f;
} // namespace

BlackHoleGpuModule::BlackHoleGpuModule()
{
    presets_ = {
        {
            "f3_static_accretion_lensing",
            "Sagittarius A* Accretion Disk + Background Stars (black_hole.cpp)",
            "black_hole/black_hole.cpp:144-149 + geodesic.comp",
            "Equatorial relativistic accretion disk (2.2..5.2 r_s) with two background stars.",
            1,
        },
        {
            "f3_einstein_ring_alignment",
            "Einstein-Ring Star Alignment (Disk Disabled)",
            "Proposed lensing configuration using black_hole.cpp sphere primitives",
            "Isolates gravitational double-imaging and Einstein-ring arcs of background stars.",
            1,
        },
        {
            "f3_legacy_experimental_object_gravity",
            "Legacy Experimental: Newtonian Object Motion on Static Metric",
            "black_hole/black_hole.cpp:658-684 (legacy Gravity branch, isolated from G key)",
            "Experimental hybrid approximation: Newtonian object pair-forces over static BH metric.",
            1,
        },
    };

    solvers_ = {
        {
            "rk4_canonical_v1",
            "Schwarzschild 3D Null-Geodesic 4-Stage RK4 (Canonical)",
            "OpenGL 4.4 Compute (image2D + std140 UBO)",
            grav_sim::SolverValidationStatus::NumericalValidated,
            "Dimensionless Cartesian 4-stage RK4 with exact photon sphere at 1.5 r_s and segment crossing.",
        },
        {
            "euler_spherical_legacy_v1",
            "Legacy Spherical 1-Stage Euler (Imported geodesic.comp)",
            "OpenGL 4.4 Compute (image2D + fixed std140 UBO)",
            grav_sim::SolverValidationStatus::SourceCompatibleLegacy,
            "Preserves single-RHS-evaluation spherical step from upstream geodesic.comp.",
        },
    };
}

BlackHoleGpuModule::~BlackHoleGpuModule() { ReleaseResources(); }

bool BlackHoleGpuModule::SetActiveSolverById(const std::string &solverId)
{
    for (std::size_t i = 0; i < solvers_.size(); ++i)
    {
        if (solvers_[i].id == solverId)
        {
            if (activeSolverIndex_ != i)
            {
                activeSolverIndex_ = i;
                imageDirty_ = true;
            }
            return true;
        }
    }
    return false;
}

void BlackHoleGpuModule::ConfigurePresetScene(std::size_t presetIndex)
{
    const BlackHoleParameters bh =
        services_.sharedBlackHole ? *services_.sharedBlackHole : BlackHoleParameters{};
    const float rs = static_cast<float>(bh.schwarzschildRadiusMeters);
    const glm::vec3 bhPos(bh.positionMeters);

    sceneObjects_.clear();
    if (presetIndex == 0 || presetIndex == 2)
    {
        diskEnabled_ = true;
        legacyGravityMotionEnabled_ = (presetIndex == 2);
        sceneObjects_.push_back(
            {glm::vec4(bhPos + glm::vec3(4.0e11f, 0.0f, 0.0f), 4.0e10f),
             glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), 1.98892e30f, glm::vec3(0.0f)});
        sceneObjects_.push_back(
            {glm::vec4(bhPos + glm::vec3(0.0f, 0.0f, 4.0e11f), 4.0e10f),
             glm::vec4(1.0f, 0.15f, 0.1f, 1.0f), 1.98892e30f, glm::vec3(0.0f)});
        sceneObjects_.push_back({glm::vec4(bhPos, rs), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
                                 static_cast<float>(bh.massKg), glm::vec3(0.0f)});
    }
    else
    {
        diskEnabled_ = false;
        legacyGravityMotionEnabled_ = false;
        sceneObjects_.push_back(
            {glm::vec4(bhPos + glm::vec3(-2.6e11f, 0.0f, 0.0f), 3.4e10f),
             glm::vec4(0.35f, 0.78f, 1.0f, 1.0f), 1.98892e30f, glm::vec3(0.0f)});
        sceneObjects_.push_back(
            {glm::vec4(bhPos + glm::vec3(-3.2e11f, 3.8e10f, -4.2e10f), 2.8e10f),
             glm::vec4(1.0f, 0.85f, 0.30f, 1.0f), 1.98892e30f, glm::vec3(0.0f)});
        sceneObjects_.push_back({glm::vec4(bhPos, rs), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
                                 static_cast<float>(bh.massKg), glm::vec3(0.0f)});
    }
    imageDirty_ = true;
}

void BlackHoleGpuModule::EnsureOutputTexture(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        return; // Guard against 0x0 allocation on window minimize (T08)
    }
    if (outputTexture_ != 0 && allocatedTexW_ == width && allocatedTexH_ == height)
    {
        return;
    }

    if (outputTexture_ == 0)
    {
        glGenTextures(1, &outputTexture_);
    }

    glBindTexture(GL_TEXTURE_2D, outputTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    allocatedTexW_ = width;
    allocatedTexH_ = height;

    const std::size_t texBytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    if (services_.resourceCensus != nullptr)
    {
        if (texCensusId_ == 0)
        {
            texCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::Texture, outputTexture_, texBytes, texBytes,
                "scene:f3_black_hole_gpu", "BlackHoleOutputRGBA8", services_.sceneGeneration);
        }
        else
        {
            services_.resourceCensus->UpdateBytes(texCensusId_, outputTexture_, texBytes, texBytes);
        }
    }
    imageDirty_ = true;
}

void BlackHoleGpuModule::SyncSharedBlackHoleAndGrid()
{
    if (services_.sharedBlackHole == nullptr)
    {
        return;
    }
    lastBhRevision_ = services_.sharedBlackHole->revision;
    camera_.SetTarget(glm::vec3(services_.sharedBlackHole->positionMeters));

    if (!sceneObjects_.empty())
    {
        sceneObjects_.back().posRadiusMeters =
            glm::vec4(glm::vec3(services_.sharedBlackHole->positionMeters),
                      static_cast<float>(services_.sharedBlackHole->schwarzschildRadiusMeters));
        sceneObjects_.back().massKg = static_cast<float>(services_.sharedBlackHole->massKg);
    }

    if (services_.sharedGrid != nullptr)
    {
        const grav_render::GridConfig cfg = grav_render::GridConfig::BlackHoleWorldGm();
        services_.sharedGrid->Configure(cfg);
        const std::vector<grav_render::GridSource> src = {
            services_.sharedBlackHole->ToGridSource(cfg.metersPerWorldUnit)};
        services_.sharedGrid->SetCpuSources(src, lastBhRevision_);
    }
    imageDirty_ = true;
}

void BlackHoleGpuModule::OnEnter(const grav_sim::ModuleServices &services, std::size_t presetIndex)
{
    services_ = services;
    activePresetIndex_ = presetIndex < presets_.size() ? presetIndex : 0;

    if (!cameraConfigured_)
    {
        camera_.ConfigureDefaults(glm::vec3(0.0f), 6.34194e10f, 1.6e10f, 1.0e12f, 0.0f, 1.40f,
                                  60.0f, false);
        cameraConfigured_ = true;
    }

    try
    {
        computeProgramCanonical_ = services_.shaderCache->GetOrCreateCompute(
            "f3:geodesic_rk4", "shaders/black_hole/geodesic_rk4.comp");
        computeProgramLegacy_ = services_.shaderCache->GetOrCreateCompute(
            "f3:geodesic_legacy", "shaders/black_hole/geodesic_legacy.comp");

        glGenBuffers(1, &cameraUbo_);
        glBindBuffer(GL_UNIFORM_BUFFER, cameraUbo_);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(GpuCameraUboStd140), nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &diskUbo_);
        glBindBuffer(GL_UNIFORM_BUFFER, diskUbo_);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(GpuDiskUboStd140), nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &objectsUbo_);
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUbo_);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(GpuObjectsUboStd140), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        if (services_.resourceCensus != nullptr)
        {
            camUboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::Ubo, cameraUbo_, sizeof(GpuCameraUboStd140),
                sizeof(GpuCameraUboStd140), "scene:f3_black_hole_gpu", "CameraUBO",
                services_.sceneGeneration);
            diskUboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::Ubo, diskUbo_, sizeof(GpuDiskUboStd140),
                sizeof(GpuDiskUboStd140), "scene:f3_black_hole_gpu", "DiskUBO",
                services_.sceneGeneration);
            objUboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::Ubo, objectsUbo_, sizeof(GpuObjectsUboStd140),
                sizeof(GpuObjectsUboStd140), "scene:f3_black_hole_gpu", "ObjectsUBO",
                services_.sceneGeneration);
        }

        ConfigurePresetScene(activePresetIndex_);
        SyncSharedBlackHoleAndGrid();
        EnsureOutputTexture(baseComputeWidth_, baseComputeHeight_);

        lastCameraRevision_ = camera_.Revision();
        dispatchCount_ = 0;
        imageAgeMs_ = 0.0;
        imageDirty_ = true;
        resourcesActive_ = true;
    }
    catch (...)
    {
        ReleaseResources();
        throw;
    }
}

void BlackHoleGpuModule::HandleInput(const grav_sim::ModuleInputState &input,
                                     double wallDeltaSeconds)
{
    (void)wallDeltaSeconds;
    if (!resourcesActive_)
    {
        return;
    }
    if (camera_.HandleInput(input))
    {
        imageDirty_ = true;
    }
}

void BlackHoleGpuModule::StepLegacyExperimentalObjectGravity()
{
    // Isolated port of black_hole.cpp:658-684 experimental Newtonian pair loop
    for (std::size_t i = 0; i < sceneObjects_.size(); ++i)
    {
        // Keep central BH anchor fixed so shared BlackHoleParameters stays consistent
        if (i + 1 == sceneObjects_.size())
        {
            continue;
        }
        for (std::size_t j = 0; j < sceneObjects_.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            const glm::vec3 pi(sceneObjects_[i].posRadiusMeters);
            const glm::vec3 pj(sceneObjects_[j].posRadiusMeters);
            const glm::vec3 diff = pj - pi;
            const double dist = static_cast<double>(glm::length(diff));
            if (dist > 1.0e8)
            {
                const glm::dvec3 dir = glm::dvec3(diff) / dist;
                const double gForce =
                    (kGravConstSi * static_cast<double>(sceneObjects_[i].massKg) *
                     static_cast<double>(sceneObjects_[j].massKg)) /
                    (dist * dist);
                const double accMag = gForce / static_cast<double>(sceneObjects_[i].massKg);
                sceneObjects_[i].velocityMps += glm::vec3(dir * accMag);
                sceneObjects_[i].posRadiusMeters.x += sceneObjects_[i].velocityMps.x;
                sceneObjects_[i].posRadiusMeters.y += sceneObjects_[i].velocityMps.y;
                sceneObjects_[i].posRadiusMeters.z += sceneObjects_[i].velocityMps.z;
            }
        }
    }
    imageDirty_ = true;
}

BlackHoleSceneConfig BlackHoleGpuModule::BuildCurrentSceneConfig(int outW, int outH) const
{
    BlackHoleSceneConfig cfg;
    glm::vec3 pos, right, up, fwd;
    camera_.CameraBasis(pos, right, up, fwd);

    cfg.camPosMeters = glm::dvec3(pos);
    cfg.camRight = glm::dvec3(right);
    cfg.camUp = glm::dvec3(up);
    cfg.camForward = glm::dvec3(fwd);
    cfg.tanHalfFov = std::tan(glm::radians(static_cast<double>(camera_.FovDegrees()) * 0.5));

    const int fbW = std::max(services_.framebufferWidth, 1);
    const int fbH = std::max(services_.framebufferHeight, 1);
    cfg.aspect = (services_.framebufferWidth > 0 && services_.framebufferHeight > 0)
                     ? (static_cast<double>(fbW) / static_cast<double>(fbH))
                     : (static_cast<double>(std::max(outW, 1)) /
                        static_cast<double>(std::max(outH, 1)));

    const BlackHoleParameters bh =
        services_.sharedBlackHole ? *services_.sharedBlackHole : BlackHoleParameters{};
    cfg.bhCenterMeters = bh.positionMeters;
    cfg.rsMeters = bh.schwarzschildRadiusMeters;

    cfg.dLambdaMeters = static_cast<double>(dLambdaMeters_);
    cfg.escapeRadiusMeters = 1.0e14;
    cfg.maxSteps = maxStepsStatic_;

    cfg.diskEnabled = diskEnabled_;
    cfg.diskInnerMeters = bh.schwarzschildRadiusMeters * 2.2;
    cfg.diskOuterMeters = bh.schwarzschildRadiusMeters * 5.2;
    cfg.diskThicknessMeters = 1.0e9;
    cfg.objects = sceneObjects_;
    return cfg;
}

void BlackHoleGpuModule::DispatchRayCompute(int targetW, int targetH, bool movingFlag,
                                            std::uint64_t sceneGen)
{
    if (targetW <= 0 || targetH <= 0)
    {
        return;
    }

    EnsureOutputTexture(targetW, targetH);

    BlackHoleSceneConfig cfg = BuildCurrentSceneConfig(targetW, targetH);
    cfg.maxSteps = movingFlag ? maxStepsMoving_ : maxStepsStatic_;

    GpuCameraUboStd140 camUboData{};
    GpuDiskUboStd140 diskUboData{};
    GpuObjectsUboStd140 objUboData{};
    PackBlackHoleUbos(cfg, movingFlag, targetW, targetH, camUboData, diskUboData, objUboData);

    glBindBuffer(GL_UNIFORM_BUFFER, cameraUbo_);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(camUboData), &camUboData);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, cameraUbo_);

    glBindBuffer(GL_UNIFORM_BUFFER, diskUbo_);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(diskUboData), &diskUboData);
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, diskUbo_);

    glBindBuffer(GL_UNIFORM_BUFFER, objectsUbo_);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(objUboData), &objUboData);
    glBindBufferBase(GL_UNIFORM_BUFFER, 3, objectsUbo_);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    const GLuint prog = (activeSolverIndex_ == 0) ? computeProgramCanonical_ : computeProgramLegacy_;

    grav_telemetry::ScopedGpuPassQuery rayQuery(
        services_.telemetry, grav_telemetry::GpuPassId::RayImageCompute, sceneGen);

    glUseProgram(prog);
    glBindImageTexture(0, outputTexture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    const GLuint groupsX = static_cast<GLuint>((targetW + 15) / 16);
    const GLuint groupsY = static_cast<GLuint>((targetH + 15) / 16);
    glDispatchCompute(groupsX, groupsY, 1);

    // Explicit image-store -> texture-fetch synchronization contract (Section 7.1 & 9.3)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    ++dispatchCount_;
    imageDirty_ = false;
    imageAgeMs_ = 0.0;
    if (services_.telemetry != nullptr)
    {
        services_.telemetry->RecordImageEvaluation(0.0);
    }
}

void BlackHoleGpuModule::Update(const grav_sim::StepContext &step)
{
    if (!resourcesActive_)
    {
        return;
    }

    if (services_.sharedBlackHole != nullptr &&
        services_.sharedBlackHole->revision != lastBhRevision_)
    {
        SyncSharedBlackHoleAndGrid();
    }

    if (camera_.Revision() != lastCameraRevision_)
    {
        lastCameraRevision_ = camera_.Revision();
        imageDirty_ = true;
    }

    if (legacyGravityMotionEnabled_ && (!step.paused || step.singleStepRequested))
    {
        StepLegacyExperimentalObjectGravity();
        if (services_.telemetry != nullptr)
        {
            services_.telemetry->RecordModelSteps(1);
        }
    }

    const bool movingLodActive =
        enableMovingLod_ && !step.benchmarkConfigLocked && camera_.IsMoving();
    const int targetW = movingLodActive ? std::max(baseComputeWidth_ / 2, 64) : baseComputeWidth_;
    const int targetH = movingLodActive ? std::max(baseComputeHeight_ / 2, 48) : baseComputeHeight_;

    if (targetW != allocatedTexW_ || targetH != allocatedTexH_)
    {
        imageDirty_ = true;
    }

    if (imageDirty_ || step.forceRecomputeStaticImage || step.singleStepRequested)
    {
        DispatchRayCompute(targetW, targetH, movingLodActive, step.sceneGeneration);
    }
    else
    {
        imageAgeMs_ += step.wallDeltaSeconds * 1000.0;
        if (services_.telemetry != nullptr)
        {
            services_.telemetry->SetCurrentImageAgeMs(imageAgeMs_);
        }
    }
}

void BlackHoleGpuModule::Render(const grav_sim::RenderContext &render)
{
    if (!resourcesActive_ || render.framebufferWidth <= 0 || render.framebufferHeight <= 0)
    {
        return;
    }

    if (services_.sharedGrid != nullptr && render.gridVisible && services_.sharedGrid->IsDirty())
    {
        grav_telemetry::ScopedGpuPassQuery deformQuery(
            services_.telemetry, grav_telemetry::GpuPassId::GridDeform, render.sceneGeneration);
        services_.sharedGrid->UpdateIfDirty(true);
    }

    grav_telemetry::ScopedGpuPassQuery drawQuery(
        services_.telemetry, grav_telemetry::GpuPassId::SceneDraw, render.sceneGeneration);

    if (services_.fullscreenPresenter != nullptr && outputTexture_ != 0)
    {
        services_.fullscreenPresenter->DrawTexture(outputTexture_, false);
    }

    // Optional diagnostic raster overlay of the shared deformed grid (Section 11)
    if (services_.sharedGrid != nullptr && render.gridVisible)
    {
        const float aspect = static_cast<float>(render.framebufferWidth) /
                             static_cast<float>(std::max(render.framebufferHeight, 1));
        const glm::mat4 view = camera_.ViewMatrix(kWorldMetersPerGm);
        const glm::mat4 proj = camera_.ProjectionMatrix(aspect, 0.5f, 5000.0f);
        services_.sharedGrid->Draw(view, proj, true, false);
    }
}

void BlackHoleGpuModule::OnResize(int framebufferWidth, int framebufferHeight)
{
    if (framebufferWidth != services_.framebufferWidth ||
        framebufferHeight != services_.framebufferHeight)
    {
        services_.framebufferWidth = framebufferWidth;
        services_.framebufferHeight = framebufferHeight;
        if (framebufferWidth > 0 && framebufferHeight > 0)
        {
            imageDirty_ = true;
        }
    }
}

void BlackHoleGpuModule::ResetCamera()
{
    camera_.Reset();
    imageDirty_ = true;
}

void BlackHoleGpuModule::SetComputeResolution(int width, int height)
{
    baseComputeWidth_ = std::clamp(width, 32, 1920);
    baseComputeHeight_ = std::clamp(height, 24, 1080);
    imageDirty_ = true;
}

void BlackHoleGpuModule::BeginExit() {}

void BlackHoleGpuModule::ReleaseResources()
{
    if (services_.resourceCensus != nullptr)
    {
        services_.resourceCensus->Unregister(texCensusId_);
        services_.resourceCensus->Unregister(camUboCensusId_);
        services_.resourceCensus->Unregister(diskUboCensusId_);
        services_.resourceCensus->Unregister(objUboCensusId_);
    }

    glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, 0);
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, 0);
    glBindBufferBase(GL_UNIFORM_BUFFER, 3, 0);

    if (outputTexture_ != 0)
    {
        glDeleteTextures(1, &outputTexture_);
        outputTexture_ = 0;
    }
    if (cameraUbo_ != 0)
    {
        glDeleteBuffers(1, &cameraUbo_);
        cameraUbo_ = 0;
    }
    if (diskUbo_ != 0)
    {
        glDeleteBuffers(1, &diskUbo_);
        diskUbo_ = 0;
    }
    if (objectsUbo_ != 0)
    {
        glDeleteBuffers(1, &objectsUbo_);
        objectsUbo_ = 0;
    }

    std::vector<BlackHoleSceneObject>().swap(sceneObjects_);
    allocatedTexW_ = 0;
    allocatedTexH_ = 0;
    computeProgramCanonical_ = 0;
    computeProgramLegacy_ = 0;
    resourcesActive_ = false;
}

grav_sim::ModuleRuntimeStats BlackHoleGpuModule::RuntimeStats() const
{
    grav_sim::ModuleRuntimeStats stats;
    stats.primaryCount = sceneObjects_.size();
    stats.primaryLabel = "Scene Objects (<=16 UBO)";
    stats.outputWidth = allocatedTexW_;
    stats.outputHeight = allocatedTexH_;
    stats.stepCount = dispatchCount_;
    stats.simulationOrAffineParam = static_cast<double>(dLambdaMeters_);
    stats.paramLabel = "Geodesic Step dLambda (m)";
    stats.usesSharedBlackHole = true;
    stats.gridOverlayDiagnosticOnly = true;
    return stats;
}

void BlackHoleGpuModule::DrawInteractiveControls(bool configLocked)
{
    ImGui::BeginDisabled(configLocked);

    if (ImGui::BeginCombo("GPU Geodesic Solver",
                          solvers_[activeSolverIndex_].displayName.c_str()))
    {
        for (std::size_t i = 0; i < solvers_.size(); ++i)
        {
            const bool selected = (i == activeSolverIndex_);
            if (ImGui::Selectable(solvers_[i].displayName.c_str(), selected))
            {
                activeSolverIndex_ = i;
                imageDirty_ = true;
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    constexpr int kWidths[] = {200, 360, 640, 960};
    constexpr int kHeights[] = {150, 270, 480, 540};
    const char *kResLabels[] = {"200 x 150 (Original black_hole.cpp)", "360 x 270 (Balanced)",
                                "640 x 480 (High)", "960 x 540 (HD Half)"};
    int currentResIdx = 1;
    for (int i = 0; i < 4; ++i)
    {
        if (baseComputeWidth_ == kWidths[i] && baseComputeHeight_ == kHeights[i])
        {
            currentResIdx = i;
        }
    }
    if (ImGui::Combo("Compute Resolution", &currentResIdx, kResLabels, 4))
    {
        SetComputeResolution(kWidths[currentResIdx], kHeights[currentResIdx]);
    }

    if (ImGui::Checkbox("Dynamic Half-Res LOD While Orbiting", &enableMovingLod_))
    {
        imageDirty_ = true;
    }
    if (ImGui::Checkbox("Accretion Disk (2.2..5.2 r_s)", &diskEnabled_))
    {
        imageDirty_ = true;
    }
    if (ImGui::SliderFloat("Affine Step dLambda (m)", &dLambdaMeters_, 2.0e6f, 5.0e7f, "%.2e",
                           ImGuiSliderFlags_Logarithmic))
    {
        imageDirty_ = true;
    }
    if (ImGui::SliderInt("Max Geodesic Steps", &maxStepsStatic_, 4000, 60000))
    {
        imageDirty_ = true;
    }

    if (services_.sharedBlackHole != nullptr)
    {
        ImGui::SeparatorText("Shared Black Hole Parameters");
        float massMul = static_cast<float>(services_.sharedBlackHole->massKg / kSagittariusAMassKg);
        if (ImGui::SliderFloat("BH Mass (x SagA*)", &massMul, 0.25f, 3.0f, "%.2fx"))
        {
            services_.sharedBlackHole->SetMassKg(static_cast<double>(massMul) *
                                                 kSagittariusAMassKg);
        }
        ImGui::Text("r_s = %.3e m", services_.sharedBlackHole->schwarzschildRadiusMeters);
    }

    ImGui::SeparatorText("Experimental Hybrid Motion");
    ImGui::Checkbox("Legacy Newtonian Object Motion (formerly G in black_hole.cpp)",
                    &legacyGravityMotionEnabled_);

    ImGui::EndDisabled();
    ImGui::TextDisabled("Note: Grid overlay (G) is an unlensed diagnostic raster visualization.");
}

std::string BlackHoleGpuModule::ControlsHelpText() const
{
    return "Left-Drag: Orbit around Black Hole | Scroll: Zoom | G: Diagnostic Grid Overlay | "
           "C: Reset Camera";
}

} // namespace grav_models

