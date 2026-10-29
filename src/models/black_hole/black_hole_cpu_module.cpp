#include "models/black_hole/black_hole_cpu_module.hpp"

#include <algorithm>
#include <cmath>
#include <imgui.h>

namespace grav_models
{
namespace
{
constexpr float kWorldMetersPerGm = 1.0e9f;
} // namespace

BlackHoleCpuModule::BlackHoleCpuModule()
{
    presets_ = {
        {
            "f5_cpu_geodesics_canonical",
            "CPU Geodesics: Accretion Disk + Stars (Canonical 4-Stage RK4)",
            "CPU-geodesic.cpp + canonical rk4_canonical_v1 parity scene",
            "Full 3D Schwarzschild null-geodesic CPU ray tracer matching F3 GPU scene.",
            1,
        },
        {
            "f5_geometric_silhouette",
            "Geometric Silhouette Submode (CPU-geodesic.cpp useGeodesics = false)",
            "black_hole/CPU-geodesic.cpp:444-455",
            "Euclidean straight-line ray-horizon sphere silhouette (no relativistic bending).",
            1,
        },
        {
            "f5_cpu_geodesics_legacy_source",
            "Legacy Source CPU Geodesic Silhouette (CPU-geodesic.cpp useGeodesics = true)",
            "black_hole/CPU-geodesic.cpp:456-468",
            "Preserves upstream spherical RK4 equations and red horizon silhouette visualization.",
            1,
        },
    };

    const std::string backend = CpuRayWorker::IsOpenMpCompiled()
                                    ? "CPU Multi-Thread Tile Worker (OpenMP Enabled)"
                                    : "CPU Multi-Thread Tile Worker (std::thread)";

    solvers_ = {
        {
            "rk4_canonical_v1",
            "Schwarzschild 3D Null-Geodesic 4-Stage RK4 (Canonical)",
            backend,
            grav_sim::SolverValidationStatus::NumericalValidated,
            "Double-precision dimensionless Cartesian 4-stage RK4 matching geodesic_rk4.comp.",
        },
        {
            "geometric_silhouette_v1",
            "Euclidean Horizon Sphere Silhouette (useGeodesics = false)",
            backend,
            grav_sim::SolverValidationStatus::SourceCompatibleLegacy,
            "Analytical straight-ray sphere test; distinct model from relativistic geodesics.",
        },
        {
            "rk4_spherical_legacy_v1",
            "Legacy Spherical 4-Stage RK4 (CPU-geodesic.cpp)",
            backend,
            grav_sim::SolverValidationStatus::SourceCompatibleLegacy,
            "Preserves upstream spherical initialization and radial angular term r*Omega^2.",
        },
    };
}

BlackHoleCpuModule::~BlackHoleCpuModule() { ReleaseResources(); }

bool BlackHoleCpuModule::SetActiveSolverById(const std::string &solverId)
{
    for (std::size_t i = 0; i < solvers_.size(); ++i)
    {
        if (solvers_[i].id == solverId)
        {
            activeSolverIndex_ = i;
            imageDirty_ = true;
            return true;
        }
    }
    return false;
}

void BlackHoleCpuModule::ApplyPresetDefaults(std::size_t presetIndex)
{
    const BlackHoleParameters bh =
        services_.sharedBlackHole ? *services_.sharedBlackHole : BlackHoleParameters{};
    const glm::vec3 bhPos(bh.positionMeters);
    const float rs = static_cast<float>(bh.schwarzschildRadiusMeters);

    sceneObjects_.clear();
    if (presetIndex == 0)
    {
        activeSolverIndex_ = 0; // rk4_canonical_v1
        diskEnabled_ = true;
        redSilhouetteStyle_ = false;
        maxSteps_ = 14000;
        sceneObjects_.push_back(
            {glm::vec4(bhPos + glm::vec3(4.0e11f, 0.0f, 0.0f), 4.0e10f),
             glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), 1.98892e30f, glm::vec3(0.0f)});
        sceneObjects_.push_back(
            {glm::vec4(bhPos + glm::vec3(0.0f, 0.0f, 4.0e11f), 4.0e10f),
             glm::vec4(1.0f, 0.15f, 0.1f, 1.0f), 1.98892e30f, glm::vec3(0.0f)});
        sceneObjects_.push_back({glm::vec4(bhPos, rs), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
                                 static_cast<float>(bh.massKg), glm::vec3(0.0f)});
    }
    else if (presetIndex == 1)
    {
        activeSolverIndex_ = 1; // geometric_silhouette_v1
        diskEnabled_ = false;
        redSilhouetteStyle_ = true;
        maxSteps_ = 1;
    }
    else
    {
        activeSolverIndex_ = 2; // rk4_spherical_legacy_v1
        diskEnabled_ = false;
        redSilhouetteStyle_ = true;
        maxSteps_ = 10000;
    }
    imageDirty_ = true;
}

void BlackHoleCpuModule::EnsureTexture(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        return;
    }
    if (texture_ != 0 && texW_ == width && texH_ == height)
    {
        return;
    }
    if (texture_ == 0)
    {
        glGenTextures(1, &texture_);
    }
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    texW_ = width;
    texH_ = height;
    const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    if (services_.resourceCensus != nullptr)
    {
        if (texCensusId_ == 0)
        {
            texCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::Texture, texture_, bytes, bytes,
                "scene:f5_black_hole_cpu", "BlackHoleCpuTextureRGBA8", services_.sceneGeneration);
        }
        else
        {
            services_.resourceCensus->UpdateBytes(texCensusId_, texture_, bytes, bytes);
        }
    }
}

void BlackHoleCpuModule::SyncSharedBlackHoleAndGrid()
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

void BlackHoleCpuModule::SubmitCpuGeodesicJob(std::uint64_t sceneGeneration)
{
    if (renderWidth_ <= 0 || renderHeight_ <= 0)
    {
        return;
    }

    BlackHoleSceneConfig cfg;
    glm::vec3 pos, right, up, fwd;
    camera_.CameraBasis(pos, right, up, fwd);
    cfg.camPosMeters = glm::dvec3(pos);
    cfg.camRight = glm::dvec3(right);
    cfg.camUp = glm::dvec3(up);
    cfg.camForward = glm::dvec3(fwd);
    cfg.tanHalfFov = std::tan(glm::radians(static_cast<double>(camera_.FovDegrees()) * 0.5));
    cfg.aspect = (services_.framebufferWidth > 0 && services_.framebufferHeight > 0)
                     ? (static_cast<double>(services_.framebufferWidth) /
                        static_cast<double>(services_.framebufferHeight))
                     : (static_cast<double>(renderWidth_) / static_cast<double>(renderHeight_));

    const BlackHoleParameters bh =
        services_.sharedBlackHole ? *services_.sharedBlackHole : BlackHoleParameters{};
    cfg.bhCenterMeters = bh.positionMeters;
    cfg.rsMeters = bh.schwarzschildRadiusMeters;
    cfg.dLambdaMeters = static_cast<double>(dLambdaMeters_);
    cfg.escapeRadiusMeters = 1.0e14;
    cfg.maxSteps = maxSteps_;
    cfg.diskEnabled = diskEnabled_;
    cfg.diskInnerMeters = bh.schwarzschildRadiusMeters * 2.2;
    cfg.diskOuterMeters = bh.schwarzschildRadiusMeters * 5.2;
    cfg.diskThicknessMeters = 1.0e9;
    cfg.redSilhouetteStyle = redSilhouetteStyle_;
    cfg.backgroundColor = redSilhouetteStyle_ ? glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
                                              : glm::vec4(0.02f, 0.02f, 0.05f, 1.0f);
    cfg.objects = sceneObjects_;

    Geodesic3DSolverKind solverKind = Geodesic3DSolverKind::CanonicalDimensionlessRk4;
    if (activeSolverIndex_ == 1)
    {
        solverKind = Geodesic3DSolverKind::GeometricSilhouette;
    }
    else if (activeSolverIndex_ == 2)
    {
        solverKind = Geodesic3DSolverKind::LegacySphericalRk4;
    }

    CpuImageJobMeta meta;
    meta.jobId = nextJobId_++;
    meta.sceneGeneration = sceneGeneration;
    meta.cameraRevision = camera_.Revision();
    meta.width = renderWidth_;
    meta.height = renderHeight_;

    lastSubmittedCameraRev_ = meta.cameraRevision;
    imageDirty_ = false;

    worker_.SubmitJob(
        meta,
        [=](int x0, int y0, int x1, int y1, int W, int H, std::uint8_t *rgba,
            const std::atomic<bool> &cancelFlag)
        {
            for (int y = y0; y < y1; ++y)
            {
                if (cancelFlag.load(std::memory_order_relaxed))
                {
                    return;
                }
                for (int x = x0; x < x1; ++x)
                {
                    const glm::dvec3 dir = PixelToCameraRayDirection(x, y, W, H, cfg);
                    const GeodesicTraceResult res =
                        TraceBlackHoleRayCPU(cfg.camPosMeters, dir, cfg, solverKind);
                    const std::size_t idx =
                        (static_cast<std::size_t>(y) * static_cast<std::size_t>(W) +
                         static_cast<std::size_t>(x)) *
                        4u;
                    rgba[idx + 0] =
                        static_cast<std::uint8_t>(std::clamp(res.color.r, 0.0f, 1.0f) * 255.0f);
                    rgba[idx + 1] =
                        static_cast<std::uint8_t>(std::clamp(res.color.g, 0.0f, 1.0f) * 255.0f);
                    rgba[idx + 2] =
                        static_cast<std::uint8_t>(std::clamp(res.color.b, 0.0f, 1.0f) * 255.0f);
                    rgba[idx + 3] = 255u;
                }
            }
        },
        24);
}

void BlackHoleCpuModule::OnEnter(const grav_sim::ModuleServices &services, std::size_t presetIndex)
{
    services_ = services;
    activePresetIndex_ = presetIndex < presets_.size() ? presetIndex : 0;

    if (!cameraConfigured_)
    {
        // Note: CPU-geodesic.cpp had minRadius = 1e12 clamping radius=6.34e10 on first scroll;
        // here we allow [1.6e10, 1e13] so initial radius 6.34194e10 m works smoothly without jump.
        camera_.ConfigureDefaults(glm::vec3(0.0f), 6.34194e10f, 1.6e10f, 1.0e13f, 0.0f, 1.40f,
                                  60.0f, true);
        cameraConfigured_ = true;
    }

    try
    {
        worker_.Start();
        ApplyPresetDefaults(activePresetIndex_);
        SyncSharedBlackHoleAndGrid();
        EnsureTexture(renderWidth_, renderHeight_);

        hasUploadedImage_ = false;
        completedImages_ = 0;
        imageAgeMs_ = 0.0;
        resourcesActive_ = true;
        SubmitCpuGeodesicJob(services_.sceneGeneration);
    }
    catch (...)
    {
        ReleaseResources();
        throw;
    }
}

void BlackHoleCpuModule::HandleInput(const grav_sim::ModuleInputState &input,
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

void BlackHoleCpuModule::Update(const grav_sim::StepContext &step)
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

    CpuImageResult completed;
    if (worker_.TryPopCompletedResult(completed, step.sceneGeneration, camera_.Revision(),
                                      renderWidth_, renderHeight_))
    {
        EnsureTexture(completed.meta.width, completed.meta.height);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, completed.meta.width, completed.meta.height,
                        GL_RGBA, GL_UNSIGNED_BYTE, completed.rgbaPixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        hasUploadedImage_ = true;
        ++completedImages_;
        imageAgeMs_ = 0.0;
        if (services_.telemetry != nullptr)
        {
            services_.telemetry->RecordImageEvaluation(completed.elapsedMs);
        }
    }
    else
    {
        imageAgeMs_ += step.wallDeltaSeconds * 1000.0;
        if (services_.telemetry != nullptr)
        {
            services_.telemetry->SetCurrentImageAgeMs(imageAgeMs_);
        }
    }

    const bool needNewTrace =
        imageDirty_ || (camera_.Revision() != lastSubmittedCameraRev_) ||
        step.singleStepRequested || (step.forceRecomputeStaticImage && worker_.IsIdle());
    if (needNewTrace && renderWidth_ > 0 && renderHeight_ > 0)
    {
        SubmitCpuGeodesicJob(step.sceneGeneration);
    }
}

void BlackHoleCpuModule::Render(const grav_sim::RenderContext &render)
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

    if (hasUploadedImage_ && services_.fullscreenPresenter != nullptr && texture_ != 0)
    {
        services_.fullscreenPresenter->DrawTexture(texture_, false);
    }

    if (services_.sharedGrid != nullptr && render.gridVisible)
    {
        const float aspect = static_cast<float>(render.framebufferWidth) /
                             static_cast<float>(std::max(render.framebufferHeight, 1));
        services_.sharedGrid->Draw(camera_.ViewMatrix(kWorldMetersPerGm),
                                   camera_.ProjectionMatrix(aspect, 0.5f, 5000.0f), true, false);
    }
}

void BlackHoleCpuModule::OnResize(int framebufferWidth, int framebufferHeight)
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

void BlackHoleCpuModule::ResetCamera()
{
    camera_.Reset();
    imageDirty_ = true;
}

void BlackHoleCpuModule::SetRenderResolution(int width, int height)
{
    renderWidth_ = std::clamp(width, 24, 960);
    renderHeight_ = std::clamp(height, 18, 720);
    imageDirty_ = true;
}

void BlackHoleCpuModule::BeginExit() { worker_.RequestCancel(); }

void BlackHoleCpuModule::ReleaseResources()
{
    worker_.StopAndJoin();

    if (services_.resourceCensus != nullptr)
    {
        services_.resourceCensus->Unregister(texCensusId_);
    }
    if (texture_ != 0)
    {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }

    std::vector<BlackHoleSceneObject>().swap(sceneObjects_);
    texW_ = 0;
    texH_ = 0;
    hasUploadedImage_ = false;
    resourcesActive_ = false;
}

grav_sim::ModuleRuntimeStats BlackHoleCpuModule::RuntimeStats() const
{
    grav_sim::ModuleRuntimeStats stats;
    stats.primaryCount =
        static_cast<std::size_t>(renderWidth_) * static_cast<std::size_t>(renderHeight_);
    stats.primaryLabel = "CPU Rays / Image";
    stats.outputWidth = renderWidth_;
    stats.outputHeight = renderHeight_;
    stats.stepCount = completedImages_;
    stats.simulationOrAffineParam = imageAgeMs_;
    stats.paramLabel = "Image Age (ms)";
    stats.usesSharedBlackHole = true;
    stats.gridOverlayDiagnosticOnly = true;
    stats.hasAsyncWorker = true;
    stats.workerBusy = worker_.IsBusy();
    stats.workerProgress = worker_.Progress();
    stats.workerThreadCount = worker_.ThreadCount();
    return stats;
}

void BlackHoleCpuModule::DrawInteractiveControls(bool configLocked)
{
    ImGui::BeginDisabled(configLocked);

    if (ImGui::BeginCombo("CPU Submode / Solver",
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

    constexpr int kW[] = {120, 200, 320, 480};
    constexpr int kH[] = {90, 150, 240, 360};
    const char *kLabels[] = {"120 x 90 (Fast Interactive)", "200 x 150 (Default CPU)",
                             "320 x 240 (Medium)", "480 x 360 (High Quality)"};
    int cur = 1;
    for (int i = 0; i < 4; ++i)
    {
        if (renderWidth_ == kW[i] && renderHeight_ == kH[i])
            cur = i;
    }
    if (ImGui::Combo("CPU Trace Resolution", &cur, kLabels, 4))
    {
        SetRenderResolution(kW[cur], kH[cur]);
    }

    if (activeSolverIndex_ != 1)
    {
        if (ImGui::Checkbox("Accretion Disk (2.2..5.2 r_s)", &diskEnabled_))
            imageDirty_ = true;
        if (ImGui::Checkbox("Red Silhouette Horizon Style (CPU-geodesic.cpp)", &redSilhouetteStyle_))
            imageDirty_ = true;
        if (ImGui::SliderFloat("Affine Step dLambda (m)", &dLambdaMeters_, 2.0e6f, 5.0e7f, "%.2e",
                               ImGuiSliderFlags_Logarithmic))
            imageDirty_ = true;
        if (ImGui::SliderInt("Max Geodesic Steps", &maxSteps_, 2000, 30000))
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
    }

    ImGui::EndDisabled();

    ImGui::Text("Worker Pool: %u threads | OpenMP: %s", worker_.ThreadCount(),
                CpuRayWorker::IsOpenMpCompiled() ? "Enabled" : "Not Compiled (std::thread active)");
    ImGui::ProgressBar(worker_.Progress(), ImVec2(-1.0f, 0.0f));
}

std::string BlackHoleCpuModule::ControlsHelpText() const
{
    return "Left-Drag: Orbit | Shift+Drag / Middle-Drag: Pan | Scroll: Zoom | Tab: Cycle Submode "
           "(Geodesics vs Geometric Silhouette)";
}

} // namespace grav_models

