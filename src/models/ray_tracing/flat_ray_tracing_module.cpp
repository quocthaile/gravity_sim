#include "models/ray_tracing/flat_ray_tracing_module.hpp"

#include <algorithm>
#include <cmath>
#include <imgui.h>
#include <limits>

namespace grav_models
{
namespace
{
bool IntersectSphereFlat(const glm::vec3 &origin, const glm::vec3 &dir, const glm::vec3 &center,
                         float radius, float &outT)
{
    const glm::vec3 oc = origin - center;
    const float a = glm::dot(dir, dir);
    const float b = 2.0f * glm::dot(oc, dir);
    const float c = glm::dot(oc, oc) - radius * radius;
    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f)
    {
        return false;
    }
    const float s = std::sqrt(disc);
    float t = (-b - s) / (2.0f * a);
    if (t < 1e-4f)
    {
        t = (-b + s) / (2.0f * a);
        if (t < 1e-4f)
        {
            return false;
        }
    }
    outT = t;
    return true;
}
} // namespace

FlatRayTracingModule::FlatRayTracingModule()
{
    presets_ = {
        {
            "f4_baseline_two_spheres",
            "Baseline Two-Sphere Ray Tracer (black_hole/ray_tracing.cpp)",
            "black_hole/ray_tracing.cpp:306-311",
            "Flat Euclidean space: ambient + diffuse + hard shadow ray; no gravitational lensing.",
            1,
        },
        {
            "f4_archive_ray_tracing2",
            "Archive Variant: Red/Blue Sphere Pair (Gravity_Sim.zip/ray_tracing2.cpp)",
            "Gravity_Sim.zip: Gravity_Sim/src/ray_tracing2.cpp",
            "Historical 2-sphere scene at z = -6 with hard shadow casting.",
            1,
        },
        {
            "f4_archive_holding_three_spheres",
            "Archive Variant: Three-Sphere Static (Gravity_Sim.zip/holding_file.cpp)",
            "Gravity_Sim.zip: Gravity_Sim/src/holding_file.cpp",
            "Large backdrop sphere (r=12) + two foreground spheres; inactive if(false) gravity preserved.",
            1,
        },
        {
            "f4_archive_working_bounce",
            "Archive Variant: Three-Sphere Constant-g Bounce (working_raytracing.cpp)",
            "Gravity_Sim.zip: Gravity_Sim/src/working_raytracing.cpp",
            "Active constant vertical acceleration and bounce floor from working_raytracing.cpp.",
            1,
        },
        {
            "f4_archive_multi_light_reflective",
            "Archive Variant: Recursive Reflective Spheres (Gravity_Sim.zip/black_hole.cpp)",
            "Gravity_Sim.zip: Gravity_Sim/src/black_hole.cpp (flat-space recursive ray tracer)",
            "Historical flat-space recursive ray tracer (depth 4) with emissive lights & reflections.",
            1,
        },
    };

    activeSolver_.id = "flat_ray_phong_shadow_v1";
    activeSolver_.displayName = "Euclidean Ray-Sphere Intersection + Shadow";
    activeSolver_.backendName = "CPU Multi-Thread Tile Worker";
    activeSolver_.status = grav_sim::SolverValidationStatus::SourceCompatibleLegacy;
    activeSolver_.notes =
        "Flat-space reference ray tracer; specular/emission fields active only in recursive archive preset.";
}

FlatRayTracingModule::~FlatRayTracingModule() { ReleaseResources(); }

void FlatRayTracingModule::LoadScenePreset(std::size_t presetIndex)
{
    spheres_.clear();
    pointLights_.clear();
    recursiveShadingMode_ = false;
    constantGravityBounceMode_ = false;
    backgroundColor_ = glm::vec3(0.0f, 0.0f, 0.1f);

    if (presetIndex == 0)
    {
        // Exact ray_tracing.cpp:307-311
        spheres_.push_back({glm::vec3(0.0f, 0.0f, -5.0f), 2.0f, glm::vec3(1.0f, 0.2f, 0.2f), 0.0f,
                            0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(3.0f, 0.0f, -7.0f), 1.5f, glm::vec3(0.2f, 1.0f, 0.2f), 0.0f,
                            0.0f, glm::vec3(0.0f)});
        pointLights_.push_back(glm::vec3(5.0f, 5.0f, 5.0f));
    }
    else if (presetIndex == 1)
    {
        // Gravity_Sim/src/ray_tracing2.cpp
        spheres_.push_back({glm::vec3(0.0f, 0.0f, -6.0f), 1.0f, glm::vec3(1.0f, 0.15f, 0.15f),
                            0.0f, 0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(2.0f, 0.0f, -6.0f), 1.0f, glm::vec3(0.15f, 0.35f, 1.0f),
                            0.0f, 0.0f, glm::vec3(0.0f)});
        pointLights_.push_back(glm::vec3(5.0f, 5.0f, 5.0f));
    }
    else if (presetIndex == 2)
    {
        // Gravity_Sim/src/holding_file.cpp
        spheres_.push_back({glm::vec3(0.0f, -5.0f, -19.0f), 12.0f, glm::vec3(0.85f, 0.25f, 0.25f),
                            0.0f, 0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(15.0f, -2.0f, -11.0f), 1.5f, glm::vec3(0.25f, 0.90f, 0.35f),
                            0.0f, 0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(-12.0f, -1.0f, -6.0f), 1.0f, glm::vec3(0.30f, 0.45f, 1.0f),
                            0.0f, 0.0f, glm::vec3(0.0f)});
        pointLights_.push_back(glm::vec3(5.0f, 5.0f, 5.0f));
    }
    else if (presetIndex == 3)
    {
        // Gravity_Sim/src/working_raytracing.cpp
        constantGravityBounceMode_ = true;
        spheres_.push_back({glm::vec3(0.0f, 2.5f, -9.0f), 2.0f, glm::vec3(0.95f, 0.25f, 0.25f),
                            0.0f, 0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(5.0f, 4.0f, -11.0f), 1.5f, glm::vec3(0.25f, 0.95f, 0.30f),
                            0.0f, 0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(-2.5f, 3.0f, -6.0f), 1.0f, glm::vec3(0.30f, 0.55f, 1.0f),
                            0.0f, 0.0f, glm::vec3(0.0f)});
        pointLights_.push_back(glm::vec3(5.0f, 8.0f, 5.0f));
    }
    else
    {
        // Gravity_Sim/src/black_hole.cpp (flat-space recursive ray tracer with emissive lights)
        recursiveShadingMode_ = true;
        backgroundColor_ = glm::vec3(0.02f, 0.02f, 0.05f);
        spheres_.push_back({glm::vec3(0.0f, 0.0f, -8.0f), 2.2f, glm::vec3(0.85f, 0.88f, 0.95f),
                            0.55f, 0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(-3.8f, -1.0f, -7.0f), 1.3f, glm::vec3(0.95f, 0.30f, 0.22f),
                            0.35f, 0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(3.6f, -0.8f, -7.2f), 1.4f, glm::vec3(0.25f, 0.70f, 1.0f),
                            0.35f, 0.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(0.0f, -102.5f, -8.0f), 100.0f, glm::vec3(0.35f, 0.38f, 0.44f),
                            0.25f, 0.0f, glm::vec3(0.0f)});
        // Emissive light spheres
        spheres_.push_back({glm::vec3(-5.0f, 5.5f, -3.0f), 0.65f, glm::vec3(1.0f, 0.92f, 0.75f),
                            0.0f, 1.0f, glm::vec3(0.0f)});
        spheres_.push_back({glm::vec3(5.5f, 4.5f, -4.0f), 0.65f, glm::vec3(0.75f, 0.88f, 1.0f),
                            0.0f, 1.0f, glm::vec3(0.0f)});
        pointLights_.push_back(glm::vec3(-5.0f, 5.5f, -3.0f));
        pointLights_.push_back(glm::vec3(5.5f, 4.5f, -4.0f));
    }

    activeSolver_.id = recursiveShadingMode_ ? "flat_ray_recursive_legacy_v1"
                                             : "flat_ray_phong_shadow_v1";
    activeSolver_.displayName = recursiveShadingMode_
                                    ? "Recursive Euclidean Ray Tracer (Depth 4, Archive)"
                                    : "Euclidean Ray-Sphere Intersection + Shadow";
    sceneDirty_ = true;
}

void FlatRayTracingModule::EnsureTexture(int width, int height)
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
                "scene:f4_flat_ray_cpu", "FlatRayCpuTextureRGBA8", services_.sceneGeneration);
        }
        else
        {
            services_.resourceCensus->UpdateBytes(texCensusId_, texture_, bytes, bytes);
        }
    }
}

glm::vec3 FlatRayTracingModule::TraceFlatSceneRay(const glm::vec3 &origin, const glm::vec3 &dir,
                                                  const std::vector<FlatSphere> &spheres,
                                                  const std::vector<glm::vec3> &lights,
                                                  const glm::vec3 &bgColor, bool recursiveMode,
                                                  int depth)
{
    float closest = std::numeric_limits<float>::infinity();
    const FlatSphere *hit = nullptr;

    for (const auto &sp : spheres)
    {
        float t = 0.0f;
        if (IntersectSphereFlat(origin, dir, sp.center, sp.radius, t) && t < closest)
        {
            closest = t;
            hit = &sp;
        }
    }

    if (hit == nullptr)
    {
        return bgColor;
    }

    if (hit->emission > 0.0f)
    {
        return hit->color * hit->emission;
    }

    const glm::vec3 hitPoint = origin + dir * closest;
    const glm::vec3 normal = glm::normalize(hitPoint - hit->center);

    constexpr float kAmbient = 0.10f;
    glm::vec3 litColor = hit->color * kAmbient;

    for (const auto &lightPos : lights)
    {
        const glm::vec3 toLight = lightPos - hitPoint;
        const float lightDist = glm::length(toLight);
        const glm::vec3 lightDir = lightDist > 1e-5f ? (toLight / lightDist) : glm::vec3(0, 1, 0);

        const glm::vec3 shadowOrigin = hitPoint + normal * 0.002f;
        bool inShadow = false;
        for (const auto &sp : spheres)
        {
            if (sp.emission > 0.0f)
            {
                continue;
            }
            float tShadow = 0.0f;
            if (IntersectSphereFlat(shadowOrigin, lightDir, sp.center, sp.radius, tShadow) &&
                tShadow < lightDist)
            {
                inShadow = true;
                break;
            }
        }

        if (!inShadow)
        {
            const float diff = std::max(glm::dot(normal, lightDir), 0.0f);
            litColor += hit->color * (diff * 0.90f / static_cast<float>(std::max<std::size_t>(lights.size(), 1)));
            if (recursiveMode)
            {
                const glm::vec3 viewDir = -dir;
                const glm::vec3 halfVec = glm::normalize(lightDir + viewDir);
                const float spec = std::pow(std::max(glm::dot(normal, halfVec), 0.0f), 48.0f);
                litColor += glm::vec3(0.35f * spec);
            }
        }
    }

    if (recursiveMode && hit->reflectivity > 0.01f && depth < 4)
    {
        const glm::vec3 reflDir = glm::normalize(glm::reflect(dir, normal));
        const glm::vec3 reflCol = TraceFlatSceneRay(hitPoint + normal * 0.002f, reflDir, spheres,
                                                    lights, bgColor, true, depth + 1);
        litColor = glm::mix(litColor, reflCol, hit->reflectivity);
    }

    return glm::clamp(litColor, glm::vec3(0.0f), glm::vec3(1.0f));
}

void FlatRayTracingModule::SubmitTraceJob(std::uint64_t sceneGeneration)
{
    if (renderWidth_ <= 0 || renderHeight_ <= 0)
    {
        return;
    }

    const std::vector<FlatSphere> spheresCopy = spheres_;
    const std::vector<glm::vec3> lightsCopy = pointLights_;
    const glm::vec3 bgCopy = backgroundColor_;
    const bool recursiveCopy = recursiveShadingMode_;

    const glm::vec3 camOrigin = camera_.Position();
    const glm::vec3 camFwd = glm::normalize(camera_.Front());
    const glm::vec3 camRight = glm::normalize(glm::cross(camFwd, camera_.Up()));
    const glm::vec3 camUp = glm::normalize(glm::cross(camRight, camFwd));
    const float tanHalfFov = std::tan(glm::radians(camera_.FovDegrees() * 0.5f));
    const float aspect =
        (services_.framebufferWidth > 0 && services_.framebufferHeight > 0)
            ? (static_cast<float>(services_.framebufferWidth) /
               static_cast<float>(services_.framebufferHeight))
            : (static_cast<float>(renderWidth_) / static_cast<float>(renderHeight_));

    CpuImageJobMeta meta;
    meta.jobId = nextJobId_++;
    meta.sceneGeneration = sceneGeneration;
    meta.cameraRevision = camera_.Revision();
    meta.width = renderWidth_;
    meta.height = renderHeight_;

    lastSubmittedCameraRev_ = meta.cameraRevision;
    sceneDirty_ = false;

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
                // Row y=0 is bottom (ndcY = -1), y=H-1 is top (ndcY = +1), matching GL texture UV
                const float ndcY =
                    2.0f * ((static_cast<float>(y) + 0.5f) / static_cast<float>(H)) - 1.0f;
                for (int x = x0; x < x1; ++x)
                {
                    const float ndcX =
                        2.0f * ((static_cast<float>(x) + 0.5f) / static_cast<float>(W)) - 1.0f;
                    const glm::vec3 dir = glm::normalize(ndcX * aspect * tanHalfFov * camRight +
                                                         ndcY * tanHalfFov * camUp + camFwd);
                    const glm::vec3 col = TraceFlatSceneRay(camOrigin, dir, spheresCopy, lightsCopy,
                                                            bgCopy, recursiveCopy, 0);
                    const std::size_t idx =
                        (static_cast<std::size_t>(y) * static_cast<std::size_t>(W) +
                         static_cast<std::size_t>(x)) *
                        4u;
                    rgba[idx + 0] = static_cast<std::uint8_t>(std::clamp(col.r, 0.0f, 1.0f) * 255.0f);
                    rgba[idx + 1] = static_cast<std::uint8_t>(std::clamp(col.g, 0.0f, 1.0f) * 255.0f);
                    rgba[idx + 2] = static_cast<std::uint8_t>(std::clamp(col.b, 0.0f, 1.0f) * 255.0f);
                    rgba[idx + 3] = 255u;
                }
            }
        },
        32);
}

void FlatRayTracingModule::OnEnter(const grav_sim::ModuleServices &services,
                                   std::size_t presetIndex)
{
    services_ = services;
    activePresetIndex_ = presetIndex < presets_.size() ? presetIndex : 0;

    if (!cameraConfigured_)
    {
        // Matches ray_tracing.cpp origin (0,0,0) looking along -Z with ~90 deg vertical scale (tanHalfFov=1 -> 90 deg)
        camera_.ConfigureDefaults(glm::vec3(0.0f, 0.0f, 0.0f), -90.0f, 0.0f, 8.0f, 2.0f, 75.0f,
                                  0.05f, 500.0f);
        cameraConfigured_ = true;
    }

    try
    {
        worker_.Start();
        LoadScenePreset(activePresetIndex_);
        EnsureTexture(renderWidth_, renderHeight_);

        // Decorative flat grid if user explicitly presses G (sourceCount = 0 -> flat plane)
        if (services_.sharedGrid != nullptr)
        {
            grav_render::GridConfig flatCfg = grav_render::GridConfig::BlackHoleWorldGm();
            flatCfg.gridExtentWorld = 60.0f;
            flatCfg.basePlaneOffsetWorld = -3.5f;
            services_.sharedGrid->Configure(flatCfg);
            services_.sharedGrid->SetCpuSources({}, services_.sceneGeneration);
        }

        hasUploadedImage_ = false;
        completedImages_ = 0;
        imageAgeMs_ = 0.0;
        resourcesActive_ = true;
        SubmitTraceJob(services_.sceneGeneration);
    }
    catch (...)
    {
        ReleaseResources();
        throw;
    }
}

void FlatRayTracingModule::HandleInput(const grav_sim::ModuleInputState &input,
                                       double wallDeltaSeconds)
{
    if (!resourcesActive_)
    {
        return;
    }
    if (camera_.HandleInput(input, wallDeltaSeconds, true))
    {
        sceneDirty_ = true;
    }
}

void FlatRayTracingModule::Update(const grav_sim::StepContext &step)
{
    if (!resourcesActive_)
    {
        return;
    }

    if (constantGravityBounceMode_ && (!step.paused || step.singleStepRequested))
    {
        const float dtSim = step.singleStepRequested
                                ? (1.0f / 60.0f)
                                : static_cast<float>(std::clamp(step.wallDeltaSeconds, 0.0, 0.05));
        for (auto &sp : spheres_)
        {
            sp.velocity.y -= 9.81f * dtSim;
            sp.center += sp.velocity * dtSim;
            const float floorY = -2.5f + sp.radius;
            if (sp.center.y < floorY)
            {
                sp.center.y = floorY;
                sp.velocity.y = std::fabs(sp.velocity.y) * 0.88f;
            }
        }
        sceneDirty_ = true;
        if (services_.telemetry != nullptr)
        {
            services_.telemetry->RecordModelSteps(1);
        }
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
        sceneDirty_ || (camera_.Revision() != lastSubmittedCameraRev_) ||
        step.singleStepRequested || (step.forceRecomputeStaticImage && worker_.IsIdle());
    if (needNewTrace && renderWidth_ > 0 && renderHeight_ > 0)
    {
        SubmitTraceJob(step.sceneGeneration);
    }
}

void FlatRayTracingModule::Render(const grav_sim::RenderContext &render)
{
    if (!resourcesActive_ || render.framebufferWidth <= 0 || render.framebufferHeight <= 0)
    {
        return;
    }

    grav_telemetry::ScopedGpuPassQuery drawQuery(
        services_.telemetry, grav_telemetry::GpuPassId::SceneDraw, render.sceneGeneration);

    if (hasUploadedImage_ && services_.fullscreenPresenter != nullptr && texture_ != 0)
    {
        services_.fullscreenPresenter->DrawTexture(texture_, false);
    }

    if (render.gridVisible && services_.sharedGrid != nullptr)
    {
        const float aspect = static_cast<float>(render.framebufferWidth) /
                             static_cast<float>(std::max(render.framebufferHeight, 1));
        services_.sharedGrid->Draw(camera_.ViewMatrix(), camera_.ProjectionMatrix(aspect), true,
                                   false);
    }
}

void FlatRayTracingModule::OnResize(int framebufferWidth, int framebufferHeight)
{
    if (framebufferWidth != services_.framebufferWidth ||
        framebufferHeight != services_.framebufferHeight)
    {
        services_.framebufferWidth = framebufferWidth;
        services_.framebufferHeight = framebufferHeight;
        if (framebufferWidth > 0 && framebufferHeight > 0)
        {
            sceneDirty_ = true;
        }
    }
}

void FlatRayTracingModule::ResetCamera()
{
    camera_.Reset();
    sceneDirty_ = true;
}

void FlatRayTracingModule::SetRenderResolution(int width, int height)
{
    renderWidth_ = std::clamp(width, 32, 1280);
    renderHeight_ = std::clamp(height, 24, 960);
    sceneDirty_ = true;
}

void FlatRayTracingModule::BeginExit() { worker_.RequestCancel(); }

void FlatRayTracingModule::ReleaseResources()
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

    std::vector<FlatSphere>().swap(spheres_);
    std::vector<glm::vec3>().swap(pointLights_);
    texW_ = 0;
    texH_ = 0;
    hasUploadedImage_ = false;
    resourcesActive_ = false;
}

grav_sim::ModuleRuntimeStats FlatRayTracingModule::RuntimeStats() const
{
    grav_sim::ModuleRuntimeStats stats;
    stats.primaryCount = spheres_.size();
    stats.primaryLabel = "Spheres";
    stats.outputWidth = renderWidth_;
    stats.outputHeight = renderHeight_;
    stats.stepCount = completedImages_;
    stats.simulationOrAffineParam = imageAgeMs_;
    stats.paramLabel = "Image Age (ms)";
    stats.hasAsyncWorker = true;
    stats.workerBusy = worker_.IsBusy();
    stats.workerProgress = worker_.Progress();
    stats.workerThreadCount = worker_.ThreadCount();
    return stats;
}

void FlatRayTracingModule::DrawInteractiveControls(bool configLocked)
{
    ImGui::BeginDisabled(configLocked);

    constexpr int kW[] = {160, 320, 480, 800};
    constexpr int kH[] = {120, 240, 360, 600};
    const char *kLabels[] = {"160 x 120 (Fast)", "320 x 240 (Interactive)", "480 x 360 (Medium)",
                             "800 x 600 (Original ray_tracing.cpp)"};
    int cur = 1;
    for (int i = 0; i < 4; ++i)
    {
        if (renderWidth_ == kW[i] && renderHeight_ == kH[i])
            cur = i;
    }
    if (ImGui::Combo("CPU Render Resolution", &cur, kLabels, 4))
    {
        SetRenderResolution(kW[cur], kH[cur]);
    }

    ImGui::EndDisabled();

    ImGui::Text("Async Worker Threads: %u (%s)", worker_.ThreadCount(),
                worker_.IsBusy() ? "Rendering tiles..." : "Idle");
    ImGui::ProgressBar(worker_.Progress(), ImVec2(-1.0f, 0.0f));
    ImGui::TextDisabled(
        "Flat Euclidean reference ray tracer. Grid is OFF by default (decorative flat plane if G).");
}

std::string FlatRayTracingModule::ControlsHelpText() const
{
    return "W/A/S/D, Space, Left Shift: Move Camera | Right-Drag: Look | Scroll: Zoom | "
           "Tab: Cycle Archive Scenes";
}

} // namespace grav_models

