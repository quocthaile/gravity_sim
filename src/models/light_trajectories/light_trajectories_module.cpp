#include "models/light_trajectories/light_trajectories_module.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <random>

namespace grav_models
{
namespace
{
// Matches GridConfig::BlackHoleWorldGm(): 1 world unit = 1.0e9 meters (1 Gm)
constexpr float kMetersPerWorldUnit = 1.0e9f;
constexpr int kCircleSegments = 128;

glm::vec3 PlanarMetersToWorldXz(double xMeters, double yMeters, float yOffsetWorld = 0.05f)
{
    const double inv = 1.0 / static_cast<double>(kMetersPerWorldUnit);
    // Maps planar (x, y) -> right-handed 3D XZ (x, yOffset, -y) so +X is East and +y_planar is North (-Z)
    return glm::vec3(static_cast<float>(xMeters * inv), yOffsetWorld,
                     static_cast<float>(-yMeters * inv));
}
} // namespace

void BoundedTrailRing::ConfigureCapacity(std::size_t capacity)
{
    const std::size_t safeCap = std::clamp<std::size_t>(capacity, 16, 8192);
    buffer_.assign(safeCap, glm::vec2(0.0f));
    head_ = 0;
    count_ = 0;
}

void BoundedTrailRing::Clear()
{
    head_ = 0;
    count_ = 0;
}

void BoundedTrailRing::Push(const glm::vec2 &posMeters)
{
    if (buffer_.empty())
    {
        return;
    }
    const std::size_t cap = buffer_.size();
    if (count_ < cap)
    {
        buffer_[(head_ + count_) % cap] = posMeters;
        ++count_;
    }
    else
    {
        buffer_[head_] = posMeters;
        head_ = (head_ + 1) % cap;
    }
}

glm::vec2 BoundedTrailRing::AtChronological(std::size_t index) const
{
    const std::size_t cap = buffer_.size();
    return buffer_[(head_ + index) % cap];
}

void BoundedTrailRing::ReleaseMemory()
{
    std::vector<glm::vec2>().swap(buffer_);
    head_ = 0;
    count_ = 0;
}

LightTrajectoriesModule::LightTrajectoriesModule()
{
    presets_ = {
        {
            "f2_parallel_impact_bundle",
            "Parallel Impact-Parameter Bundle (28 Rays across Photon Sphere)",
            "Proposed preset built on 2D_lensing.cpp solver (b in [-6.2 rs, +6.2 rs])",
            "Shows horizon capture (|b| < 2.60 rs), photon-sphere looping, and weak deflection.",
            1,
        },
        {
            "f2_critical_single_ray_source",
            "Critical Single Ray (Commented Example from 2D_lensing.cpp:388)",
            "black_hole/2D_lensing.cpp:388: pos=(-1e11, 3.2760630272e10), vel=(c, 0)",
            "Exact single-ray initial condition commented in upstream 2D_lensing.cpp.",
            1,
        },
        {
            "f2_isotropic_point_source_seed42",
            "Point-Source Fan Bundle (36 Rays, seed 42)",
            "Proposed isotropic/fan point-source bundle around Sagittarius A* (seed 42)",
            "Demonstrates relativistic lensing caustics and Einstein-ring wrapping.",
            1,
        },
    };

    solvers_ = {
        {
            "rk4_canonical_v1",
            "Schwarzschild Null-Geodesic 4-Stage RK4 (Dimensionless Canonical)",
            "CPU Double-Precision RK4 + Core-Profile Batched VBO",
            grav_sim::SolverValidationStatus::NumericalValidated,
            "Integrates du/dh = v, dv/dh = -1.5*|u x v|^2/|u|^5 * u with exact horizon crossing.",
        },
        {
            "rk4_planar_legacy_v1",
            "Schwarzschild Planar Polar 4-Stage RK4 (2D_lensing.cpp)",
            "CPU Double-Precision Polar RK4 + Core-Profile Batched VBO",
            grav_sim::SolverValidationStatus::SourceCompatibleLegacy,
            "Preserves exact polar equations from 2D_lensing.cpp with separated trail ring buffer.",
        },
    };
}

LightTrajectoriesModule::~LightTrajectoriesModule() { ReleaseResources(); }

bool LightTrajectoriesModule::SetActiveSolverById(const std::string &solverId)
{
    for (std::size_t i = 0; i < solvers_.size(); ++i)
    {
        if (solvers_[i].id == solverId)
        {
            activeSolverIndex_ = i;
            return true;
        }
    }
    return false;
}

void LightTrajectoriesModule::BuildPresetRays(std::size_t presetIndex)
{
    rays_.clear();
    trails_.clear();

    const BlackHoleParameters bh =
        services_.sharedBlackHole ? *services_.sharedBlackHole : BlackHoleParameters{};
    const double rs = std::max(bh.schwarzschildRadiusMeters, 1.0);

    // Note: relPosMeters is relative to BH center in planar coordinates; trail stores world planar (x, y)
    auto pushRelRay = [&](const glm::dvec2 &relPosMeters, const glm::dvec2 &velMps)
    {
        PlanarRayState st = InitializePlanarRay(relPosMeters, velMps, rs);
        rays_.push_back(st);
        BoundedTrailRing ring;
        ring.ConfigureCapacity(trailCapacity_);
        ring.Push(glm::vec2(static_cast<float>(bh.positionMeters.x + st.x),
                            static_cast<float>(-bh.positionMeters.z + st.y)));
        trails_.push_back(std::move(ring));
    };

    if (presetIndex == 1)
    {
        // Exact commented example from 2D_lensing.cpp:388 (scaled relative to current BH rs if mass changed)
        const double rsRatio = rs / ComputeSchwarzschildRadiusMeters(kSagittariusAMassKg);
        pushRelRay(glm::dvec2(-1.0e11 * rsRatio, 3.27606302719999999e10 * rsRatio),
                   glm::dvec2(kSpeedOfLightMps, 0.0));
    }
    else if (presetIndex == 0)
    {
        constexpr int kBundleCount = 28;
        const double startX = -8.8 * rs;
        for (int i = 0; i < kBundleCount; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(kBundleCount - 1);
            double b = (-6.2 + 12.4 * t) * rs;
            if (std::fabs(b) < 0.12 * rs)
            {
                b += (b >= 0.0 ? 0.15 : -0.15) * rs;
            }
            pushRelRay(glm::dvec2(startX, b), glm::dvec2(kSpeedOfLightMps, 0.0));
        }
        // Also include two near-critical photon-sphere grazing rays (+/- 2.602 rs)
        pushRelRay(glm::dvec2(startX, 2.5985 * rs), glm::dvec2(kSpeedOfLightMps, 0.0));
        pushRelRay(glm::dvec2(startX, -2.5985 * rs), glm::dvec2(kSpeedOfLightMps, 0.0));
    }
    else
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> jitter(-0.004, 0.004);
        constexpr int kFanCount = 36;
        const glm::dvec2 emitter(-7.5 * rs, 2.1 * rs);
        const double baseAngle = std::atan2(-emitter.y, -emitter.x);
        for (int i = 0; i < kFanCount; ++i)
        {
            const double frac =
                (static_cast<double>(i) / static_cast<double>(kFanCount - 1)) - 0.5;
            const double angle = baseAngle + frac * 1.35 + jitter(rng);
            pushRelRay(emitter,
                       glm::dvec2(kSpeedOfLightMps * std::cos(angle),
                                  kSpeedOfLightMps * std::sin(angle)));
        }
    }

    stepCount_ = 0;
    cumulativeAffine_ = 0.0;
    maxNullRelError_ = 0.0;
    maxAngMomRelError_ = 0.0;
    trailGeometryDirty_ = true;
}

void LightTrajectoriesModule::RebuildHorizonGeometry()
{
    const BlackHoleParameters bh =
        services_.sharedBlackHole ? *services_.sharedBlackHole : BlackHoleParameters{};
    const double rs = std::max(bh.schwarzschildRadiusMeters, 1.0);
    const double cx = bh.positionMeters.x;
    const double cyPlanar = -bh.positionMeters.z;

    std::vector<TrailVertex> verts;
    verts.reserve(static_cast<std::size_t>(kCircleSegments + 2) * 3);

    // 1. Event horizon triangle fan (r = rs)
    verts.push_back(
        {PlanarMetersToWorldXz(cx, cyPlanar, 0.02f), glm::vec4(0.03f, 0.01f, 0.02f, 0.96f)});
    for (int i = 0; i <= kCircleSegments; ++i)
    {
        const double a =
            2.0 * glm::pi<double>() * static_cast<double>(i) / static_cast<double>(kCircleSegments);
        const double px = cx + rs * std::cos(a);
        const double py = cyPlanar + rs * std::sin(a);
        verts.push_back(
            {PlanarMetersToWorldXz(px, py, 0.02f), glm::vec4(0.35f, 0.03f, 0.04f, 0.96f)});
    }
    horizonFanVertexCount_ = static_cast<GLsizei>(verts.size());

    // 2. Event horizon bright red rim (LINE_LOOP, r = rs)
    horizonRimFirst_ = static_cast<GLsizei>(verts.size());
    for (int i = 0; i < kCircleSegments; ++i)
    {
        const double a =
            2.0 * glm::pi<double>() * static_cast<double>(i) / static_cast<double>(kCircleSegments);
        verts.push_back({PlanarMetersToWorldXz(cx + rs * std::cos(a), cyPlanar + rs * std::sin(a),
                                               0.04f),
                         glm::vec4(1.0f, 0.22f, 0.16f, 1.0f)});
    }
    horizonRimCount_ = static_cast<GLsizei>(verts.size()) - horizonRimFirst_;

    // 3. Photon sphere ring (LINE_LOOP, r = 1.5 * rs)
    photonSphereFirst_ = static_cast<GLsizei>(verts.size());
    const double rPhoton = 1.5 * rs;
    for (int i = 0; i < kCircleSegments; ++i)
    {
        const double a =
            2.0 * glm::pi<double>() * static_cast<double>(i) / static_cast<double>(kCircleSegments);
        verts.push_back(
            {PlanarMetersToWorldXz(cx + rPhoton * std::cos(a), cyPlanar + rPhoton * std::sin(a),
                                   0.03f),
             glm::vec4(1.0f, 0.78f, 0.25f, 0.60f)});
    }
    photonSphereCount_ = static_cast<GLsizei>(verts.size()) - photonSphereFirst_;

    const std::size_t byteSize = verts.size() * sizeof(TrailVertex);
    glBindBuffer(GL_ARRAY_BUFFER, horizonVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(byteSize), verts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (services_.resourceCensus != nullptr && horizonVboCensusId_ != 0)
    {
        services_.resourceCensus->UpdateBytes(horizonVboCensusId_, horizonVbo_, byteSize, byteSize);
    }
}

void LightTrajectoriesModule::SyncBlackHoleToGridAndHorizon()
{
    if (services_.sharedBlackHole == nullptr)
    {
        return;
    }

    lastObservedBhRevision_ = services_.sharedBlackHole->revision;
    if (services_.sharedGrid != nullptr)
    {
        const grav_render::GridConfig cfg = grav_render::GridConfig::BlackHoleWorldGm();
        services_.sharedGrid->Configure(cfg);
        const std::vector<grav_render::GridSource> bhSrc = {
            services_.sharedBlackHole->ToGridSource(cfg.metersPerWorldUnit)};
        services_.sharedGrid->SetCpuSources(bhSrc, lastObservedBhRevision_);
    }

    RebuildHorizonGeometry();
}

void LightTrajectoriesModule::UploadTrailGeometry()
{
    cpuScratchVertices_.clear();

    // Estimate max vertices: 2 * (trailLen - 1) for GL_LINES + 1 per ray head
    std::size_t neededVerts = rays_.size();
    for (const auto &tr : trails_)
    {
        if (tr.Size() >= 2)
        {
            neededVerts += (tr.Size() - 1) * 2;
        }
    }
    cpuScratchVertices_.reserve(neededVerts);

    // 1. Pack all trail segments into a single GL_LINES batch with smooth alpha fade
    for (std::size_t rIdx = 0; rIdx < trails_.size(); ++rIdx)
    {
        const BoundedTrailRing &ring = trails_[rIdx];
        const std::size_t n = ring.Size();
        if (n < 2)
        {
            continue;
        }

        const bool captured = rays_[rIdx].captured;
        const glm::vec3 baseRgb =
            captured ? glm::vec3(1.0f, 0.55f, 0.38f) : glm::vec3(0.95f, 0.98f, 1.0f);

        const float denom = static_cast<float>(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i)
        {
            const float a0 = std::max(static_cast<float>(i) / denom, 0.05f);
            const float a1 = std::max(static_cast<float>(i + 1) / denom, 0.05f);
            const glm::vec2 p0 = ring.AtChronological(i);
            const glm::vec2 p1 = ring.AtChronological(i + 1);

            cpuScratchVertices_.push_back(
                {PlanarMetersToWorldXz(p0.x, p0.y, 0.06f), glm::vec4(baseRgb, a0)});
            cpuScratchVertices_.push_back(
                {PlanarMetersToWorldXz(p1.x, p1.y, 0.06f), glm::vec4(baseRgb, a1)});
        }
    }
    uploadedLineVertexCount_ = static_cast<GLsizei>(cpuScratchVertices_.size());

    // 2. Pack all ray heads into a single GL_POINTS batch right after the line segments
    const BlackHoleParameters bh =
        services_.sharedBlackHole ? *services_.sharedBlackHole : BlackHoleParameters{};
    uploadedHeadFirst_ = uploadedLineVertexCount_;
    for (const auto &ray : rays_)
    {
        const double wx = bh.positionMeters.x + ray.x;
        const double wy = -bh.positionMeters.z + ray.y;
        const glm::vec4 headColor = ray.captured ? glm::vec4(1.0f, 0.25f, 0.15f, 0.85f)
                                                 : glm::vec4(1.0f, 0.88f, 0.25f, 1.0f);
        cpuScratchVertices_.push_back({PlanarMetersToWorldXz(wx, wy, 0.08f), headColor});
    }
    uploadedHeadCount_ = static_cast<GLsizei>(cpuScratchVertices_.size()) - uploadedHeadFirst_;

    const std::size_t requiredCap = std::max<std::size_t>(cpuScratchVertices_.size(), 256);
    glBindBuffer(GL_ARRAY_BUFFER, trailVbo_);
    if (requiredCap > trailVboCapacityVertices_)
    {
        // Bounded to at most maxRays * (2 * trailCapacity + 1) vertices
        trailVboCapacityVertices_ = requiredCap;
        const std::size_t allocBytes = trailVboCapacityVertices_ * sizeof(TrailVertex);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(allocBytes), nullptr,
                     GL_DYNAMIC_DRAW);
    }

    const std::size_t logicalBytes = cpuScratchVertices_.size() * sizeof(TrailVertex);
    if (logicalBytes > 0)
    {
        glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(logicalBytes),
                        cpuScratchVertices_.data());
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (services_.resourceCensus != nullptr && trailVboCensusId_ != 0)
    {
        services_.resourceCensus->UpdateBytes(
            trailVboCensusId_, trailVbo_, trailVboCapacityVertices_ * sizeof(TrailVertex),
            logicalBytes);
    }
    trailGeometryDirty_ = false;
}

void LightTrajectoriesModule::OnEnter(const grav_sim::ModuleServices &services,
                                      std::size_t presetIndex)
{
    services_ = services;
    activePresetIndex_ = presetIndex < presets_.size() ? presetIndex : 0;

    if (!camerasConfigured_)
    {
        // Orbit camera operates directly in world Gm units (1 unit = 1e9 m)
        orbitCamera_.ConfigureDefaults(glm::vec3(0.0f, 0.0f, 0.0f), 195.0f, 18.0f, 1800.0f,
                                       1.15f, 0.98f, 48.0f, true);
        orthoCamera_.ConfigureDefaults(145.0f, 10.0f, 1500.0f);
        camerasConfigured_ = true;
    }

    try
    {
        trailProgram_ = services_.shaderCache->GetOrCreateGraphics(
            "f2:trail_render", "shaders/trajectories/trail.vert",
            "shaders/trajectories/trail.frag");
        locViewProj_ = glGetUniformLocation(trailProgram_, "u_viewProj");

        auto setupTrailVao = [](GLuint &vao, GLuint &vbo)
        {
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TrailVertex), nullptr);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(TrailVertex),
                                  reinterpret_cast<const void *>(sizeof(glm::vec3)));
            glEnableVertexAttribArray(1);
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        };

        setupTrailVao(horizonVao_, horizonVbo_);
        setupTrailVao(trailVao_, trailVbo_);
        trailVboCapacityVertices_ = 0;

        if (services_.resourceCensus != nullptr)
        {
            horizonVaoCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::VertexArray, horizonVao_, 0, 0,
                "scene:f2_trajectories", "HorizonVAO", services_.sceneGeneration);
            horizonVboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::GeometryBuffer, horizonVbo_, 0, 0,
                "scene:f2_trajectories", "HorizonVBO", services_.sceneGeneration);
            trailVaoCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::VertexArray, trailVao_, 0, 0,
                "scene:f2_trajectories", "TrailsVAO", services_.sceneGeneration);
            trailVboCensusId_ = services_.resourceCensus->Register(
                grav_render::ResourceCategory::GeometryBuffer, trailVbo_, 0, 0,
                "scene:f2_trajectories", "TrailsVBO", services_.sceneGeneration);
        }

        SyncBlackHoleToGridAndHorizon();
        BuildPresetRays(activePresetIndex_);
        UploadTrailGeometry();
        resourcesActive_ = true;
    }
    catch (...)
    {
        ReleaseResources();
        throw;
    }
}

void LightTrajectoriesModule::HandleInput(const grav_sim::ModuleInputState &input,
                                          double wallDeltaSeconds)
{
    (void)wallDeltaSeconds;
    if (!resourcesActive_)
    {
        return;
    }
    if (use2DOrthoCamera_)
    {
        orthoCamera_.HandleInput(input, services_.framebufferWidth, services_.framebufferHeight);
    }
    else
    {
        orbitCamera_.HandleInput(input);
    }
}

void LightTrajectoriesModule::Update(const grav_sim::StepContext &step)
{
    if (!resourcesActive_)
    {
        return;
    }

    if (services_.sharedBlackHole != nullptr &&
        services_.sharedBlackHole->revision != lastObservedBhRevision_)
    {
        SyncBlackHoleToGridAndHorizon();
        BuildPresetRays(activePresetIndex_);
    }

    const bool shouldStep = (!step.paused || step.singleStepRequested);
    if (shouldStep && stepCount_ < maxAffineSteps_)
    {
        const BlackHoleParameters bh =
            services_.sharedBlackHole ? *services_.sharedBlackHole : BlackHoleParameters{};
        const double rs = std::max(bh.schwarzschildRadiusMeters, 1.0);
        const double escapeR = std::max(30.0 * rs, 3.5e11);
        const PlanarGeodesicSolverKind kind =
            (activeSolverIndex_ == 0) ? PlanarGeodesicSolverKind::CanonicalDimensionlessRk4
                                      : PlanarGeodesicSolverKind::LegacyPolarRk4;

        const int loops = step.singleStepRequested ? 1 : std::max(substepsPerFrame_, 1);
        std::uint64_t executed = 0;

        for (int s = 0; s < loops && stepCount_ < maxAffineSteps_; ++s)
        {
            bool anyActive = false;
            for (std::size_t i = 0; i < rays_.size(); ++i)
            {
                if (rays_[i].captured || rays_[i].escaped)
                {
                    continue;
                }
                anyActive = true;
                StepPlanarRay(rays_[i], static_cast<double>(dLambdaAffine_), rs, escapeR, kind);
                trails_[i].Push(glm::vec2(static_cast<float>(bh.positionMeters.x + rays_[i].x),
                                          static_cast<float>(-bh.positionMeters.z + rays_[i].y)));

                if (!rays_[i].captured)
                {
                    maxNullRelError_ = std::max(maxNullRelError_,
                                                EvaluatePlanarNullRelativeError(rays_[i], rs, kind));
                    maxAngMomRelError_ = std::max(
                        maxAngMomRelError_,
                        EvaluatePlanarAngularMomentumRelativeError(rays_[i], rs, kind));
                }
            }
            if (!anyActive)
            {
                break;
            }
            ++stepCount_;
            cumulativeAffine_ += static_cast<double>(dLambdaAffine_);
            ++executed;
            trailGeometryDirty_ = true;
        }

        if (executed > 0 && services_.telemetry != nullptr)
        {
            services_.telemetry->RecordModelSteps(executed);
        }
    }

    if (trailGeometryDirty_)
    {
        UploadTrailGeometry();
    }
}

void LightTrajectoriesModule::Render(const grav_sim::RenderContext &render)
{
    if (!resourcesActive_ || render.framebufferWidth <= 0 || render.framebufferHeight <= 0)
    {
        return;
    }

    const float aspect = static_cast<float>(render.framebufferWidth) /
                         static_cast<float>(std::max(render.framebufferHeight, 1));
    const glm::mat4 view =
        use2DOrthoCamera_ ? orthoCamera_.ViewMatrix() : orbitCamera_.ViewMatrix(1.0f);
    const glm::mat4 proj = use2DOrthoCamera_
                               ? orthoCamera_.ProjectionMatrix(aspect)
                               : orbitCamera_.ProjectionMatrix(aspect, 0.5f, 6000.0f);
    const glm::mat4 viewProj = proj * view;

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

    lastFrameDrawCalls_ = 0;

    if (services_.sharedGrid != nullptr)
    {
        services_.sharedGrid->Draw(view, proj, render.gridVisible, true);
        if (render.gridVisible)
        {
            ++lastFrameDrawCalls_;
        }
    }

    // Draw event horizon disk, rim, and photon-sphere marker
    glUseProgram(trailProgram_);
    glUniformMatrix4fv(locViewProj_, 1, GL_FALSE, glm::value_ptr(viewProj));

    glBindVertexArray(horizonVao_);
    if (horizonFanVertexCount_ > 0)
    {
        glDrawArrays(GL_TRIANGLE_FAN, 0, horizonFanVertexCount_);
        ++lastFrameDrawCalls_;
    }
    if (horizonRimCount_ > 0)
    {
        glDrawArrays(GL_LINE_LOOP, horizonRimFirst_, horizonRimCount_);
        ++lastFrameDrawCalls_;
    }
    if (photonSphereCount_ > 0)
    {
        glDrawArrays(GL_LINE_LOOP, photonSphereFirst_, photonSphereCount_);
        ++lastFrameDrawCalls_;
    }

    // Draw ALL ray trails in a single batched draw call + ALL ray heads in a single draw call
    glBindVertexArray(trailVao_);
    if (uploadedLineVertexCount_ > 0)
    {
        glDrawArrays(GL_LINES, 0, uploadedLineVertexCount_);
        ++lastFrameDrawCalls_;
    }
    if (uploadedHeadCount_ > 0)
    {
        glPointSize(5.0f);
        glDrawArrays(GL_POINTS, uploadedHeadFirst_, uploadedHeadCount_);
        glPointSize(1.0f);
        ++lastFrameDrawCalls_;
    }
    glBindVertexArray(0);
}

void LightTrajectoriesModule::OnResize(int framebufferWidth, int framebufferHeight)
{
    services_.framebufferWidth = framebufferWidth;
    services_.framebufferHeight = framebufferHeight;
}

void LightTrajectoriesModule::ResetCamera()
{
    if (use2DOrthoCamera_)
        orthoCamera_.Reset();
    else
        orbitCamera_.Reset();
}

std::uint64_t LightTrajectoriesModule::CameraRevision() const
{
    return use2DOrthoCamera_ ? orthoCamera_.Revision() : orbitCamera_.Revision();
}

void LightTrajectoriesModule::BeginExit() {}

void LightTrajectoriesModule::ReleaseResources()
{
    if (services_.resourceCensus != nullptr)
    {
        services_.resourceCensus->Unregister(horizonVaoCensusId_);
        services_.resourceCensus->Unregister(horizonVboCensusId_);
        services_.resourceCensus->Unregister(trailVaoCensusId_);
        services_.resourceCensus->Unregister(trailVboCensusId_);
    }

    if (horizonVao_ != 0)
    {
        glDeleteVertexArrays(1, &horizonVao_);
        horizonVao_ = 0;
    }
    if (horizonVbo_ != 0)
    {
        glDeleteBuffers(1, &horizonVbo_);
        horizonVbo_ = 0;
    }
    if (trailVao_ != 0)
    {
        glDeleteVertexArrays(1, &trailVao_);
        trailVao_ = 0;
    }
    if (trailVbo_ != 0)
    {
        glDeleteBuffers(1, &trailVbo_);
        trailVbo_ = 0;
    }

    for (auto &ring : trails_)
    {
        ring.ReleaseMemory();
    }
    std::vector<BoundedTrailRing>().swap(trails_);
    std::vector<PlanarRayState>().swap(rays_);
    std::vector<TrailVertex>().swap(cpuScratchVertices_);

    trailVboCapacityVertices_ = 0;
    uploadedLineVertexCount_ = 0;
    uploadedHeadCount_ = 0;
    trailProgram_ = 0;
    resourcesActive_ = false;
}

std::size_t LightTrajectoriesModule::TotalAllocatedTrailPoints() const
{
    std::size_t sum = 0;
    for (const auto &ring : trails_)
    {
        sum += ring.Capacity();
    }
    return sum;
}

grav_sim::ModuleRuntimeStats LightTrajectoriesModule::RuntimeStats() const
{
    grav_sim::ModuleRuntimeStats stats;
    stats.primaryCount = rays_.size();
    stats.primaryLabel = "Photon Rays";
    stats.outputWidth = services_.framebufferWidth;
    stats.outputHeight = services_.framebufferHeight;
    stats.stepCount = stepCount_;
    stats.simulationOrAffineParam = cumulativeAffine_;
    stats.paramLabel = "Affine Parameter (lambda)";
    if (activePresetIndex_ == 2)
    {
        stats.hasSeed = true;
        stats.seed = 42;
    }
    stats.usesSharedBlackHole = true;
    return stats;
}

void LightTrajectoriesModule::DrawInteractiveControls(bool configLocked)
{
    ImGui::BeginDisabled(configLocked);

    ImGui::Checkbox("2D Top-Down Orthographic View (2D_lensing style)", &use2DOrthoCamera_);

    if (ImGui::BeginCombo("Geodesic Solver", solvers_[activeSolverIndex_].displayName.c_str()))
    {
        for (std::size_t i = 0; i < solvers_.size(); ++i)
        {
            const bool selected = (i == activeSolverIndex_);
            if (ImGui::Selectable(solvers_[i].displayName.c_str(), selected))
            {
                activeSolverIndex_ = i;
                BuildPresetRays(activePresetIndex_);
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::SliderFloat("Affine Step dLambda", &dLambdaAffine_, 0.1f, 5.0f, "%.2f");
    ImGui::SliderInt("Substeps / Frame", &substepsPerFrame_, 1, 24);

    int capInt = static_cast<int>(trailCapacity_);
    if (ImGui::SliderInt("Bounded Trail Ring Capacity", &capInt, 128, 4096))
    {
        trailCapacity_ = static_cast<std::size_t>(capInt);
        for (auto &ring : trails_)
        {
            ring.ConfigureCapacity(trailCapacity_);
        }
        BuildPresetRays(activePresetIndex_);
    }

    if (services_.sharedBlackHole != nullptr)
    {
        ImGui::SeparatorText("Shared Black Hole (Sagittarius A*)");
        float massMul = static_cast<float>(services_.sharedBlackHole->massKg / kSagittariusAMassKg);
        if (ImGui::SliderFloat("Mass (x SagA*)", &massMul, 0.25f, 3.0f, "%.2fx"))
        {
            services_.sharedBlackHole->SetMassKg(static_cast<double>(massMul) *
                                                 kSagittariusAMassKg);
        }
        ImGui::Text("Schwarzschild Radius r_s: %.3e m (Photon sphere: 1.5 r_s)",
                    services_.sharedBlackHole->schwarzschildRadiusMeters);
    }

    ImGui::EndDisabled();

    ImGui::SeparatorText("Numerical Invariant Drift (Live)");
    ImGui::Text("Max |delta(E^2)| / E0^2 : %.3e", maxNullRelError_);
    ImGui::Text("Max |delta(L)|   / |L0| : %.3e", maxAngMomRelError_);
}

std::string LightTrajectoriesModule::ControlsHelpText() const
{
    return use2DOrthoCamera_
               ? "2D Ortho: Left/Middle-Drag: Pan | Scroll: Zoom | C: Reset View | P/N: Pause/Step"
               : "3D Orbit: Left-Drag: Orbit | Shift+Drag / Middle-Drag: Pan | Scroll: Zoom | "
                 "C: Reset View";
}

} // namespace grav_models
