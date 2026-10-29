#include "test_harness.hpp"

#include "models/black_hole/black_hole_gpu_module.hpp"
#include "models/gravity/gravity_gpu_module.hpp"
#include "models/light_trajectories/light_trajectories_module.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace
{

struct SnapshotBody
{
    glm::vec3 pos{0.0f};
    glm::vec3 vel{0.0f};
    float mass = 0.0f;
    float radius = 0.0f;
    bool initializing = false;
};

void StepCpuSnapshotOracle(std::vector<SnapshotBody> &bodies, float epsilonMeters)
{
    const std::size_t n = bodies.size();
    std::vector<glm::vec3> nextPos(n);
    std::vector<glm::vec3> nextVel(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (bodies[i].initializing)
        {
            nextPos[i] = bodies[i].pos;
            nextVel[i] = bodies[i].vel;
            continue;
        }

        glm::vec3 totalAcc(0.0f);
        std::uint32_t overlapCount = 0;

        for (std::size_t j = 0; j < n; ++j)
        {
            if (i == j || bodies[j].initializing)
            {
                continue;
            }

            const glm::vec3 disp = bodies[j].pos - bodies[i].pos;
            const float dist = glm::length(disp);
            if (dist <= 0.0f)
            {
                continue;
            }

            const glm::vec3 dir = disp / dist;
            const float distMeters = dist * 1000.0f;
            const float softenedSq = distMeters * distMeters + epsilonMeters * epsilonMeters;
            const double gForce =
                (kGravitationalConstant * static_cast<double>(bodies[i].mass) *
                 static_cast<double>(bodies[j].mass)) /
                static_cast<double>(softenedSq);
            const float accMag =
                static_cast<float>(gForce / static_cast<double>(bodies[i].mass));
            totalAcc += dir * accMag;

            if (bodies[j].radius + bodies[i].radius > dist)
            {
                ++overlapCount;
            }
        }

        glm::vec3 vel = bodies[i].vel + totalAcc / 96.0f;
        float collisionScale = 1.0f;
        for (std::uint32_t k = 0; k < overlapCount; ++k)
        {
            collisionScale *= -0.2f;
        }
        vel *= collisionScale;

        nextVel[i] = vel;
        nextPos[i] = bodies[i].pos + vel / 94.0f;
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        bodies[i].pos = nextPos[i];
        bodies[i].vel = nextVel[i];
    }
}

std::vector<glm::vec4> ReadDeformedGridBuffer()
{
    GRAV_ASSERT(gpuMemoryManager.Contains(BufferRole::DeformedGrid));
    const GpuBuffer &buf = gpuMemoryManager.Get(BufferRole::DeformedGrid);
    const std::size_t count = buf.LogicalCount();
    std::vector<glm::vec4> out(count);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    gpuMemoryManager.WaitForGpuCompletion();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf.Handle());
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       static_cast<GLsizeiptr>(count * sizeof(glm::vec4)), out.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return out;
}

void RunT01(const char *argv0)
{
    std::cout << "[RUN] T01: F1 Gravity GPU Default Preset Parity vs Snapshot Oracle...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 960;
    opts.height = 540;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.initialPresetIndex = 0;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT_NO_GL_ERROR("T01 Initialize");

    // Initial load has executed 0 simulation steps yet.
    GRAV_ASSERT(objs.size() == 202);
    std::vector<SnapshotBody> oracle;
    oracle.reserve(objs.size());
    for (const auto &o : objs)
    {
        SnapshotBody b{};
        b.pos = o.position;
        b.vel = o.velocity;
        b.mass = o.mass;
        b.radius = o.radius;
        b.initializing = o.initializing;
        oracle.push_back(b);
    }

    constexpr int kSteps = 12;
    for (int s = 0; s < kSteps; ++s)
    {
        StepCpuSnapshotOracle(oracle, 10.0f);
        app.StepFrame(1.0 / 60.0);
    }
    GRAV_ASSERT_NO_GL_ERROR("T01 StepFrames");

    const auto stats = app.SimManager().ActiveModule()->RuntimeStats();
    GRAV_ASSERT_MSG(stats.stepCount == static_cast<std::uint64_t>(kSteps),
                    "Expected " << kSteps << " GPU steps, got " << stats.stepCount);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    gpuMemoryManager.WaitForGpuCompletion();

    GRAV_ASSERT(gpuMemoryManager.Contains(BufferRole::CurrentState));
    const GpuBuffer &curBuf = gpuMemoryManager.Get(BufferRole::CurrentState);
    std::vector<GpuObjectState> gpuStates(objs.size());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, curBuf.Handle());
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       static_cast<GLsizeiptr>(gpuStates.size() * sizeof(GpuObjectState)),
                       gpuStates.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    float maxPosErr = 0.0f;
    float maxVelErr = 0.0f;
    for (std::size_t i = 0; i < oracle.size(); ++i)
    {
        const float posErr = glm::length(glm::vec3(gpuStates[i].position) - oracle[i].pos);
        const float velErr = glm::length(glm::vec3(gpuStates[i].velocity) - oracle[i].vel);
        maxPosErr = std::max(maxPosErr, posErr);
        maxVelErr = std::max(maxVelErr, velErr);
    }

    std::cout << "  T01 maxPosErr = " << maxPosErr << " world units, maxVelErr = " << maxVelErr
              << " world/step\n";
    GRAV_ASSERT_MSG(maxPosErr < 1.0e-2f, "Position drift exceeded tolerance: " << maxPosErr);
    GRAV_ASSERT_MSG(maxVelErr < 1.0e-2f, "Velocity drift exceeded tolerance: " << maxVelErr);

    app.Shutdown();
    std::cout << "[PASS] T01\n";
}

void RunT02(const char *argv0)
{
    std::cout << "[RUN] T02: SharedGridService 0/1/Many GridSources with Zero Object Buffers...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 800;
    opts.height = 600;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::Diagnostic_Triangle;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    // Verify all 8 N-body object SSBO roles are completely absent!
    for (BufferRole role : kGravityObjectBufferRoles)
    {
        GRAV_ASSERT_MSG(!gpuMemoryManager.Contains(role),
                        "Object role should not exist in non-F1 mode: " << BufferRoleName(role));
    }

    auto &grid = app.SharedGrid();
    const auto bhCfg = grav_render::GridConfig::BlackHoleWorldGm();
    grid.Configure(bhCfg);

    // Case 1: 0 sources -> flat grid at basePlaneOffsetWorld
    grid.SetCpuSources({}, 5001);
    GRAV_ASSERT(grid.ActiveSourceCount() == 0);
    GRAV_ASSERT(grid.UpdateIfDirty(true));
    const auto flatVerts = ReadDeformedGridBuffer();
    GRAV_ASSERT(flatVerts.size() == 201u * 201u);
    for (const auto &v : flatVerts)
    {
        GRAV_ASSERT_NEAR(v.y, bhCfg.basePlaneOffsetWorld, 1.0e-4);
    }

    // Case 2: 1 static Black Hole source at origin
    grav_models::BlackHoleParameters bh{};
    bh.SetMassKg(grav_models::kSagittariusAMassKg);
    const grav_render::GridSource bhSrc = bh.ToGridSource(bhCfg.metersPerWorldUnit);
    grid.SetCpuSources({bhSrc}, 5002);
    GRAV_ASSERT(grid.ActiveSourceCount() == 1);
    GRAV_ASSERT(grid.UpdateIfDirty(true));

    const auto oneVerts = ReadDeformedGridBuffer();
    const std::size_t centerIdx = 100u * 201u + 100u; // (0, basePlaneOffsetWorld, 0)
    const std::size_t cornerIdx = 0u;                 // (-150, basePlaneOffsetWorld, -150)

    const float centerY = oneVerts[centerIdx].y;
    const float cornerY = oneVerts[cornerIdx].y;
    std::cout << "  T02 1-BH centerY = " << centerY << ", cornerY = " << cornerY
              << " (base = " << bhCfg.basePlaneOffsetWorld << ")\n";
    GRAV_ASSERT(centerY < bhCfg.basePlaneOffsetWorld - 10.0f);
    GRAV_ASSERT(cornerY > centerY + 10.0f);

    // Analytical formula verification for 1 source & 4 sources
    auto evalAnalyticalY = [&](const glm::vec3 &basePos,
                               const std::vector<grav_render::GridSource> &sources)
    {
        float disp = 0.0f;
        for (const auto &s : sources)
        {
            const float rs = s.positionRs.w;
            if (rs <= 0.0f)
            {
                continue;
            }
            const float distWorld = glm::length(glm::vec3(s.positionRs) - basePos);
            const float minDistM = std::max(rs * (1.0f + bhCfg.clampDelta), 1.0e-6f);
            const float distM = std::max(distWorld * bhCfg.metersPerWorldUnit, minDistM);
            disp -= (bhCfg.deformationAmplitudeWorld * rs) / distM;
        }
        return basePos.y + disp;
    };

    GRAV_ASSERT_NEAR(centerY,
                     evalAnalyticalY(glm::vec3(0.0f, bhCfg.basePlaneOffsetWorld, 0.0f), {bhSrc}),
                     1.0e-2);

    // Case 3: Many (4) sources
    std::vector<grav_render::GridSource> manySources = {
        {glm::vec4(-60.0f, 0.0f, -60.0f, bhSrc.positionRs.w * 0.5f)},
        {glm::vec4(60.0f, 0.0f, -60.0f, bhSrc.positionRs.w * 0.7f)},
        {glm::vec4(-60.0f, 0.0f, 60.0f, bhSrc.positionRs.w * 0.9f)},
        {glm::vec4(60.0f, 0.0f, 60.0f, bhSrc.positionRs.w * 1.1f)},
    };
    grid.SetCpuSources(manySources, 5003);
    GRAV_ASSERT(grid.ActiveSourceCount() == 4);
    GRAV_ASSERT(grid.UpdateIfDirty(true));
    const auto manyVerts = ReadDeformedGridBuffer();

    float maxAnalyticalDiff = 0.0f;
    for (std::size_t idx = 0; idx < manyVerts.size(); idx += 17)
    {
        const glm::vec3 basePos(flatVerts[idx].x, bhCfg.basePlaneOffsetWorld, flatVerts[idx].z);
        const float expectedY = evalAnalyticalY(basePos, manySources);
        maxAnalyticalDiff = std::max(maxAnalyticalDiff, std::abs(manyVerts[idx].y - expectedY));
    }
    std::cout << "  T02 4-source maxAnalyticalDiff = " << maxAnalyticalDiff << "\n";
    GRAV_ASSERT(maxAnalyticalDiff < 1.0e-2f);
    GRAV_ASSERT_NO_GL_ERROR("T02 End");

    app.Shutdown();
    std::cout << "[PASS] T02\n";
}

void RunT11(const char *argv0)
{
    std::cout << "[RUN] T11: Paused Grid Visibility/Mass Dirty Deferral & F2<->F3 Shared BH...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 800;
    opts.height = 600;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F2_LightTrajectories;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    const std::size_t centerIdx = 100u * 201u + 100u;
    const float initialCenterY = ReadDeformedGridBuffer()[centerIdx].y;

    // Pause simulation and hide grid
    app.SetPaused(true);
    app.SimManager().SetGridVisible(false);
    app.StepFrame(1.0 / 60.0);

    const std::uint64_t dispatchesWhileHiddenStart = app.SharedGrid().ComputeDispatchCount();

    // Double Black Hole mass while paused and grid is hidden
    const double doubledMass = 2.0 * grav_models::kSagittariusAMassKg;
    app.SharedBlackHole().SetMassKg(doubledMass);
    app.StepFrame(1.0 / 60.0);
    app.StepFrame(1.0 / 60.0);

    // Grid must be marked dirty by F2's BH revision sync, but NOT dispatched while hidden!
    GRAV_ASSERT(app.SharedGrid().IsDirty());
    GRAV_ASSERT(app.SharedGrid().ComputeDispatchCount() == dispatchesWhileHiddenStart);

    // Re-enable grid visibility while still paused -> must dispatch grid.comp once and deepen funnel
    app.SimManager().SetGridVisible(true);
    app.StepFrame(1.0 / 60.0);

    const std::uint64_t dispatchesAfterShow = app.SharedGrid().ComputeDispatchCount();
    GRAV_ASSERT(dispatchesAfterShow == dispatchesWhileHiddenStart + 1);
    GRAV_ASSERT(!app.SharedGrid().IsDirty());

    const float heavierCenterY = ReadDeformedGridBuffer()[centerIdx].y;
    std::cout << "  T11 initialCenterY = " << initialCenterY
              << ", heavierCenterY = " << heavierCenterY << "\n";
    GRAV_ASSERT(heavierCenterY < initialCenterY - 5.0f);

    // Switch F2 -> F3 while paused: F3 must observe the exact same BH mass & rs AND reuse grid
    app.SimManager().RequestMode(grav_sim::ModeId::F3_BlackHoleGpu);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().ActiveMode() == grav_sim::ModeId::F3_BlackHoleGpu);

    auto *f3 = dynamic_cast<grav_models::BlackHoleGpuModule *>(app.SimManager().ActiveModule());
    GRAV_ASSERT(f3 != nullptr);
    const auto f3Scene = f3->BuildCurrentSceneConfig(160, 120);
    GRAV_ASSERT_NEAR(f3Scene.rsMeters, app.SharedBlackHole().schwarzschildRadiusMeters, 1.0);
    GRAV_ASSERT_MSG(app.SharedGrid().ComputeDispatchCount() == dispatchesAfterShow,
                    "Expected F2->F3 to reuse unchanged BH grid deformation without re-dispatching");

    GRAV_ASSERT_NO_GL_ERROR("T11 End");
    app.Shutdown();
    std::cout << "[PASS] T11\n";
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const std::string which = grav_test::ParseCaseArg(argc, argv);
        const char *argv0 = (argc > 0) ? argv[0] : nullptr;

        if (which == "ALL" || which == "T01")
        {
            RunT01(argv0);
        }
        if (which == "ALL" || which == "T02")
        {
            RunT02(argv0);
        }
        if (which == "ALL" || which == "T11")
        {
            RunT11(argv0);
        }
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}

