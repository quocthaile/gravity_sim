#include "test_harness.hpp"

#include "models/black_hole/black_hole_core.hpp"
#include "models/black_hole/black_hole_gpu_module.hpp"
#include "models/gravity_baseline/gravity_baseline_module.hpp"
#include "render/asset_paths.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace
{

void RunT12(const char *argv0)
{
    std::cout << "[RUN] T12: BenchmarkRunner Fixed Workload Lock, Export & Mutation Abort...\n";

    const std::filesystem::path tempBenchDir =
        std::filesystem::temp_directory_path() / "grav_sim_t12_bench";
    std::error_code ec;
    std::filesystem::remove_all(tempBenchDir, ec);

    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.initialPresetIndex = 1;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    auto &bench = app.Benchmark();
    bench.MutableConfig().warmupFrames = 3;
    bench.MutableConfig().measuredFrames = 8;
    bench.MutableConfig().outputDirectory = tempBenchDir.string();
    bench.MutableConfig().autoExportOnComplete = true;

    // 1. Clean fixed-workload run to Completion
    app.StartBenchmarkRun();
    GRAV_ASSERT(bench.IsActive());
    GRAV_ASSERT(bench.IsConfigLocked());

    for (int i = 0; i < 20 && bench.IsActive(); ++i)
    {
        app.StepFrame(1.0 / 60.0);
    }

    GRAV_ASSERT(bench.State() == grav_bench::BenchmarkState::Completed);
    GRAV_ASSERT(bench.Summary().sampleCount == 8);
    GRAV_ASSERT(std::filesystem::exists(bench.LastCsvPath()));
    GRAV_ASSERT(std::filesystem::exists(bench.LastManifestPath()));

    {
        std::ifstream jf(bench.LastManifestPath());
        std::ostringstream buf;
        buf << jf.rdbuf();
        const std::string manifest = buf.str();
        GRAV_ASSERT(manifest.find("\"status\": \"Completed\"") != std::string::npos);
        GRAV_ASSERT(manifest.find("a5f9f930efaa828afa39ff79af00ed7b6add4ee1") != std::string::npos);
        GRAV_ASSERT(manifest.find("dc263bb5a1b72f5a24190d2e38b84b5c8fe9a317") != std::string::npos);
    }

    // 2. Camera mutation mid-run -> must abort with explicit reason (never mix samples!)
    app.StartBenchmarkRun();
    app.StepFrame(1.0 / 60.0); // Preparing -> Warmup
    app.StepFrame(1.0 / 60.0);
    app.SimManager().ResetActiveCamera(); // Bumps cameraRevision!
    app.StepFrame(1.0 / 60.0);

    GRAV_ASSERT(bench.State() == grav_bench::BenchmarkState::Aborted);
    GRAV_ASSERT(!bench.AbortReason().empty());
    std::cout << "  T12 Camera mutation abort reason: " << bench.AbortReason() << "\n";

    // 3. Mode switch hotkey mid-run -> must abort immediately before switching
    app.StartBenchmarkRun();
    app.StepFrame(1.0 / 60.0);
    app.HandleKeyAction(GLFW_KEY_F2, 0, GLFW_PRESS, 0, true);
    GRAV_ASSERT(bench.State() == grav_bench::BenchmarkState::Aborted);
    std::cout << "  T12 Mode switch abort reason: " << bench.AbortReason() << "\n";

    std::filesystem::remove_all(tempBenchDir, ec);
    app.Shutdown();
    std::cout << "[PASS] T12\n";
}

void RunT13(const char *argv0)
{
    std::cout << "[RUN] T13: Canonical RK4 Order-4 Convergence, Invariants & CPU/GPU Parity...\n";

    const double rs = grav_models::ComputeSchwarzschildRadiusMeters(grav_models::kSagittariusAMassKg);

    // 1. Step-halving Richardson convergence order (~4.0) for a strongly lensed photon (b = 3.2 rs)
    auto integrateFixedHorizonUnits = [&](int steps, double totalAffineUnits)
    {
        grav_models::PlanarRayState ray = grav_models::InitializePlanarRay(
            glm::dvec2(-10.0 * rs, 3.2 * rs), glm::dvec2(grav_models::kSpeedOfLightMps, 0.0), rs);
        // StepPlanarRay uses baseStep = 0.02 * clamp(dLambda, 0.05, 10.0) in dimensionless h,
        // so dLambda = (totalAffineUnits / steps) / 0.02 gives exact step h = totalAffineUnits / steps.
        const double dLambdaArg = (totalAffineUnits / static_cast<double>(steps)) / 0.02;
        for (int i = 0; i < steps; ++i)
        {
            grav_models::StepPlanarRay(ray, dLambdaArg, rs, 1.0e15,
                                       grav_models::PlanarGeodesicSolverKind::CanonicalDimensionlessRk4);
        }
        return ray;
    };

    const double totalH = 12.0; // travels ~12 rs near photon sphere
    const auto rayN1 = integrateFixedHorizonUnits(120, totalH); // h = 0.10
    const auto rayN2 = integrateFixedHorizonUnits(240, totalH); // h = 0.05
    const auto rayN4 = integrateFixedHorizonUnits(480, totalH); // h = 0.025

    const double err12 = glm::length(rayN1.uPos - rayN2.uPos);
    const double err24 = glm::length(rayN2.uPos - rayN4.uPos);
    GRAV_ASSERT(err24 > 1.0e-15);
    const double empiricalOrder = std::log2(err12 / err24);
    std::cout << "  T13 RK4 step-halving err(h,h/2)=" << err12 << ", err(h/2,h/4)=" << err24
              << " -> empirical order p = " << empiricalOrder << "\n";
    GRAV_ASSERT_MSG(empiricalOrder >= 3.7 && empiricalOrder <= 4.3,
                    "Expected ~4th-order RK4 convergence, got p = " << empiricalOrder);

    // 2. Physical Null & Angular Momentum conservation from true phase-space coordinates
    const double nullErr = grav_models::EvaluatePlanarNullRelativeError(
        rayN4, rs, grav_models::PlanarGeodesicSolverKind::CanonicalDimensionlessRk4);
    const double angMomErr = grav_models::EvaluatePlanarAngularMomentumRelativeError(
        rayN4, rs, grav_models::PlanarGeodesicSolverKind::CanonicalDimensionlessRk4);
    std::cout << "  T13 Planar invariant drift: nullRelErr=" << nullErr
              << ", angMomRelErr=" << angMomErr << "\n";
    GRAV_ASSERT(nullErr < 1.0e-8);
    GRAV_ASSERT(angMomErr < 1.0e-8);

    // 3. 2D Planar vs 3D Equatorial parity on identical impact parameters
    grav_models::BlackHoleSceneConfig eqScene{};
    eqScene.bhCenterMeters = glm::dvec3(0.0);
    eqScene.rsMeters = rs;
    eqScene.diskEnabled = false;
    eqScene.dLambdaMeters = 0.02 * rs;
    eqScene.escapeRadiusMeters = 30.0 * rs;
    eqScene.maxSteps = 4000;

    for (double bMult : {1.8, 3.4, 5.5})
    {
        // 3D ray in equatorial plane (z = 0, y = 0 -> launched at (-12 rs, 0, b*rs) along +x)
        const auto res3d = grav_models::TraceBlackHoleRayCPU(
            glm::dvec3(-12.0 * rs, 0.0, bMult * rs), glm::dvec3(1.0, 0.0, 0.0), eqScene,
            grav_models::Geodesic3DSolverKind::CanonicalDimensionlessRk4);

        grav_models::PlanarRayState ray2d = grav_models::InitializePlanarRay(
            glm::dvec2(-12.0 * rs, bMult * rs), glm::dvec2(grav_models::kSpeedOfLightMps, 0.0), rs);
        for (int s = 0; s < eqScene.maxSteps && !ray2d.captured && !ray2d.escaped; ++s)
        {
            grav_models::StepPlanarRay(ray2d, 1.0, rs, eqScene.escapeRadiusMeters,
                                       grav_models::PlanarGeodesicSolverKind::CanonicalDimensionlessRk4);
        }

        const bool captured3d = (res3d.outcome == grav_models::GeodesicOutcome::HitHorizon);
        GRAV_ASSERT_MSG(captured3d == ray2d.captured,
                        "2D vs 3D capture mismatch at b/rs = " << bMult);
        if (!captured3d)
        {
            GRAV_ASSERT(res3d.nullRelativeError < 1.0e-7);
            GRAV_ASSERT(res3d.angularMomentumRelativeError < 1.0e-7);
            const glm::dvec2 dir2d = glm::normalize(ray2d.uVel);
            const glm::dvec2 dir3d =
                glm::normalize(glm::dvec2(res3d.finalDirection.x, res3d.finalDirection.z));
            const double angleCos = glm::dot(dir2d, dir3d);
            GRAV_ASSERT_MSG(angleCos > 0.999999,
                            "2D vs 3D deflection direction mismatch at b/rs=" << bMult
                                                                              << " dot=" << angleCos);
        }
    }

    // 4. CPU vs GPU (F3 geodesic_rk4.comp) fixed-ray image parity
    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F3_BlackHoleGpu;
    opts.initialPresetIndex = 0; // Sagittarius A* + Accretion Disk + Background Stars (rk4_canonical_v1)
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    auto *f3 = dynamic_cast<grav_models::BlackHoleGpuModule *>(app.SimManager().ActiveModule());
    GRAV_ASSERT(f3 != nullptr);
    GRAV_ASSERT(f3->ActiveSolver().id == "rk4_canonical_v1");

    constexpr int kTestW = 64;
    constexpr int kTestH = 48;
    f3->SetComputeResolution(kTestW, kTestH);
    app.StepFrame(1.0 / 60.0); // Renders static full-res 64x48 image on GPU

    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    gpuMemoryManager.WaitForGpuCompletion();

    GRAV_ASSERT(f3->ActiveComputeWidth() == kTestW && f3->ActiveComputeHeight() == kTestH);
    std::vector<std::uint8_t> gpuPixels(static_cast<std::size_t>(kTestW * kTestH * 4), 0);
    glBindTexture(GL_TEXTURE_2D, f3->OutputTextureHandle());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, gpuPixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    const auto sceneCfg = f3->BuildCurrentSceneConfig(kTestW, kTestH);
    int matchingPixels = 0;
    int horizonHits = 0;
    int diskOrObjectHits = 0;
    const int totalPixels = kTestW * kTestH;

    for (int py = 0; py < kTestH; ++py)
    {
        for (int px = 0; px < kTestW; ++px)
        {
            const glm::dvec3 dir =
                grav_models::PixelToCameraRayDirection(px, py, kTestW, kTestH, sceneCfg);
            const auto cpuRes = grav_models::TraceBlackHoleRayCPU(
                sceneCfg.camPosMeters, dir, sceneCfg,
                grav_models::Geodesic3DSolverKind::CanonicalDimensionlessRk4);

            if (cpuRes.outcome == grav_models::GeodesicOutcome::HitHorizon)
            {
                ++horizonHits;
            }
            else if (cpuRes.outcome == grav_models::GeodesicOutcome::HitDisk ||
                     cpuRes.outcome == grav_models::GeodesicOutcome::HitObject)
            {
                ++diskOrObjectHits;
            }

            const std::size_t base =
                (static_cast<std::size_t>(py) * static_cast<std::size_t>(kTestW) +
                 static_cast<std::size_t>(px)) *
                4u;
            const glm::vec3 gpuRgb(static_cast<float>(gpuPixels[base + 0]) / 255.0f,
                                   static_cast<float>(gpuPixels[base + 1]) / 255.0f,
                                   static_cast<float>(gpuPixels[base + 2]) / 255.0f);
            const glm::vec3 cpuRgb(cpuRes.color);
            if (glm::length(gpuRgb - cpuRgb) < 0.12f)
            {
                ++matchingPixels;
            }
        }
    }

    const double agreementRatio =
        static_cast<double>(matchingPixels) / static_cast<double>(totalPixels);
    std::cout << "  T13 CPU vs GPU 64x48 parity: agreement = " << (agreementRatio * 100.0)
              << "% (horizonPx=" << horizonHits << ", disk/objPx=" << diskOrObjectHits << ")\n";

    GRAV_ASSERT(horizonHits > 20);
    GRAV_ASSERT(diskOrObjectHits > 20);
    // Single-precision GPU vs double-precision CPU only differs on grazing photon-ring boundary pixels (<3%)
    GRAV_ASSERT_MSG(agreementRatio >= 0.96,
                    "CPU/GPU pixel agreement below 96%: " << (agreementRatio * 100.0) << "%");

    GRAV_ASSERT_NO_GL_ERROR("T13 End");
    app.Shutdown();
    std::cout << "[PASS] T13\n";
}

void RunT15(const char *argv0)
{
    std::cout << "[RUN] T15: External CWD Asset Resolution & Shader/CSV Loading...\n";

    const std::filesystem::path originalCwd = std::filesystem::current_path();
    const std::filesystem::path exeAbsPath =
        (argv0 != nullptr && argv0[0] != '\0') ? std::filesystem::absolute(argv0) : originalCwd;

    const std::filesystem::path tempOutsideDir =
        std::filesystem::temp_directory_path() / "grav_sim_t15_clean_cwd";
    std::error_code ec;
    std::filesystem::create_directories(tempOutsideDir, ec);

    struct CwdRestoreGuard
    {
        std::filesystem::path saved;
        ~CwdRestoreGuard()
        {
            std::error_code e;
            std::filesystem::current_path(saved, e);
        }
    } guard{originalCwd};

    // Move CWD completely outside the repository and build tree!
    std::filesystem::current_path(tempOutsideDir);
    const std::string exeStr = exeAbsPath.string();

    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.initialPresetIndex = 3; // CSV loader preset ("shaders/presets/sample_four_body.csv")
    opts.argv0 = exeStr.c_str();

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().State() == grav_app::LifecycleState::Running);
    GRAV_ASSERT(app.SimManager().ActiveModule()->RuntimeStats().primaryCount == 4);

    // Also load F2 and F3 shaders from external CWD
    app.SimManager().RequestMode(grav_sim::ModeId::F2_LightTrajectories);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().State() == grav_app::LifecycleState::Running);

    app.SimManager().RequestMode(grav_sim::ModeId::F3_BlackHoleGpu);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().State() == grav_app::LifecycleState::Running);

    GRAV_ASSERT_NO_GL_ERROR("T15 End");
    app.Shutdown();
    std::filesystem::current_path(originalCwd, ec);
    std::filesystem::remove_all(tempOutsideDir, ec);
    std::cout << "[PASS] T15\n";
}

// Verbatim reference transcription of src/gravity_sim_3Dgrid_baseline.cpp:423-470 for T16
void ReferenceHistoricalBaselineStep(std::vector<Object> &refObjs, double refDt)
{
    const double G = 6.6743e-11;
    for (auto &obj : refObjs)
    {
        for (auto &obj2 : refObjs)
        {
            if (&obj2 != &obj && !obj.initializing && !obj2.initializing)
            {
                float dx = obj2.GetPosition()[0] - obj.GetPosition()[0];
                float dy = obj2.GetPosition()[1] - obj.GetPosition()[1];
                float dz = obj2.GetPosition()[2] - obj.GetPosition()[2];
                float distance = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (distance > 0.0f)
                {
                    float dir0 = dx / distance;
                    float dir1 = dy / distance;
                    float dir2 = dz / distance;
                    distance *= 1000.0f;
                    double Gforce = (G * obj.mass * obj2.mass) / (distance * distance);
                    float acc1 = static_cast<float>(Gforce / obj.mass);
                    obj.Accelerate(dir0 * acc1, dir1 * acc1, dir2 * acc1, refDt);
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
            obj.UpdatePosition(refDt);
        }
    }
}

void RunT16(const char *argv0)
{
    std::cout << "[RUN] T16: Historical CPU Baseline Reference vs F6 Adapter Physics Parity...\n";

    // Part A: Direct 120-step deterministic parity including collision & initializing object
    std::vector<Object> refBodies = CreateObjects({}, 10);
    // Add an overlapping pair to exercise CheckCollision (-0.2f multiplier) and an initializing body
    refBodies.emplace_back(glm::vec3(15.0f, 0.0f, 0.0f), glm::vec3(-5.0f, 0.0f, 0.0f),
                           5.0e23f, 3344.0f);
    refBodies.emplace_back(glm::vec3(800.0f, 200.0f, 0.0f), glm::vec3(0.0f), 1.0e21f, 3344.0f);
    refBodies.back().initializing = true;

    std::vector<Object> adapterBodies = refBodies;

    for (int step = 0; step < 120; ++step)
    {
        ReferenceHistoricalBaselineStep(refBodies, 1.0 / 96.0);
        grav_models::StepBaselineBodiesInPlace(adapterBodies, 1.0 / 96.0);
    }

    float maxPosDiff = 0.0f;
    float maxVelDiff = 0.0f;
    for (std::size_t i = 0; i < refBodies.size(); ++i)
    {
        maxPosDiff =
            std::max(maxPosDiff, glm::length(refBodies[i].position - adapterBodies[i].position));
        maxVelDiff =
            std::max(maxVelDiff, glm::length(refBodies[i].velocity - adapterBodies[i].velocity));
    }
    std::cout << "  T16 120-step F6 vs Baseline Reference maxPosDiff=" << maxPosDiff
              << ", maxVelDiff=" << maxVelDiff << "\n";
    GRAV_ASSERT(maxPosDiff < 1.0e-5f);
    GRAV_ASSERT(maxVelDiff < 1.0e-5f);

    // Part B: Verify F6 module inside Application matches ReferenceHistoricalBaselineStep
    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F6_GravityBaselineCpu;
    opts.initialPresetIndex = 1; // Two-body Earth-Moon
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));

    std::vector<Object> twoBodyRef = CreateObjects({}, 0);
    for (int s = 0; s < 15; ++s)
    {
        ReferenceHistoricalBaselineStep(twoBodyRef, 1.0 / 96.0);
        app.StepFrame(1.0 / 60.0);
    }

    const auto *f6 =
        dynamic_cast<const grav_models::GravityBaselineModule *>(app.SimManager().ActiveModule());
    GRAV_ASSERT(f6 != nullptr);
    GRAV_ASSERT(f6->Bodies().size() == twoBodyRef.size());
    for (std::size_t i = 0; i < twoBodyRef.size(); ++i)
    {
        GRAV_ASSERT(glm::length(f6->Bodies()[i].position - twoBodyRef[i].position) < 1.0e-5f);
        GRAV_ASSERT(glm::length(f6->Bodies()[i].velocity - twoBodyRef[i].velocity) < 1.0e-5f);
    }

    GRAV_ASSERT_NO_GL_ERROR("T16 End");
    app.Shutdown();
    std::cout << "[PASS] T16\n";
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const std::string which = grav_test::ParseCaseArg(argc, argv);
        const char *argv0 = (argc > 0) ? argv[0] : nullptr;

        if (which == "ALL" || which == "T12")
        {
            RunT12(argv0);
        }
        if (which == "ALL" || which == "T13")
        {
            RunT13(argv0);
        }
        if (which == "ALL" || which == "T15")
        {
            RunT15(argv0);
        }
        if (which == "ALL" || which == "T16")
        {
            RunT16(argv0);
        }
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}

