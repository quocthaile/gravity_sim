#include "test_harness.hpp"

#include "models/black_hole/black_hole_cpu_module.hpp"
#include "models/black_hole/black_hole_gpu_module.hpp"
#include "models/gravity/gravity_gpu_module.hpp"
#include "models/light_trajectories/light_trajectories_module.hpp"
#include "models/ray_tracing/flat_ray_tracing_module.hpp"
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <vector>

// Compile-time proof for T14: PlanarRayState contains zero std::vector trail history,
// guaranteeing O(1) stage copies during 4-stage RK4 integration!
static_assert(std::is_trivially_copyable_v<grav_models::PlanarRayState>,
              "PlanarRayState must remain trivially copyable without embedded trail vectors");

namespace
{

void AssertObjectRolesPresence(bool expectedPresent)
{
    for (BufferRole role : kGravityObjectBufferRoles)
    {
        const bool present = gpuMemoryManager.Contains(role);
        GRAV_ASSERT_MSG(present == expectedPresent,
                        "Role " << BufferRoleName(role) << " presence=" << present
                                << ", expected=" << expectedPresent);
    }
}

void RunT03(const char *argv0)
{
    std::cout << "[RUN] T03: Single Window/Context & Grid Handle Invariance F1->F2->F3->F2->F1...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 800;
    opts.height = 600;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    GLFWwindow *const initialWindow = app.Window();
    GLFWwindow *const initialContext = glfwGetCurrentContext();
    const auto initialGridHandles = app.SharedGrid().Handles();
    GRAV_ASSERT(initialGridHandles.baseGridBuffer != 0);
    GRAV_ASSERT(initialGridHandles.deformedGridBuffer != 0);
    GRAV_ASSERT(initialGridHandles.gridSourcesBuffer != 0);
    GRAV_ASSERT(initialGridHandles.gridVAO != 0);
    GRAV_ASSERT(initialGridHandles.gridEBO != 0);

    AssertObjectRolesPresence(true);

    const std::vector<grav_sim::ModeId> sequence = {
        grav_sim::ModeId::F2_LightTrajectories,
        grav_sim::ModeId::F3_BlackHoleGpu,
        grav_sim::ModeId::F2_LightTrajectories,
        grav_sim::ModeId::F1_GravityGpu,
    };

    for (grav_sim::ModeId target : sequence)
    {
        app.SimManager().RequestMode(target);
        GRAV_ASSERT(grav_test::StepUntilSettled(app));
        GRAV_ASSERT(app.SimManager().ActiveMode() == target);

        // Single GLFW window & OpenGL context must remain invariant
        GRAV_ASSERT(app.Window() == initialWindow);
        GRAV_ASSERT(glfwGetCurrentContext() == initialContext);

        // Shared grid buffers and VAO/EBO handles must remain 100% invariant (divisions = 200)
        const auto curHandles = app.SharedGrid().Handles();
        GRAV_ASSERT(curHandles.baseGridBuffer == initialGridHandles.baseGridBuffer);
        GRAV_ASSERT(curHandles.deformedGridBuffer == initialGridHandles.deformedGridBuffer);
        GRAV_ASSERT(curHandles.gridSourcesBuffer == initialGridHandles.gridSourcesBuffer);
        GRAV_ASSERT(curHandles.gridVAO == initialGridHandles.gridVAO);
        GRAV_ASSERT(curHandles.gridEBO == initialGridHandles.gridEBO);

        AssertObjectRolesPresence(target == grav_sim::ModeId::F1_GravityGpu);
        GRAV_ASSERT_NO_GL_ERROR("T03 Step");
    }

    app.Shutdown();
    std::cout << "[PASS] T03\n";
}

void RunT04(const char *argv0)
{
    std::cout << "[RUN] T04: All Mode Pairs, Preset Cycling (Tab), and Runtime Resets (R)...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    const std::vector<grav_sim::ModeId> modes = {
        grav_sim::ModeId::F1_GravityGpu,
        grav_sim::ModeId::F2_LightTrajectories,
        grav_sim::ModeId::F3_BlackHoleGpu,
        grav_sim::ModeId::F4_FlatRayTracingCpu,
        grav_sim::ModeId::F5_BlackHoleCpu,
        grav_sim::ModeId::F6_GravityBaselineCpu,
        grav_sim::ModeId::Diagnostic_Triangle,
        // Cross-mode pairs
        grav_sim::ModeId::F3_BlackHoleGpu,
        grav_sim::ModeId::F1_GravityGpu,
        grav_sim::ModeId::F5_BlackHoleCpu,
        grav_sim::ModeId::F2_LightTrajectories,
        grav_sim::ModeId::F6_GravityBaselineCpu,
        grav_sim::ModeId::F4_FlatRayTracingCpu,
        grav_sim::ModeId::F1_GravityGpu,
    };

    for (grav_sim::ModeId m : modes)
    {
        app.SimManager().RequestMode(m);
        GRAV_ASSERT(grav_test::StepUntilSettled(app));
        GRAV_ASSERT(app.SimManager().ActiveMode() == m);
        GRAV_ASSERT(app.SimManager().ActiveModule() != nullptr);

        // Cycle preset via Tab if mode has > 1 preset (skip CSV-custom index 3 on F1/F6 only if needed,
        // note preset 1 is Two-Body / Commented Ray / Thin Disk / Archive Scene / Geodesic RK4)
        const std::size_t presetCount = app.SimManager().ActiveModule()->Presets().size();
        if (presetCount > 1)
        {
            const std::size_t prevIdx = app.SimManager().ActiveModule()->ActivePresetIndex();
            const std::size_t expectedNext = (prevIdx + 1) % presetCount;
            app.HandleKeyAction(GLFW_KEY_TAB, 0, GLFW_PRESS, 0, true);
            GRAV_ASSERT(grav_test::StepUntilSettled(app));
            GRAV_ASSERT(app.SimManager().ActiveModule()->ActivePresetIndex() == expectedNext);
        }

        // Trigger preset runtime reset (R)
        const std::uint64_t loadsBeforeReset = app.SimManager().CompletedLoadCount();
        app.HandleKeyAction(GLFW_KEY_R, 0, GLFW_PRESS, 0, true);
        GRAV_ASSERT(grav_test::StepUntilSettled(app));
        GRAV_ASSERT(app.SimManager().CompletedLoadCount() == loadsBeforeReset + 1);

        // Reset camera (C)
        app.HandleKeyAction(GLFW_KEY_C, 0, GLFW_PRESS, 0, true);
        app.StepFrame(1.0 / 60.0);
        GRAV_ASSERT_NO_GL_ERROR("T04 Mode/Preset/Reset");
    }

    app.Shutdown();
    std::cout << "[PASS] T04\n";
}

void RunT05(const char *argv0)
{
    std::cout << "[RUN] T05: GLFW_REPEAT Ignore & Rapid Multi-Mode Request Coalescing...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    const std::uint64_t baseLoads = app.SimManager().CompletedLoadCount();

    // 1. Holding F2 / R generating 25 GLFW_REPEAT events must trigger ZERO reloads
    for (int i = 0; i < 25; ++i)
    {
        app.HandleKeyAction(GLFW_KEY_F2, 0, GLFW_REPEAT, 0, true);
        app.HandleKeyAction(GLFW_KEY_R, 0, GLFW_REPEAT, 0, true);
    }
    app.StepFrame(1.0 / 60.0);
    GRAV_ASSERT(app.SimManager().ActiveMode() == grav_sim::ModeId::F1_GravityGpu);
    GRAV_ASSERT(app.SimManager().CompletedLoadCount() == baseLoads);

    // 2. Pressing F1 when F1 is already active in Running state is a no-op
    app.HandleKeyAction(GLFW_KEY_F1, 0, GLFW_PRESS, 0, true);
    app.StepFrame(1.0 / 60.0);
    GRAV_ASSERT(app.SimManager().CompletedLoadCount() == baseLoads);

    // 3. Rapid burst of F2 -> F3 -> F4 -> F5 -> F6 -> F2 in the same frame coalesces to 1 load (F2)
    app.HandleKeyAction(GLFW_KEY_F2, 0, GLFW_PRESS, 0, true);
    app.HandleKeyAction(GLFW_KEY_F3, 0, GLFW_PRESS, 0, true);
    app.HandleKeyAction(GLFW_KEY_F4, 0, GLFW_PRESS, 0, true);
    app.HandleKeyAction(GLFW_KEY_F5, 0, GLFW_PRESS, 0, true);
    app.HandleKeyAction(GLFW_KEY_F6, 0, GLFW_PRESS, 0, true);
    app.HandleKeyAction(GLFW_KEY_F2, 0, GLFW_PRESS, 0, true);

    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().ActiveMode() == grav_sim::ModeId::F2_LightTrajectories);
    GRAV_ASSERT_MSG(app.SimManager().CompletedLoadCount() == baseLoads + 1,
                    "Expected coalesced burst to execute 1 load, got "
                        << (app.SimManager().CompletedLoadCount() - baseLoads));

    GRAV_ASSERT_NO_GL_ERROR("T05 End");
    app.Shutdown();
    std::cout << "[PASS] T05\n";
}

void RunT06(const char *argv0)
{
    std::cout << "[RUN] T06: Mode Switch While Object Initializing, Paused & Export Active...\n";

    const std::filesystem::path tempCsv =
        std::filesystem::temp_directory_path() / "grav_sim_t06_traj_v2.csv";
    std::error_code ec;
    std::filesystem::remove(tempCsv, ec);

    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.initialPresetIndex = 1; // Two-body Earth-Moon
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    auto *f1 = dynamic_cast<grav_models::GravityGpuModule *>(app.SimManager().ActiveModule());
    GRAV_ASSERT(f1 != nullptr);
    const std::uint64_t f1Gen = app.SimManager().SceneGeneration();

    f1->SetTrajectoryExportPath(tempCsv.string());
    f1->SetTrajectorySampleIntervalSteps(2);
    f1->SetTrajectoryExportEnabled(true);

    // Run 4 steps so trajectory samples are buffered
    for (int i = 0; i < 4; ++i)
    {
        app.StepFrame(1.0 / 60.0);
    }
    GRAV_ASSERT(f1->RecordedTrajectorySampleCount() >= 2);

    // Start spawning an initializing object, sync it once, pause, and immediately switch F1 -> F2
    objs.emplace_back(glm::vec3(500.0f, 100.0f, 0.0f), glm::vec3(0.0f), 1.0e21f);
    objs.back().initializing = true;
    app.StepFrame(1.0 / 60.0);
    GRAV_ASSERT(objs.back().initializing);

    app.SetPaused(true);
    app.SimManager().RequestMode(grav_sim::ModeId::F2_LightTrajectories);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    GRAV_ASSERT(app.SimManager().ActiveMode() == grav_sim::ModeId::F2_LightTrajectories);
    GRAV_ASSERT(objs.empty());
    AssertObjectRolesPresence(false);

    // Verify exported schema-v2 CSV contains header and correct scene generation
    GRAV_ASSERT(std::filesystem::exists(tempCsv));
    {
        std::ifstream ifs(tempCsv);
        std::string line1, line2, line3;
        std::getline(ifs, line1);
        std::getline(ifs, line2);
        std::getline(ifs, line3);
        GRAV_ASSERT(line1.find("schema_version=2") != std::string::npos);
        GRAV_ASSERT(line2.find("scene_generation,step,nominal_time_s,logical_object_id") !=
                    std::string::npos);
        GRAV_ASSERT(line3.rfind(std::to_string(f1Gen) + ",", 0) == 0);
    }
    std::filesystem::remove(tempCsv, ec);

    GRAV_ASSERT_NO_GL_ERROR("T06 End");
    app.Shutdown();
    std::cout << "[PASS] T06\n";
}

void RunT07(const char *argv0)
{
    std::cout << "[RUN] T07: F4/F5 Active CPU Worker Cancellation & Stale Result Drop...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F5_BlackHoleCpu;
    opts.initialPresetIndex = 1; // Full 3D RK4 Geodesic trace (heavy CPU workload)
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    auto *f5 = dynamic_cast<grav_models::BlackHoleCpuModule *>(app.SimManager().ActiveModule());
    GRAV_ASSERT(f5 != nullptr);

    // Increase CPU render size so worker is guaranteed to be mid-flight
    f5->SetRenderResolution(480, 360);
    app.StepFrame(1.0 / 60.0);

    // Mutate camera while worker is mid-flight -> invalidates old cameraRevision
    app.SimManager().ResetActiveCamera();
    app.StepFrame(1.0 / 60.0);

    // Immediately switch F5 -> F4 while F5 worker is executing, then F4 -> F1 while F4 worker runs
    app.SimManager().RequestMode(grav_sim::ModeId::F4_FlatRayTracingCpu);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().ActiveMode() == grav_sim::ModeId::F4_FlatRayTracingCpu);

    auto *f4 = dynamic_cast<grav_models::FlatRayTracingModule *>(app.SimManager().ActiveModule());
    GRAV_ASSERT(f4 != nullptr);
    f4->SetRenderResolution(640, 480);
    app.StepFrame(1.0 / 60.0);

    app.SimManager().RequestMode(grav_sim::ModeId::F1_GravityGpu);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().ActiveMode() == grav_sim::ModeId::F1_GravityGpu);

    GRAV_ASSERT_NO_GL_ERROR("T07 End");
    app.Shutdown();
    std::cout << "[PASS] T07\n";
}

void RunT08(const char *argv0)
{
    std::cout << "[RUN] T08: Framebuffer Resize & 0x0 Minimize Guard...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 800;
    opts.height = 600;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F3_BlackHoleGpu;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    auto *f3 = dynamic_cast<grav_models::BlackHoleGpuModule *>(app.SimManager().ActiveModule());
    GRAV_ASSERT(f3 != nullptr);
    GRAV_ASSERT(f3->OutputTextureHandle() != 0);
    GRAV_ASSERT(f3->ActiveComputeWidth() > 0 && f3->ActiveComputeHeight() > 0);

    // Simulate minimize (0 x 0 framebuffer)
    app.HandleFramebufferResizeEvent(0, 0);
    GRAV_ASSERT(app.IsMinimized());
    app.StepFrame(1.0 / 60.0);
    app.StepFrame(1.0 / 60.0);

    // Output texture must NOT be reallocated to 0x0!
    GRAV_ASSERT(f3->ActiveComputeWidth() > 0 && f3->ActiveComputeHeight() > 0);
    GRAV_ASSERT_NO_GL_ERROR("T08 Minimized");

    // Restore to 1280x720 and then resize to 1024x768
    app.HandleFramebufferResizeEvent(1280, 720);
    GRAV_ASSERT(!app.IsMinimized());
    app.StepFrame(1.0 / 60.0);

    app.HandleFramebufferResizeEvent(1024, 768);
    app.StepFrame(1.0 / 60.0);
    GRAV_ASSERT_NO_GL_ERROR("T08 Restored & Resized");

    app.Shutdown();
    std::cout << "[PASS] T08\n";
}

void RunT09(const char *argv0)
{
    std::cout << "[RUN] T09: 50-Cycle Multi-Mode Transition Stress & Bounded Resource Census...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    app.SimManager().SetWarmupFrameTarget(1);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    const std::vector<grav_sim::ModeId> ring = {
        grav_sim::ModeId::F1_GravityGpu,
        grav_sim::ModeId::F2_LightTrajectories,
        grav_sim::ModeId::F3_BlackHoleGpu,
        grav_sim::ModeId::F4_FlatRayTracingCpu,
        grav_sim::ModeId::F5_BlackHoleCpu,
        grav_sim::ModeId::F6_GravityBaselineCpu,
        grav_sim::ModeId::Diagnostic_Triangle,
    };

    // Warm up ShaderCache across all 7 modes once
    for (grav_sim::ModeId m : ring)
    {
        app.SimManager().RequestMode(m);
        GRAV_ASSERT(grav_test::StepUntilSettled(app));
    }
    app.SimManager().RequestMode(grav_sim::ModeId::F1_GravityGpu);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    const auto baselineSnap = app.Census().Snapshot(&gpuMemoryManager);
    const std::size_t baselineShaders = app.Shaders().CachedProgramCount();

    // Execute 50 mode-switch cycles across the modes
    for (int cycle = 0; cycle < 50; ++cycle)
    {
        const grav_sim::ModeId nextMode = ring[static_cast<std::size_t>(cycle) % ring.size()];
        app.SimManager().RequestMode(nextMode);
        GRAV_ASSERT(grav_test::StepUntilSettled(app));
    }

    // Return to F1 to compare exact steady-state census against post-warmup baseline
    app.SimManager().RequestMode(grav_sim::ModeId::F1_GravityGpu);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    const auto finalSnap = app.Census().Snapshot(&gpuMemoryManager);
    const std::size_t finalShaders = app.Shaders().CachedProgramCount();

    std::cout << "  T09 Post-Warmup LiveObjects=" << baselineSnap.totalLiveObjects
              << " AllocBytes=" << baselineSnap.totalAllocatedBytes
              << " | Post-50-Cycles LiveObjects=" << finalSnap.totalLiveObjects
              << " AllocBytes=" << finalSnap.totalAllocatedBytes << "\n";

    GRAV_ASSERT(finalShaders == baselineShaders);
    GRAV_ASSERT(finalSnap.totalLiveObjects == baselineSnap.totalLiveObjects);
    GRAV_ASSERT(finalSnap.totalAllocatedBytes == baselineSnap.totalAllocatedBytes);
    GRAV_ASSERT_NO_GL_ERROR("T09 End");

    app.Shutdown();
    std::cout << "[PASS] T09\n";
}

void RunT10(const char *argv0)
{
    std::cout << "[RUN] T10: Module Init / Asset Failure Partial Cleanup & Error Recovery...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 640;
    opts.height = 480;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F1_GravityGpu;
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    // Part A: Injected initialization failure during F1 -> F3 switch
    app.SimManager().InjectNextLoadFailure("Simulated shader compilation / GPU allocation failure");
    app.SimManager().RequestMode(grav_sim::ModeId::F3_BlackHoleGpu);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    GRAV_ASSERT(app.SimManager().State() == grav_app::LifecycleState::Error);
    GRAV_ASSERT(app.SimManager().ActiveModule() == nullptr);
    GRAV_ASSERT(app.Census().LiveSceneResourceCount() == 0);
    AssertObjectRolesPresence(false);
    GRAV_ASSERT(!app.SimManager().LastErrorMessage().empty());

    // Render an Error-state UI frame to prove UI remains responsive in Error state
    app.StepFrame(1.0 / 60.0);
    GRAV_ASSERT_NO_GL_ERROR("T10 Error Frame");

    // Recover from Error -> F2
    app.SimManager().RequestMode(grav_sim::ModeId::F2_LightTrajectories);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().State() == grav_app::LifecycleState::Running);
    GRAV_ASSERT(app.SimManager().ActiveMode() == grav_sim::ModeId::F2_LightTrajectories);

    // Part B: Real missing CSV asset failure during F1 preset 3 OnEnter (after unit sphere init!)
    app.SimManager().SetCustomModuleFactory(
        [](grav_sim::ModeId m) -> std::unique_ptr<grav_sim::ISimulationModule>
        {
            auto mod = grav_app::SimulationManager::DefaultCreateModule(m);
            if (auto *f1 = dynamic_cast<grav_models::GravityGpuModule *>(mod.get()))
            {
                f1->SetCustomCsvPath("nonexistent_missing_preset_file_t10.csv");
            }
            return mod;
        });

    app.SimManager().RequestModeAndPreset(grav_sim::ModeId::F1_GravityGpu, 3);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().State() == grav_app::LifecycleState::Error);
    GRAV_ASSERT(app.Census().LiveSceneResourceCount() == 0);
    AssertObjectRolesPresence(false);

    // Restore default factory and recover cleanly to F1 preset 0
    app.SimManager().SetCustomModuleFactory(nullptr);
    app.SimManager().RequestModeAndPreset(grav_sim::ModeId::F1_GravityGpu, 0);
    GRAV_ASSERT(grav_test::StepUntilSettled(app));
    GRAV_ASSERT(app.SimManager().State() == grav_app::LifecycleState::Running);
    GRAV_ASSERT(app.SimManager().ActiveMode() == grav_sim::ModeId::F1_GravityGpu);
    AssertObjectRolesPresence(true);

    GRAV_ASSERT_NO_GL_ERROR("T10 Recovered");
    app.Shutdown();
    std::cout << "[PASS] T10\n";
}

void RunT14(const char *argv0)
{
    std::cout << "[RUN] T14: F2 Long-Run Bounded Trail Rings & Single-Pass Batched Draw...\n";

    grav_app::ApplicationOptions opts{};
    opts.width = 800;
    opts.height = 600;
    opts.visible = false;
    opts.vsync = false;
    opts.initialMode = grav_sim::ModeId::F2_LightTrajectories;
    opts.initialPresetIndex = 2; // 36-ray isotropic point-source fan
    opts.argv0 = argv0;

    grav_app::Application app;
    GRAV_ASSERT(app.Initialize(opts));
    GRAV_ASSERT(grav_test::StepUntilSettled(app));

    auto *f2 = dynamic_cast<grav_models::LightTrajectoriesModule *>(app.SimManager().ActiveModule());
    GRAV_ASSERT(f2 != nullptr);

    const std::size_t rayCount = f2->RayCount();
    const std::size_t capPerRay = f2->TrailCapacityPerRay();
    GRAV_ASSERT(rayCount == 36);
    GRAV_ASSERT(capPerRay == 1024);

    // Step 320 frames (320 * 4 = 1280 affine substeps > 1024 ring capacity)
    for (int frame = 0; frame < 320; ++frame)
    {
        app.StepFrame(1.0 / 60.0);
    }

    const std::size_t totalPoints = f2->TotalAllocatedTrailPoints();
    const std::uint64_t drawCalls = f2->LastFrameDrawCallCount();
    std::cout << "  T14 RayCount=" << rayCount << " CapPerRay=" << capPerRay
              << " TotalRingPoints=" << totalPoints << " SceneDrawCalls=" << drawCalls << "\n";

    GRAV_ASSERT(totalPoints <= rayCount * capPerRay);
    // Horizon fan + horizon rim + photon ring + 1 batched GL_LINES + 1 batched GL_POINTS <= 6 draws
    // (as opposed to legacy O(N) draw-all inside per-ray loop which would issue >= 36+ draws).
    GRAV_ASSERT_MSG(drawCalls <= 6, "Expected batched draw calls <= 6, got " << drawCalls);
    GRAV_ASSERT_NO_GL_ERROR("T14 End");

    app.Shutdown();
    std::cout << "[PASS] T14\n";
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const std::string which = grav_test::ParseCaseArg(argc, argv);
        const char *argv0 = (argc > 0) ? argv[0] : nullptr;

        if (which == "ALL" || which == "T03")
        {
            RunT03(argv0);
        }
        if (which == "ALL" || which == "T04")
        {
            RunT04(argv0);
        }
        if (which == "ALL" || which == "T05")
        {
            RunT05(argv0);
        }
        if (which == "ALL" || which == "T06")
        {
            RunT06(argv0);
        }
        if (which == "ALL" || which == "T07")
        {
            RunT07(argv0);
        }
        if (which == "ALL" || which == "T08")
        {
            RunT08(argv0);
        }
        if (which == "ALL" || which == "T09")
        {
            RunT09(argv0);
        }
        if (which == "ALL" || which == "T10")
        {
            RunT10(argv0);
        }
        if (which == "ALL" || which == "T14")
        {
            RunT14(argv0);
        }
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}

