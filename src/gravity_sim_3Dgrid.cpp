#include "app/application.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace
{

bool ParseModeArg(const std::string &arg, grav_sim::ModeId &outMode)
{
    if (arg == "F1" || arg == "f1" || arg == "gravity_gpu")
    {
        outMode = grav_sim::ModeId::F1_GravityGpu;
        return true;
    }
    if (arg == "F2" || arg == "f2" || arg == "light_trajectories")
    {
        outMode = grav_sim::ModeId::F2_LightTrajectories;
        return true;
    }
    if (arg == "F3" || arg == "f3" || arg == "black_hole_gpu")
    {
        outMode = grav_sim::ModeId::F3_BlackHoleGpu;
        return true;
    }
    if (arg == "F4" || arg == "f4" || arg == "ray_tracing_cpu")
    {
        outMode = grav_sim::ModeId::F4_FlatRayTracingCpu;
        return true;
    }
    if (arg == "F5" || arg == "f5" || arg == "black_hole_cpu")
    {
        outMode = grav_sim::ModeId::F5_BlackHoleCpu;
        return true;
    }
    if (arg == "F6" || arg == "f6" || arg == "gravity_baseline")
    {
        outMode = grav_sim::ModeId::F6_GravityBaselineCpu;
        return true;
    }
    if (arg == "diag" || arg == "triangle")
    {
        outMode = grav_sim::ModeId::Diagnostic_Triangle;
        return true;
    }
    return false;
}

} // namespace

int main(int argc, char **argv)
{
    grav_app::ApplicationOptions options{};
    options.argv0 = (argc > 0) ? argv[0] : nullptr;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--hidden")
        {
            options.visible = false;
        }
        else if (arg == "--no-vsync")
        {
            options.vsync = false;
        }
        else if (arg == "--smoke-frames" && i + 1 < argc)
        {
            options.maxFrames = std::max(std::atoi(argv[++i]), 1);
        }
        else if (arg == "--mode" && i + 1 < argc)
        {
            grav_sim::ModeId parsed{};
            if (ParseModeArg(argv[++i], parsed))
            {
                options.initialMode = parsed;
            }
            else
            {
                std::cerr << "Unknown --mode value: " << argv[i] << "\n";
                return 2;
            }
        }
        else if (arg == "--preset" && i + 1 < argc)
        {
            options.initialPresetIndex = static_cast<std::size_t>(std::max(std::atoi(argv[++i]), 0));
        }
        else if (arg == "--asset-root" && i + 1 < argc)
        {
            options.explicitAssetRoot = argv[++i];
        }
        else if (arg == "--width" && i + 1 < argc)
        {
            options.width = std::max(std::atoi(argv[++i]), 320);
        }
        else if (arg == "--height" && i + 1 < argc)
        {
            options.height = std::max(std::atoi(argv[++i]), 240);
        }
    }

    grav_app::Application app;
    if (!app.Initialize(options))
    {
        return 1;
    }
    return app.Run();
}
