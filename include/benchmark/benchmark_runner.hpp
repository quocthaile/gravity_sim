#pragma once

#include "models/simulation_module.hpp"
#include "render/resource_census.hpp"
#include "telemetry/telemetry_service.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace grav_bench
{

enum class BenchmarkState : std::uint8_t
{
    Idle = 0,
    Preparing,
    Warmup,
    Measuring,
    Completed,
    Aborted,
};

const char *BenchmarkStateName(BenchmarkState state);

struct BenchmarkConfig
{
    int warmupFrames = 60;
    int measuredFrames = 180;
    bool disableVsyncDuringRun = true;
    bool forceRecomputeEachIteration = true;
    bool hideHudDuringRun = false;
    bool autoExportOnComplete = true;
    std::string outputDirectory = "benchmarks";
};

struct WorkloadFingerprint
{
    grav_sim::ModeId mode = grav_sim::ModeId::F1_GravityGpu;
    std::string modeCode;
    std::string modeName;
    std::string presetId;
    std::string presetName;
    std::uint32_t presetRevision = 1;
    std::string solverId;
    std::string solverName;
    std::string solverStatus;
    std::string backendName;

    int displayWidth = 0;
    int displayHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    int gridDivisions = 200;
    bool gridVisible = true;
    bool hudVisible = true;
    bool vsyncEnabled = false;

    std::size_t primaryCount = 0;
    std::string primaryLabel = "Objects";
    bool hasSeed = false;
    std::uint32_t seed = 0;

    std::uint64_t cameraRevision = 0;
    std::uint64_t blackHoleRevision = 0;
};

struct DeviceEnvironmentInfo
{
    std::string glVendor;
    std::string glRenderer;
    std::string glVersion;
    std::string glslVersion;
    std::string dxgiAdapter;
    std::string osName = "Windows x64";
    std::string compilerInfo;
    std::string buildType;
};

struct BenchmarkFrameSample
{
    std::uint32_t sampleIndex = 0;
    double wallFrameMs = 0.0;
    double cpuUpdateMs = 0.0;
    double cpuRenderSubmitMs = 0.0;
    double cpuFenceWaitMs = 0.0;
    double cpuUiMs = 0.0;
    std::array<double, grav_telemetry::TelemetryService::kPassCount> gpuPassMs{};
    double modelStepsPerSec = 0.0;
    double imageEvaluationsPerSec = 0.0;
    double processCpuPercent = 0.0;
    std::size_t workingSetBytes = 0;
    std::size_t appAllocatedBytes = 0;
};

struct BenchmarkSummary
{
    std::size_t sampleCount = 0;
    double meanFrameMs = 0.0;
    double medianFrameMs = 0.0;
    double p95FrameMs = 0.0;
    double p99FrameMs = 0.0;
    double minFrameMs = 0.0;
    double maxFrameMs = 0.0;
    double presentationFps = 0.0;

    double meanCpuUpdateMs = 0.0;
    double meanCpuRenderSubmitMs = 0.0;
    double meanCpuUiMs = 0.0;
    std::array<double, grav_telemetry::TelemetryService::kPassCount> meanGpuPassMs{};

    double meanModelStepsPerSec = 0.0;
    double meanImageEvaluationsPerSec = 0.0;
    double lastModeLoadOrSwitchMs = 0.0;
    std::size_t runPeakAllocatedBytes = 0;
};

class BenchmarkRunner
{
  public:
    BenchmarkRunner() = default;

    void SetEnvironmentInfo(const DeviceEnvironmentInfo &env) { env_ = env; }
    const DeviceEnvironmentInfo &EnvironmentInfo() const { return env_; }

    BenchmarkConfig &MutableConfig() { return config_; }
    const BenchmarkConfig &Config() const { return config_; }

    BenchmarkState State() const { return state_; }
    bool IsActive() const
    {
        return state_ == BenchmarkState::Preparing || state_ == BenchmarkState::Warmup ||
               state_ == BenchmarkState::Measuring;
    }
    bool IsConfigLocked() const { return IsActive(); }
    bool ShouldForceStaticRecompute() const
    {
        return IsActive() && config_.forceRecomputeEachIteration;
    }

    void Start(const WorkloadFingerprint &initialFingerprint, double lastSwitchDurationMs,
               grav_render::ResourceCensus *census, const GpuMemoryManager *gpuManager);

    void Abort(const std::string &reason);

    // Validates that workload parameters have not drifted and records warmup/measured samples.
    void OnFrameSample(const WorkloadFingerprint &currentFingerprint, double wallFrameMs,
                       const grav_telemetry::TelemetryService &telemetry,
                       grav_render::ResourceCensus *census, const GpuMemoryManager *gpuManager);

    bool ExportResults(std::string *outCsvPath = nullptr, std::string *outManifestPath = nullptr);

    int WarmupFramesCompleted() const { return warmupDone_; }
    int MeasuredFramesCompleted() const { return static_cast<int>(samples_.size()); }
    float OverallProgress() const;

    const std::string &AbortReason() const { return abortReason_; }
    const std::string &LastCsvPath() const { return lastCsvPath_; }
    const std::string &LastManifestPath() const { return lastManifestPath_; }
    const BenchmarkSummary &Summary() const { return summary_; }
    const WorkloadFingerprint &LockedFingerprint() const { return lockedFingerprint_; }
    const std::vector<BenchmarkFrameSample> &Samples() const { return samples_; }

  private:
    bool CheckFingerprintInvariant(const WorkloadFingerprint &current,
                                   std::string &outMismatchReason) const;
    void ComputeSummary(grav_render::ResourceCensus *census, const GpuMemoryManager *gpuManager);

    BenchmarkConfig config_{};
    DeviceEnvironmentInfo env_{};
    BenchmarkState state_ = BenchmarkState::Idle;

    WorkloadFingerprint lockedFingerprint_{};
    double lastSwitchDurationMs_ = 0.0;
    int warmupDone_ = 0;

    std::vector<BenchmarkFrameSample> samples_;
    BenchmarkSummary summary_{};
    std::string abortReason_;
    std::string runTimestampTag_;
    std::string lastCsvPath_;
    std::string lastManifestPath_;
};

} // namespace grav_bench
