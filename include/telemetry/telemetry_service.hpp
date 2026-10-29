#pragma once

#include <GL/glew.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace grav_telemetry
{

enum class GpuPassId : std::uint8_t
{
    NBody = 0,
    Integrate,
    GridMaterialize,
    GridDeform,
    RayImageCompute,
    SceneDraw,
    UiDraw,
    Count,
};

const char *GpuPassName(GpuPassId pass);

struct SystemMetricsSnapshot
{
    bool cpuAvailable = false;
    double processCpuPercent = 0.0;
    std::uint32_t logicalCpuCount = 1;

    bool ramAvailable = false;
    std::size_t workingSetBytes = 0;
    std::size_t privateUsageBytes = 0;

    bool dxgiAvailable = false;
    std::string dxgiAdapterName;
    std::uint64_t dxgiUsageBytes = 0;
    std::uint64_t dxgiBudgetBytes = 0;

    // Device-wide utilization/temp/power require optional vendor SDKs (NVML/ADL/IGCL);
    // reported honestly as N/A when unavailable.
    bool gpuDeviceTelemetryAvailable = false;
};

struct PassTimingsSnapshot
{
    double cpuUpdateMs = 0.0;
    double cpuRenderSubmitMs = 0.0;
    double cpuFenceWaitMs = 0.0;
    double cpuUiMs = 0.0;

    std::array<double, static_cast<std::size_t>(GpuPassId::Count)> gpuPassMs{};
    std::array<bool, static_cast<std::size_t>(GpuPassId::Count)> gpuPassValid{};
};

class ScopedGpuPassQuery
{
  public:
    ScopedGpuPassQuery(class TelemetryService *service, GpuPassId pass,
                       std::uint64_t sceneGeneration);
    ~ScopedGpuPassQuery();

    ScopedGpuPassQuery(const ScopedGpuPassQuery &) = delete;
    ScopedGpuPassQuery &operator=(const ScopedGpuPassQuery &) = delete;

  private:
    class TelemetryService *service_ = nullptr;
    GpuPassId pass_ = GpuPassId::SceneDraw;
    bool active_ = false;
};

class TelemetryService
{
  public:
    static constexpr std::size_t kFrameHistorySize = 240;
    static constexpr std::size_t kQueryRingFrames = 5;
    static constexpr std::size_t kPassCount = static_cast<std::size_t>(GpuPassId::Count);

    TelemetryService() = default;
    ~TelemetryService();

    TelemetryService(const TelemetryService &) = delete;
    TelemetryService &operator=(const TelemetryService &) = delete;

    void Initialize(const std::string &glRendererString = {});
    void Shutdown();

    void BeginFrame(double wallDeltaSeconds, std::uint64_t currentSceneGeneration);
    void EndFrame();

    // Resets steady-state frame history and drops previous scene generation timings on mode switch.
    void OnSceneGenerationChanged(std::uint64_t newGeneration);

    bool BeginGpuPass(GpuPassId pass, std::uint64_t sceneGeneration);
    void EndGpuPass(GpuPassId pass);

    void AddCpuUpdateMs(double ms) { currentPassTimings_.cpuUpdateMs += ms; }
    void AddCpuRenderSubmitMs(double ms) { currentPassTimings_.cpuRenderSubmitMs += ms; }
    void AddCpuFenceWaitMs(double ms) { currentPassTimings_.cpuFenceWaitMs += ms; }
    void AddCpuUiMs(double ms) { currentPassTimings_.cpuUiMs += ms; }

    void RecordModelSteps(std::uint64_t stepsExecutedThisFrame);
    void RecordImageEvaluation(double evaluationDurationMs);
    void SetCurrentImageAgeMs(double ageMs) { currentImageAgeMs_ = ageMs; }

    void SetOsSamplingHz(double hz);
    double OsSamplingHz() const { return osSamplingHz_; }

    double PresentationFps() const { return smoothedFps_; }
    double LastFrameMs() const { return lastFrameMs_; }
    double ModelStepsPerSecond() const { return modelStepsPerSec_; }
    double ImageEvaluationsPerSecond() const { return imageEvaluationsPerSec_; }
    double CurrentImageAgeMs() const { return currentImageAgeMs_; }

    const std::array<float, kFrameHistorySize> &FrameMsHistory() const { return frameMsHistory_; }
    std::size_t FrameHistoryOffset() const { return frameHistoryCursor_; }
    std::size_t ValidFrameSamples() const { return validFrameSamples_; }

    const PassTimingsSnapshot &PassTimings() const { return publishedPassTimings_; }
    const SystemMetricsSnapshot &SystemMetrics() const { return systemMetrics_; }

  private:
    struct QuerySlot
    {
        GLuint queryId = 0;
        bool pending = false;
        std::uint64_t sceneGeneration = 0;
    };

    void CollectAvailableQueries(std::uint64_t activeGeneration);
    void PollSystemProvidersIfDue();
    void InitializeDxgiProvider(const std::string &glRendererString);
    void ShutdownDxgiProvider();

    bool initialized_ = false;
    bool gpuQueriesSupported_ = false;
    bool queryCurrentlyOpen_ = false;
    GpuPassId openQueryPass_ = GpuPassId::SceneDraw;

    std::array<std::array<QuerySlot, kPassCount>, kQueryRingFrames> queryRing_{};
    std::size_t writeRingIndex_ = 0;

    std::uint64_t activeGeneration_ = 1;
    double lastFrameMs_ = 0.0;
    double smoothedFps_ = 0.0;

    std::array<float, kFrameHistorySize> frameMsHistory_{};
    std::size_t frameHistoryCursor_ = 0;
    std::size_t validFrameSamples_ = 0;

    PassTimingsSnapshot currentPassTimings_{};
    PassTimingsSnapshot publishedPassTimings_{};

    // Rate counters over a 0.5 s rolling window
    double rateWindowAccumSec_ = 0.0;
    std::uint64_t accumModelSteps_ = 0;
    std::uint64_t accumImageEvals_ = 0;
    double modelStepsPerSec_ = 0.0;
    double imageEvaluationsPerSec_ = 0.0;
    double currentImageAgeMs_ = 0.0;

    // OS / DXGI sampling (default 3 Hz, within 2-4 Hz recommendation)
    double osSamplingHz_ = 3.0;
    std::chrono::steady_clock::time_point lastOsPollTime_{};
    std::uint64_t prevProcKernelPlusUser100ns_ = 0;
    std::chrono::steady_clock::time_point prevCpuSampleWall_{};
    bool hasPrevCpuSample_ = false;

    void *dxgiAdapter3Ptr_ = nullptr; // IDXGIAdapter3*, opaque in header to keep Windows headers out
    SystemMetricsSnapshot systemMetrics_{};
};

} // namespace grav_telemetry

