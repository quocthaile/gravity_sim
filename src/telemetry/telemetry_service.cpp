#include "telemetry/telemetry_service.hpp"

#include <algorithm>
#include <cmath>
#include <cwctype>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <psapi.h>
#endif

namespace grav_telemetry
{

const char *GpuPassName(GpuPassId pass)
{
    switch (pass)
    {
    case GpuPassId::NBody:
        return "N-Body Force";
    case GpuPassId::Integrate:
        return "Integration";
    case GpuPassId::GridMaterialize:
        return "Grid Materialize";
    case GpuPassId::GridDeform:
        return "Grid Deform";
    case GpuPassId::RayImageCompute:
        return "Ray Image Compute";
    case GpuPassId::SceneDraw:
        return "Scene Draw";
    case GpuPassId::UiDraw:
        return "UI Draw";
    case GpuPassId::Count:
        break;
    }
    return "Unknown";
}

ScopedGpuPassQuery::ScopedGpuPassQuery(TelemetryService *service, GpuPassId pass,
                                       std::uint64_t sceneGeneration)
    : service_(service), pass_(pass)
{
    if (service_ != nullptr)
    {
        active_ = service_->BeginGpuPass(pass_, sceneGeneration);
    }
}

ScopedGpuPassQuery::~ScopedGpuPassQuery()
{
    if (active_ && service_ != nullptr)
    {
        service_->EndGpuPass(pass_);
    }
}

TelemetryService::~TelemetryService() { ShutdownDxgiProvider(); }

void TelemetryService::Initialize(const std::string &glRendererString)
{
    if (initialized_)
    {
        return;
    }

    gpuQueriesSupported_ = (glGenQueries != nullptr && glBeginQuery != nullptr &&
                            glEndQuery != nullptr && glGetQueryObjectuiv != nullptr);
    if (gpuQueriesSupported_)
    {
        for (auto &frameSlots : queryRing_)
        {
            for (auto &slot : frameSlots)
            {
                glGenQueries(1, &slot.queryId);
                slot.pending = false;
                slot.sceneGeneration = 0;
            }
        }
    }

#ifdef _WIN32
    SYSTEM_INFO sysInfo{};
    GetSystemInfo(&sysInfo);
    systemMetrics_.logicalCpuCount =
        sysInfo.dwNumberOfProcessors > 0 ? static_cast<std::uint32_t>(sysInfo.dwNumberOfProcessors)
                                         : 1u;
#endif

    InitializeDxgiProvider(glRendererString);
    initialized_ = true;
    PollSystemProvidersIfDue();
}

void TelemetryService::Shutdown()
{
    if (initialized_ && gpuQueriesSupported_)
    {
        if (queryCurrentlyOpen_)
        {
            glEndQuery(GL_TIME_ELAPSED);
            queryCurrentlyOpen_ = false;
        }
        for (auto &frameSlots : queryRing_)
        {
            for (auto &slot : frameSlots)
            {
                if (slot.queryId != 0)
                {
                    glDeleteQueries(1, &slot.queryId);
                    slot.queryId = 0;
                }
                slot.pending = false;
            }
        }
    }
    ShutdownDxgiProvider();
    initialized_ = false;
}

void TelemetryService::OnSceneGenerationChanged(std::uint64_t newGeneration)
{
    activeGeneration_ = newGeneration;
    frameMsHistory_.fill(0.0f);
    frameHistoryCursor_ = 0;
    validFrameSamples_ = 0;
    smoothedFps_ = 0.0;
    lastFrameMs_ = 0.0;

    publishedPassTimings_ = PassTimingsSnapshot{};
    currentPassTimings_ = PassTimingsSnapshot{};

    rateWindowAccumSec_ = 0.0;
    accumModelSteps_ = 0;
    accumImageEvals_ = 0;
    modelStepsPerSec_ = 0.0;
    imageEvaluationsPerSec_ = 0.0;
    currentImageAgeMs_ = 0.0;
}

void TelemetryService::BeginFrame(double wallDeltaSeconds, std::uint64_t currentSceneGeneration)
{
    activeGeneration_ = currentSceneGeneration;

    if (wallDeltaSeconds > 0.0 && wallDeltaSeconds < 10.0)
    {
        lastFrameMs_ = wallDeltaSeconds * 1000.0;
        const double instFps = 1.0 / wallDeltaSeconds;
        smoothedFps_ = (smoothedFps_ <= 0.0) ? instFps : (0.9 * smoothedFps_ + 0.1 * instFps);

        frameMsHistory_[frameHistoryCursor_] = static_cast<float>(lastFrameMs_);
        frameHistoryCursor_ = (frameHistoryCursor_ + 1) % kFrameHistorySize;
        validFrameSamples_ = std::min(validFrameSamples_ + 1, kFrameHistorySize);

        rateWindowAccumSec_ += wallDeltaSeconds;
        if (rateWindowAccumSec_ >= 0.5)
        {
            modelStepsPerSec_ = static_cast<double>(accumModelSteps_) / rateWindowAccumSec_;
            imageEvaluationsPerSec_ = static_cast<double>(accumImageEvals_) / rateWindowAccumSec_;
            accumModelSteps_ = 0;
            accumImageEvals_ = 0;
            rateWindowAccumSec_ = 0.0;
        }
    }

    currentPassTimings_.cpuUpdateMs = 0.0;
    currentPassTimings_.cpuRenderSubmitMs = 0.0;
    currentPassTimings_.cpuFenceWaitMs = 0.0;
    currentPassTimings_.cpuUiMs = 0.0;

    if (initialized_ && gpuQueriesSupported_)
    {
        CollectAvailableQueries(currentSceneGeneration);
    }
    PollSystemProvidersIfDue();
}

void TelemetryService::EndFrame()
{
    if (queryCurrentlyOpen_ && gpuQueriesSupported_)
    {
        glEndQuery(GL_TIME_ELAPSED);
        queryCurrentlyOpen_ = false;
    }

    publishedPassTimings_.cpuUpdateMs = currentPassTimings_.cpuUpdateMs;
    publishedPassTimings_.cpuRenderSubmitMs = currentPassTimings_.cpuRenderSubmitMs;
    publishedPassTimings_.cpuFenceWaitMs = currentPassTimings_.cpuFenceWaitMs;
    publishedPassTimings_.cpuUiMs = currentPassTimings_.cpuUiMs;

    writeRingIndex_ = (writeRingIndex_ + 1) % kQueryRingFrames;
}

bool TelemetryService::BeginGpuPass(GpuPassId pass, std::uint64_t sceneGeneration)
{
    if (!initialized_ || !gpuQueriesSupported_ || queryCurrentlyOpen_)
    {
        return false;
    }

    const std::size_t passIdx = static_cast<std::size_t>(pass);
    QuerySlot &slot = queryRing_[writeRingIndex_][passIdx];
    if (slot.queryId == 0 || slot.pending)
    {
        // Slot from kQueryRingFrames ago is still in-flight; never block waiting for it.
        return false;
    }

    glBeginQuery(GL_TIME_ELAPSED, slot.queryId);
    slot.pending = true;
    slot.sceneGeneration = sceneGeneration;
    queryCurrentlyOpen_ = true;
    openQueryPass_ = pass;
    return true;
}

void TelemetryService::EndGpuPass(GpuPassId pass)
{
    if (!queryCurrentlyOpen_ || pass != openQueryPass_)
    {
        return;
    }
    glEndQuery(GL_TIME_ELAPSED);
    queryCurrentlyOpen_ = false;
}

void TelemetryService::CollectAvailableQueries(std::uint64_t activeGeneration)
{
    for (std::size_t f = 0; f < kQueryRingFrames; ++f)
    {
        if (f == writeRingIndex_)
        {
            continue;
        }
        for (std::size_t p = 0; p < kPassCount; ++p)
        {
            QuerySlot &slot = queryRing_[f][p];
            if (!slot.pending || slot.queryId == 0)
            {
                continue;
            }

            GLuint available = GL_FALSE;
            glGetQueryObjectuiv(slot.queryId, GL_QUERY_RESULT_AVAILABLE, &available);
            if (available == GL_TRUE)
            {
                GLuint64 elapsedNs = 0;
                glGetQueryObjectui64v(slot.queryId, GL_QUERY_RESULT, &elapsedNs);
                slot.pending = false;

                // Discard late results from previous scene generations so timings never leak across
                // mode transitions.
                if (slot.sceneGeneration == activeGeneration)
                {
                    publishedPassTimings_.gpuPassMs[p] = static_cast<double>(elapsedNs) * 1.0e-6;
                    publishedPassTimings_.gpuPassValid[p] = true;
                }
            }
        }
    }
}

void TelemetryService::RecordModelSteps(std::uint64_t stepsExecutedThisFrame)
{
    accumModelSteps_ += stepsExecutedThisFrame;
}

void TelemetryService::RecordImageEvaluation(double evaluationDurationMs)
{
    (void)evaluationDurationMs;
    ++accumImageEvals_;
    currentImageAgeMs_ = 0.0;
}

void TelemetryService::SetOsSamplingHz(double hz)
{
    osSamplingHz_ = std::clamp(hz, 0.5, 10.0);
}

void TelemetryService::InitializeDxgiProvider(const std::string &glRendererString)
{
    ShutdownDxgiProvider();
#ifdef _WIN32
    IDXGIFactory4 *factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void **>(&factory))) ||
        factory == nullptr)
    {
        return;
    }

    std::string lowerRenderer = glRendererString;
    std::transform(lowerRenderer.begin(), lowerRenderer.end(), lowerRenderer.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    IDXGIAdapter3 *bestAdapter = nullptr;
    std::string bestName;

    for (UINT idx = 0;; ++idx)
    {
        IDXGIAdapter1 *adapter1 = nullptr;
        if (factory->EnumAdapters1(idx, &adapter1) == DXGI_ERROR_NOT_FOUND)
        {
            break;
        }
        DXGI_ADAPTER_DESC1 desc{};
        adapter1->GetDesc1(&desc);

        const bool isSoftware = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        IDXGIAdapter3 *adapter3 = nullptr;
        if (!isSoftware &&
            SUCCEEDED(adapter1->QueryInterface(__uuidof(IDXGIAdapter3),
                                               reinterpret_cast<void **>(&adapter3))) &&
            adapter3 != nullptr)
        {
            char narrowName[256] = {};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, narrowName,
                                static_cast<int>(sizeof(narrowName)), nullptr, nullptr);
            std::string candidateName(narrowName);
            std::string lowerCandidate = candidateName;
            std::transform(lowerCandidate.begin(), lowerCandidate.end(), lowerCandidate.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

            const bool matchesRenderer =
                !lowerRenderer.empty() &&
                (lowerRenderer.find(lowerCandidate) != std::string::npos ||
                 (lowerRenderer.find("nvidia") != std::string::npos &&
                  lowerCandidate.find("nvidia") != std::string::npos) ||
                 (lowerRenderer.find("amd") != std::string::npos &&
                  lowerCandidate.find("amd") != std::string::npos) ||
                 (lowerRenderer.find("radeon") != std::string::npos &&
                  lowerCandidate.find("radeon") != std::string::npos) ||
                 (lowerRenderer.find("intel") != std::string::npos &&
                  lowerCandidate.find("intel") != std::string::npos));

            if (bestAdapter == nullptr || matchesRenderer)
            {
                if (bestAdapter != nullptr)
                {
                    bestAdapter->Release();
                }
                bestAdapter = adapter3;
                bestName = candidateName;
                if (matchesRenderer)
                {
                    adapter1->Release();
                    break;
                }
            }
            else
            {
                adapter3->Release();
            }
        }
        adapter1->Release();
    }

    factory->Release();

    if (bestAdapter != nullptr)
    {
        dxgiAdapter3Ptr_ = bestAdapter;
        systemMetrics_.dxgiAdapterName = bestName;
    }
#else
    (void)glRendererString;
#endif
}

void TelemetryService::ShutdownDxgiProvider()
{
#ifdef _WIN32
    if (dxgiAdapter3Ptr_ != nullptr)
    {
        static_cast<IDXGIAdapter3 *>(dxgiAdapter3Ptr_)->Release();
        dxgiAdapter3Ptr_ = nullptr;
    }
#endif
    systemMetrics_.dxgiAvailable = false;
}

void TelemetryService::PollSystemProvidersIfDue()
{
    const auto now = std::chrono::steady_clock::now();
    if (lastOsPollTime_.time_since_epoch().count() != 0)
    {
        const double elapsedSec = std::chrono::duration<double>(now - lastOsPollTime_).count();
        if (elapsedSec < (1.0 / osSamplingHz_))
        {
            return;
        }
    }
    lastOsPollTime_ = now;

#ifdef _WIN32
    const HANDLE process = GetCurrentProcess();

    FILETIME creationTime{}, exitTime{}, kernelTime{}, userTime{};
    if (GetProcessTimes(process, &creationTime, &exitTime, &kernelTime, &userTime) != FALSE)
    {
        ULARGE_INTEGER k{}, u{};
        k.LowPart = kernelTime.dwLowDateTime;
        k.HighPart = kernelTime.dwHighDateTime;
        u.LowPart = userTime.dwLowDateTime;
        u.HighPart = userTime.dwHighDateTime;
        const std::uint64_t total100ns = k.QuadPart + u.QuadPart;

        if (hasPrevCpuSample_)
        {
            const double wallSec = std::chrono::duration<double>(now - prevCpuSampleWall_).count();
            if (wallSec > 1e-4 && total100ns >= prevProcKernelPlusUser100ns_)
            {
                const double procSec =
                    static_cast<double>(total100ns - prevProcKernelPlusUser100ns_) * 1.0e-7;
                const double denom =
                    wallSec * static_cast<double>(std::max(systemMetrics_.logicalCpuCount, 1u));
                systemMetrics_.processCpuPercent = std::clamp((procSec / denom) * 100.0, 0.0, 100.0);
                systemMetrics_.cpuAvailable = true;
            }
        }
        prevProcKernelPlusUser100ns_ = total100ns;
        prevCpuSampleWall_ = now;
        hasPrevCpuSample_ = true;
    }

    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(process, reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
                             sizeof(pmc)) != FALSE)
    {
        systemMetrics_.workingSetBytes = static_cast<std::size_t>(pmc.WorkingSetSize);
        systemMetrics_.privateUsageBytes = static_cast<std::size_t>(pmc.PrivateUsage);
        systemMetrics_.ramAvailable = true;
    }
    else
    {
        systemMetrics_.ramAvailable = false;
    }

    if (dxgiAdapter3Ptr_ != nullptr)
    {
        DXGI_QUERY_VIDEO_MEMORY_INFO memInfo{};
        auto *adapter3 = static_cast<IDXGIAdapter3 *>(dxgiAdapter3Ptr_);
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo)))
        {
            systemMetrics_.dxgiUsageBytes = memInfo.CurrentUsage;
            systemMetrics_.dxgiBudgetBytes = memInfo.Budget;
            systemMetrics_.dxgiAvailable = true;
        }
        else
        {
            systemMetrics_.dxgiAvailable = false;
        }
    }
#endif
}

} // namespace grav_telemetry

