#include "benchmark/benchmark_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace grav_bench
{

namespace
{

constexpr const char *kGravitySimBaseCommit = "a5f9f930efaa828afa39ff79af00ed7b6add4ee1";
constexpr const char *kBlackHoleImportCommit = "dc263bb5a1b72f5a24190d2e38b84b5c8fe9a317";

std::string MakeTimestampTag()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
#if defined(_WIN32)
    localtime_s(&tmBuf, &tt);
#else
    localtime_r(&tt, &tmBuf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tmBuf, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string SanitizeFilenameToken(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (char ch : s)
    {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
            ch == '_' || ch == '-')
        {
            out.push_back(ch);
        }
        else
        {
            out.push_back('_');
        }
    }
    return out.empty() ? "run" : out;
}

std::string EscapeJsonString(const std::string &s)
{
    std::ostringstream oss;
    for (char c : s)
    {
        switch (c)
        {
        case '"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << c;
            break;
        }
    }
    return oss.str();
}

double PercentileSorted(const std::vector<double> &sorted, double quantile)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    const double idx = std::clamp(quantile, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

} // namespace

const char *BenchmarkStateName(BenchmarkState state)
{
    switch (state)
    {
    case BenchmarkState::Idle:
        return "Idle";
    case BenchmarkState::Preparing:
        return "Preparing";
    case BenchmarkState::Warmup:
        return "Warmup";
    case BenchmarkState::Measuring:
        return "Measuring";
    case BenchmarkState::Completed:
        return "Completed";
    case BenchmarkState::Aborted:
        return "Aborted";
    }
    return "Unknown";
}

void BenchmarkRunner::Start(const WorkloadFingerprint &initialFingerprint,
                            double lastSwitchDurationMs, grav_render::ResourceCensus *census,
                            const GpuMemoryManager *gpuManager)
{
    config_.warmupFrames = std::max(config_.warmupFrames, 0);
    config_.measuredFrames = std::max(config_.measuredFrames, 1);

    lockedFingerprint_ = initialFingerprint;
    lastSwitchDurationMs_ = lastSwitchDurationMs;
    warmupDone_ = 0;
    samples_.clear();
    samples_.reserve(static_cast<std::size_t>(config_.measuredFrames));
    summary_ = {};
    abortReason_.clear();
    lastCsvPath_.clear();
    lastManifestPath_.clear();
    runTimestampTag_ = MakeTimestampTag();

    if (census != nullptr)
    {
        census->ResetRunPeak(gpuManager);
    }

    state_ = BenchmarkState::Preparing;
}

void BenchmarkRunner::Abort(const std::string &reason)
{
    if (!IsActive())
    {
        return;
    }
    abortReason_ = reason.empty() ? "Aborted by user or configuration change" : reason;
    state_ = BenchmarkState::Aborted;
}

bool BenchmarkRunner::CheckFingerprintInvariant(const WorkloadFingerprint &current,
                                                std::string &outMismatchReason) const
{
    if (current.mode != lockedFingerprint_.mode)
    {
        outMismatchReason = "Active mode changed during benchmark run";
        return false;
    }
    if (current.presetId != lockedFingerprint_.presetId)
    {
        outMismatchReason = "Active preset changed during benchmark run";
        return false;
    }
    if (current.solverId != lockedFingerprint_.solverId)
    {
        outMismatchReason = "Active solver changed during benchmark run";
        return false;
    }
    if (current.displayWidth != lockedFingerprint_.displayWidth ||
        current.displayHeight != lockedFingerprint_.displayHeight)
    {
        outMismatchReason = "Display framebuffer resized during benchmark run";
        return false;
    }
    if (current.outputWidth != lockedFingerprint_.outputWidth ||
        current.outputHeight != lockedFingerprint_.outputHeight)
    {
        outMismatchReason = "Output compute/image resolution changed during benchmark run";
        return false;
    }
    if (current.gridVisible != lockedFingerprint_.gridVisible ||
        current.gridDivisions != lockedFingerprint_.gridDivisions)
    {
        outMismatchReason = "Grid visibility or topology changed during benchmark run";
        return false;
    }
    if (current.cameraRevision != lockedFingerprint_.cameraRevision)
    {
        outMismatchReason = "Camera moved/mutated during benchmark run";
        return false;
    }
    if (current.blackHoleRevision != lockedFingerprint_.blackHoleRevision)
    {
        outMismatchReason = "Shared Black Hole parameters changed during benchmark run";
        return false;
    }
    return true;
}

void BenchmarkRunner::OnFrameSample(const WorkloadFingerprint &currentFingerprint,
                                    double wallFrameMs,
                                    const grav_telemetry::TelemetryService &telemetry,
                                    grav_render::ResourceCensus *census,
                                    const GpuMemoryManager *gpuManager)
{
    if (!IsActive())
    {
        return;
    }

    if (state_ == BenchmarkState::Preparing)
    {
        // Lock exact post-apply fingerprint (e.g. after VSync/HUD run overrides take effect)
        lockedFingerprint_ = currentFingerprint;
        state_ = (config_.warmupFrames > 0) ? BenchmarkState::Warmup : BenchmarkState::Measuring;
        return;
    }

    std::string mismatchReason;
    if (!CheckFingerprintInvariant(currentFingerprint, mismatchReason))
    {
        Abort(mismatchReason);
        return;
    }

    if (state_ == BenchmarkState::Warmup)
    {
        ++warmupDone_;
        if (warmupDone_ >= config_.warmupFrames)
        {
            state_ = BenchmarkState::Measuring;
        }
        return;
    }

    if (state_ == BenchmarkState::Measuring)
    {
        BenchmarkFrameSample sample{};
        sample.sampleIndex = static_cast<std::uint32_t>(samples_.size());
        sample.wallFrameMs = wallFrameMs;

        const auto &pt = telemetry.PassTimings();
        sample.cpuUpdateMs = pt.cpuUpdateMs;
        sample.cpuRenderSubmitMs = pt.cpuRenderSubmitMs;
        sample.cpuFenceWaitMs = pt.cpuFenceWaitMs;
        sample.cpuUiMs = pt.cpuUiMs;
        sample.gpuPassMs = pt.gpuPassMs;

        sample.modelStepsPerSec = telemetry.ModelStepsPerSecond();
        sample.imageEvaluationsPerSec = telemetry.ImageEvaluationsPerSecond();

        const auto &sys = telemetry.SystemMetrics();
        sample.processCpuPercent = sys.processCpuPercent;
        sample.workingSetBytes = sys.workingSetBytes;

        if (census != nullptr)
        {
            const auto snap = census->Snapshot(gpuManager);
            sample.appAllocatedBytes = snap.totalAllocatedBytes;
        }
        else if (gpuManager != nullptr)
        {
            sample.appAllocatedBytes = gpuManager->AllocatedBytes();
        }

        samples_.push_back(sample);

        if (static_cast<int>(samples_.size()) >= config_.measuredFrames)
        {
            ComputeSummary(census, gpuManager);
            state_ = BenchmarkState::Completed;
            if (config_.autoExportOnComplete)
            {
                ExportResults();
            }
        }
    }
}

void BenchmarkRunner::ComputeSummary(grav_render::ResourceCensus *census,
                                     const GpuMemoryManager *gpuManager)
{
    summary_ = {};
    summary_.sampleCount = samples_.size();
    summary_.lastModeLoadOrSwitchMs = lastSwitchDurationMs_;

    if (census != nullptr)
    {
        summary_.runPeakAllocatedBytes = census->Snapshot(gpuManager).runPeakAllocatedBytes;
    }
    else if (gpuManager != nullptr)
    {
        summary_.runPeakAllocatedBytes = gpuManager->PeakAllocatedBytes();
    }

    if (samples_.empty())
    {
        return;
    }

    std::vector<double> sortedFrameMs;
    sortedFrameMs.reserve(samples_.size());

    double sumFrameMs = 0.0;
    double sumCpuUpdate = 0.0;
    double sumCpuRender = 0.0;
    double sumCpuUi = 0.0;
    double sumStepsRate = 0.0;
    double sumImgEvalRate = 0.0;
    std::array<double, grav_telemetry::TelemetryService::kPassCount> sumGpu{};

    for (const auto &s : samples_)
    {
        sortedFrameMs.push_back(s.wallFrameMs);
        sumFrameMs += s.wallFrameMs;
        sumCpuUpdate += s.cpuUpdateMs;
        sumCpuRender += s.cpuRenderSubmitMs;
        sumCpuUi += s.cpuUiMs;
        sumStepsRate += s.modelStepsPerSec;
        sumImgEvalRate += s.imageEvaluationsPerSec;
        for (std::size_t p = 0; p < sumGpu.size(); ++p)
        {
            sumGpu[p] += s.gpuPassMs[p];
        }
    }

    std::sort(sortedFrameMs.begin(), sortedFrameMs.end());
    const double invN = 1.0 / static_cast<double>(samples_.size());

    summary_.meanFrameMs = sumFrameMs * invN;
    summary_.medianFrameMs = PercentileSorted(sortedFrameMs, 0.50);
    summary_.p95FrameMs = PercentileSorted(sortedFrameMs, 0.95);
    summary_.p99FrameMs = PercentileSorted(sortedFrameMs, 0.99);
    summary_.minFrameMs = sortedFrameMs.front();
    summary_.maxFrameMs = sortedFrameMs.back();
    summary_.presentationFps =
        (summary_.meanFrameMs > 1.0e-6) ? (1000.0 / summary_.meanFrameMs) : 0.0;

    summary_.meanCpuUpdateMs = sumCpuUpdate * invN;
    summary_.meanCpuRenderSubmitMs = sumCpuRender * invN;
    summary_.meanCpuUiMs = sumCpuUi * invN;
    summary_.meanModelStepsPerSec = sumStepsRate * invN;
    summary_.meanImageEvaluationsPerSec = sumImgEvalRate * invN;
    for (std::size_t p = 0; p < sumGpu.size(); ++p)
    {
        summary_.meanGpuPassMs[p] = sumGpu[p] * invN;
    }
}

float BenchmarkRunner::OverallProgress() const
{
    if (state_ == BenchmarkState::Idle)
    {
        return 0.0f;
    }
    if (state_ == BenchmarkState::Completed)
    {
        return 1.0f;
    }
    const int total = std::max(config_.warmupFrames + config_.measuredFrames, 1);
    const int done = warmupDone_ + static_cast<int>(samples_.size());
    return std::clamp(static_cast<float>(done) / static_cast<float>(total), 0.0f, 1.0f);
}

bool BenchmarkRunner::ExportResults(std::string *outCsvPath, std::string *outManifestPath)
{
    if (state_ != BenchmarkState::Completed && state_ != BenchmarkState::Aborted)
    {
        return false;
    }

    std::error_code ec;
    std::filesystem::path outDir(config_.outputDirectory.empty() ? "benchmarks"
                                                                 : config_.outputDirectory);
    std::filesystem::create_directories(outDir, ec);

    const std::string tag = runTimestampTag_.empty() ? MakeTimestampTag() : runTimestampTag_;
    const std::string stem =
        tag + "_" + SanitizeFilenameToken(lockedFingerprint_.modeCode) + "_" +
        SanitizeFilenameToken(lockedFingerprint_.presetId) +
        (state_ == BenchmarkState::Aborted ? "_ABORTED" : "");

    const std::filesystem::path csvPath = outDir / (stem + "_samples.csv");
    const std::filesystem::path jsonPath = outDir / (stem + "_manifest.json");

    {
        std::ofstream csv(csvPath);
        if (!csv.is_open())
        {
            return false;
        }
        csv << "# schema_version=1\n";
        csv << "# status=" << BenchmarkStateName(state_) << "\n";
        csv << "sample_index,wall_frame_ms,cpu_update_ms,cpu_render_submit_ms,cpu_fence_wait_ms,"
               "cpu_ui_ms,gpu_nbody_ms,gpu_integrate_ms,gpu_grid_materialize_ms,"
               "gpu_grid_deform_ms,gpu_ray_image_ms,gpu_scene_draw_ms,gpu_ui_draw_ms,"
               "model_steps_per_s,image_evals_per_s,process_cpu_pct,working_set_bytes,"
               "app_allocated_bytes\n";
        csv << std::fixed << std::setprecision(6);
        for (const auto &s : samples_)
        {
            csv << s.sampleIndex << ',' << s.wallFrameMs << ',' << s.cpuUpdateMs << ','
                << s.cpuRenderSubmitMs << ',' << s.cpuFenceWaitMs << ',' << s.cpuUiMs << ','
                << s.gpuPassMs[0] << ',' << s.gpuPassMs[1] << ',' << s.gpuPassMs[2] << ','
                << s.gpuPassMs[3] << ',' << s.gpuPassMs[4] << ',' << s.gpuPassMs[5] << ','
                << s.gpuPassMs[6] << ',' << s.modelStepsPerSec << ',' << s.imageEvaluationsPerSec
                << ',' << s.processCpuPercent << ',' << s.workingSetBytes << ','
                << s.appAllocatedBytes << '\n';
        }
        if (!csv.good())
        {
            return false;
        }
    }

    {
        std::ofstream jf(jsonPath);
        if (!jf.is_open())
        {
            return false;
        }
        jf << std::fixed << std::setprecision(6);
        jf << "{\n";
        jf << "  \"schema_version\": 1,\n";
        jf << "  \"status\": \"" << BenchmarkStateName(state_) << "\",\n";
        jf << "  \"abort_reason\": \"" << EscapeJsonString(abortReason_) << "\",\n";
        jf << "  \"provenance\": {\n";
        jf << "    \"gravity_sim_base_commit\": \"" << kGravitySimBaseCommit << "\",\n";
        jf << "    \"black_hole_import_commit\": \"" << kBlackHoleImportCommit << "\"\n";
        jf << "  },\n";
        jf << "  \"environment\": {\n";
        jf << "    \"os\": \"" << EscapeJsonString(env_.osName) << "\",\n";
        jf << "    \"compiler\": \"" << EscapeJsonString(env_.compilerInfo) << "\",\n";
        jf << "    \"build_type\": \"" << EscapeJsonString(env_.buildType) << "\",\n";
        jf << "    \"gl_vendor\": \"" << EscapeJsonString(env_.glVendor) << "\",\n";
        jf << "    \"gl_renderer\": \"" << EscapeJsonString(env_.glRenderer) << "\",\n";
        jf << "    \"gl_version\": \"" << EscapeJsonString(env_.glVersion) << "\",\n";
        jf << "    \"glsl_version\": \"" << EscapeJsonString(env_.glslVersion) << "\",\n";
        jf << "    \"dxgi_adapter\": \"" << EscapeJsonString(env_.dxgiAdapter) << "\"\n";
        jf << "  },\n";
        jf << "  \"workload\": {\n";
        jf << "    \"mode_code\": \"" << EscapeJsonString(lockedFingerprint_.modeCode) << "\",\n";
        jf << "    \"mode_name\": \"" << EscapeJsonString(lockedFingerprint_.modeName) << "\",\n";
        jf << "    \"preset_id\": \"" << EscapeJsonString(lockedFingerprint_.presetId) << "\",\n";
        jf << "    \"preset_name\": \"" << EscapeJsonString(lockedFingerprint_.presetName) << "\",\n";
        jf << "    \"preset_revision\": " << lockedFingerprint_.presetRevision << ",\n";
        jf << "    \"solver_id\": \"" << EscapeJsonString(lockedFingerprint_.solverId) << "\",\n";
        jf << "    \"solver_name\": \"" << EscapeJsonString(lockedFingerprint_.solverName) << "\",\n";
        jf << "    \"solver_status\": \"" << EscapeJsonString(lockedFingerprint_.solverStatus) << "\",\n";
        jf << "    \"backend\": \"" << EscapeJsonString(lockedFingerprint_.backendName) << "\",\n";
        jf << "    \"display_width\": " << lockedFingerprint_.displayWidth << ",\n";
        jf << "    \"display_height\": " << lockedFingerprint_.displayHeight << ",\n";
        jf << "    \"output_width\": " << lockedFingerprint_.outputWidth << ",\n";
        jf << "    \"output_height\": " << lockedFingerprint_.outputHeight << ",\n";
        jf << "    \"grid_divisions\": " << lockedFingerprint_.gridDivisions << ",\n";
        jf << "    \"grid_visible\": " << (lockedFingerprint_.gridVisible ? "true" : "false") << ",\n";
        jf << "    \"hud_visible\": " << (lockedFingerprint_.hudVisible ? "true" : "false") << ",\n";
        jf << "    \"vsync_enabled\": " << (lockedFingerprint_.vsyncEnabled ? "true" : "false") << ",\n";
        jf << "    \"force_recompute_each_iteration\": "
           << (config_.forceRecomputeEachIteration ? "true" : "false") << ",\n";
        jf << "    \"primary_count\": " << lockedFingerprint_.primaryCount << ",\n";
        jf << "    \"primary_label\": \"" << EscapeJsonString(lockedFingerprint_.primaryLabel) << "\",\n";
        jf << "    \"has_seed\": " << (lockedFingerprint_.hasSeed ? "true" : "false") << ",\n";
        jf << "    \"seed\": " << lockedFingerprint_.seed << ",\n";
        jf << "    \"warmup_frames_target\": " << config_.warmupFrames << ",\n";
        jf << "    \"warmup_frames_completed\": " << warmupDone_ << ",\n";
        jf << "    \"measured_frames_target\": " << config_.measuredFrames << ",\n";
        jf << "    \"measured_frames_completed\": " << samples_.size() << "\n";
        jf << "  },\n";
        jf << "  \"summary\": {\n";
        jf << "    \"sample_count\": " << summary_.sampleCount << ",\n";
        jf << "    \"mean_frame_ms\": " << summary_.meanFrameMs << ",\n";
        jf << "    \"median_frame_ms\": " << summary_.medianFrameMs << ",\n";
        jf << "    \"p95_frame_ms\": " << summary_.p95FrameMs << ",\n";
        jf << "    \"p99_frame_ms\": " << summary_.p99FrameMs << ",\n";
        jf << "    \"min_frame_ms\": " << summary_.minFrameMs << ",\n";
        jf << "    \"max_frame_ms\": " << summary_.maxFrameMs << ",\n";
        jf << "    \"presentation_fps\": " << summary_.presentationFps << ",\n";
        jf << "    \"mean_cpu_update_ms\": " << summary_.meanCpuUpdateMs << ",\n";
        jf << "    \"mean_cpu_render_submit_ms\": " << summary_.meanCpuRenderSubmitMs << ",\n";
        jf << "    \"mean_cpu_ui_ms\": " << summary_.meanCpuUiMs << ",\n";
        jf << "    \"mean_model_steps_per_s\": " << summary_.meanModelStepsPerSec << ",\n";
        jf << "    \"mean_image_evaluations_per_s\": " << summary_.meanImageEvaluationsPerSec << ",\n";
        jf << "    \"mode_load_or_switch_ms\": " << summary_.lastModeLoadOrSwitchMs << ",\n";
        jf << "    \"run_peak_allocated_bytes\": " << summary_.runPeakAllocatedBytes << "\n";
        jf << "  }\n";
        jf << "}\n";
        if (!jf.good())
        {
            return false;
        }
    }

    lastCsvPath_ = csvPath.string();
    lastManifestPath_ = jsonPath.string();
    if (outCsvPath != nullptr)
    {
        *outCsvPath = lastCsvPath_;
    }
    if (outManifestPath != nullptr)
    {
        *outManifestPath = lastManifestPath_;
    }
    return true;
}

} // namespace grav_bench

