#pragma once

#include "render/shared_grid.hpp"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace grav_models
{

constexpr double kSpeedOfLightMps = 299792458.0;
constexpr double kGravConstSi = 6.67430e-11;
constexpr double kSagittariusAMassKg = 8.54e36;

inline double ComputeSchwarzschildRadiusMeters(double massKg)
{
    return (2.0 * kGravConstSi * massKg) / (kSpeedOfLightMps * kSpeedOfLightMps);
}

struct BlackHoleParameters
{
    glm::dvec3 positionMeters{0.0, 0.0, 0.0};
    double massKg = kSagittariusAMassKg;
    double schwarzschildRadiusMeters = ComputeSchwarzschildRadiusMeters(kSagittariusAMassKg);
    std::uint64_t revision = 1;

    void SetMassKg(double newMassKg);
    void SetPositionMeters(const glm::dvec3 &newPosMeters);
    void ResetToSagittariusA();
    grav_render::GridSource ToGridSource(float metersPerWorldUnit) const;
};

// ============================================================================
// Fixed std140 UBO layouts for F3 Black-Hole GPU compute shaders
// Matches shaders/black_hole/geodesic_rk4.comp and geodesic_legacy.comp
// ============================================================================

constexpr std::size_t kMaxBlackHoleSceneObjects = 16;

struct alignas(16) GpuCameraUboStd140
{
    glm::vec4 camPos_tanHalfFov{0.0f};   // xyz: camera pos (m), w: tan(fovY / 2)
    glm::vec4 camRight_aspect{0.0f};     // xyz: camera right, w: aspect ratio
    glm::vec4 camUp_moving{0.0f};        // xyz: camera up, w: moving flag (0 or 1)
    glm::vec4 camForward_maxSteps{0.0f}; // xyz: camera forward, w: max integration steps
    glm::vec4 bhCenter_rs{0.0f};         // xyz: black hole pos (m), w: rs (m)
    glm::vec4 stepParams{0.0f};          // x: dLambda (m), y: escapeRadius (m), z: outW, w: outH
};

struct alignas(16) GpuDiskUboStd140
{
    glm::vec4 diskParams{0.0f}; // x: innerR (m), y: outerR (m), z: enabled (1/0), w: thickness (m)
};

struct alignas(16) GpuObjectsUboStd140
{
    glm::ivec4 counts{0}; // x: numObjects (0..16)
    glm::vec4 objPosRadius[kMaxBlackHoleSceneObjects]{};
    glm::vec4 objColor[kMaxBlackHoleSceneObjects]{};
    glm::vec4 objMassPad[kMaxBlackHoleSceneObjects]{};
};

static_assert(sizeof(GpuCameraUboStd140) == 6 * sizeof(glm::vec4),
              "GpuCameraUboStd140 must be 96 bytes");
static_assert(offsetof(GpuCameraUboStd140, bhCenter_rs) == 4 * sizeof(glm::vec4),
              "bhCenter_rs std140 offset mismatch");
static_assert(sizeof(GpuDiskUboStd140) == sizeof(glm::vec4), "GpuDiskUboStd140 must be 16 bytes");
static_assert(sizeof(GpuObjectsUboStd140) == 49 * sizeof(glm::vec4),
              "GpuObjectsUboStd140 must be 784 bytes");
static_assert(offsetof(GpuObjectsUboStd140, objPosRadius) == 16,
              "objPosRadius std140 offset mismatch");
static_assert(offsetof(GpuObjectsUboStd140, objColor) == 16 + 16 * 16,
              "objColor std140 offset mismatch");
static_assert(offsetof(GpuObjectsUboStd140, objMassPad) == 16 + 32 * 16,
              "objMassPad std140 offset mismatch");

// ============================================================================
// Planar (2D) Schwarzschild Null-Geodesic State & Solvers (F2 & W8)
// Separated from trail buffer so RK4 stages never copy trail history!
// ============================================================================

enum class PlanarGeodesicSolverKind : std::uint8_t
{
    LegacyPolarRk4 = 0,        // rk4_planar_legacy_v1 (exact 2D_lensing.cpp equations)
    CanonicalDimensionlessRk4, // rk4_canonical_v1 (dimensionless Cartesian u = r / rs)
};

struct PlanarRayState
{
    // Relative to BH center in meters
    double x = 0.0;
    double y = 0.0;

    // Polar state (for LegacyPolarRk4 and diagnostic readouts)
    double r = 0.0;
    double phi = 0.0;
    double dr = 0.0;
    double dphi = 0.0;
    double E = 0.0;
    double L = 0.0;

    // Dimensionless Cartesian state u = pos / rs, v = du / dh (for CanonicalDimensionlessRk4)
    glm::dvec2 uPos{0.0, 0.0};
    glm::dvec2 uVel{0.0, 0.0};
    double uE0Squared = 0.0;
    double uL0 = 0.0;

    bool captured = false;
    bool escaped = false;
};

PlanarRayState InitializePlanarRay(const glm::dvec2 &relPosMeters, const glm::dvec2 &velocityMps,
                                   double rsMeters);

void StepPlanarRay(PlanarRayState &ray, double dLambdaAffine, double rsMeters,
                   double escapeRadiusMeters, PlanarGeodesicSolverKind solverKind);

// Evaluates instantaneous relative drift in null condition g_{uv} k^u k^v = 0 and angular momentum
// directly from current phase-space coordinates (not from cached constants E, L).
double EvaluatePlanarNullRelativeError(const PlanarRayState &ray, double rsMeters,
                                       PlanarGeodesicSolverKind solverKind);
double EvaluatePlanarAngularMomentumRelativeError(const PlanarRayState &ray, double rsMeters,
                                                  PlanarGeodesicSolverKind solverKind);

// ============================================================================
// 3D Schwarzschild Null-Geodesic Scene & Solvers (F3 CPU parity, F5, W8/T13)
// ============================================================================

struct BlackHoleSceneObject
{
    glm::vec4 posRadiusMeters{0.0f}; // xyz: world pos (m), w: radius (m)
    glm::vec4 color{1.0f};
    float massKg = 0.0f;
    glm::vec3 velocityMps{0.0f};
};

struct BlackHoleSceneConfig
{
    glm::dvec3 camPosMeters{0.0, 0.0, 6.34194e10};
    glm::dvec3 camRight{1.0, 0.0, 0.0};
    glm::dvec3 camUp{0.0, 1.0, 0.0};
    glm::dvec3 camForward{0.0, 0.0, -1.0};
    double tanHalfFov = 0.57735026919; // tan(60 deg / 2)
    double aspect = 4.0 / 3.0;

    glm::dvec3 bhCenterMeters{0.0, 0.0, 0.0};
    double rsMeters = ComputeSchwarzschildRadiusMeters(kSagittariusAMassKg);

    double dLambdaMeters = 1.0e7;
    double escapeRadiusMeters = 1.0e14;
    int maxSteps = 20000;

    bool diskEnabled = true;
    double diskInnerMeters = 0.0;
    double diskOuterMeters = 0.0;
    double diskThicknessMeters = 1.0e9;

    // F5 CPU-geodesic.cpp compatibility option: render horizon hits in red (1,0,0) vs black (0,0,0)
    bool redSilhouetteStyle = false;
    glm::vec4 backgroundColor{0.02f, 0.02f, 0.05f, 1.0f};

    std::vector<BlackHoleSceneObject> objects;
};

enum class Geodesic3DSolverKind : std::uint8_t
{
    GeometricSilhouette = 0,   // Straight-ray sphere silhouette (CPU-geodesic.cpp useGeodesics=false)
    LegacySphericalEuler,      // euler_spherical_legacy_v1 (matches geodesic_legacy.comp)
    LegacySphericalRk4,        // rk4_spherical_legacy_v1 (matches CPU-geodesic.cpp useGeodesics=true)
    CanonicalDimensionlessRk4, // rk4_canonical_v1 (matches geodesic_rk4.comp)
};

enum class GeodesicOutcome : std::uint8_t
{
    Escaped = 0,
    HitHorizon,
    HitDisk,
    HitObject,
};

struct GeodesicTraceResult
{
    GeodesicOutcome outcome = GeodesicOutcome::Escaped;
    glm::vec4 color{0.02f, 0.02f, 0.05f, 1.0f};
    glm::dvec3 finalRelPosMeters{0.0};
    glm::dvec3 finalDirection{0.0};
    double nullRelativeError = 0.0;
    double angularMomentumRelativeError = 0.0;
    int stepsTaken = 0;
};

GeodesicTraceResult TraceBlackHoleRayCPU(const glm::dvec3 &rayOriginMeters,
                                         const glm::dvec3 &rayDirNormalized,
                                         const BlackHoleSceneConfig &scene,
                                         Geodesic3DSolverKind solverKind);

glm::dvec3 PixelToCameraRayDirection(int px, int py, int width, int height,
                                     const BlackHoleSceneConfig &scene);

void PackBlackHoleUbos(const BlackHoleSceneConfig &scene, bool cameraMoving, int outW, int outH,
                       GpuCameraUboStd140 &outCam, GpuDiskUboStd140 &outDisk,
                       GpuObjectsUboStd140 &outObjects);

// ============================================================================
// Cancellable Multi-Threaded CPU Tile Worker (F4 & F5)
// Worker threads NEVER invoke OpenGL calls; results carry generation & revision.
// ============================================================================

struct CpuImageJobMeta
{
    std::uint64_t jobId = 0;
    std::uint64_t sceneGeneration = 0;
    std::uint64_t cameraRevision = 0;
    int width = 0;
    int height = 0;
};

struct CpuImageResult
{
    CpuImageJobMeta meta{};
    std::vector<std::uint8_t> rgbaPixels;
    double elapsedMs = 0.0;
    std::uint32_t workerThreadsUsed = 1;
    bool usedOpenMp = false;
};

using CpuTileRenderFn =
    std::function<void(int x0, int y0, int x1, int y1, int width, int height,
                       std::uint8_t *rgbaPixels, const std::atomic<bool> &cancelFlag)>;

class CpuRayWorker
{
  public:
    CpuRayWorker();
    ~CpuRayWorker();

    CpuRayWorker(const CpuRayWorker &) = delete;
    CpuRayWorker &operator=(const CpuRayWorker &) = delete;

    void Start(std::uint32_t desiredThreadCount = 0);
    void SubmitJob(const CpuImageJobMeta &meta, CpuTileRenderFn renderFn, int tileSize = 32);

    // Non-blocking cancellation signal (safe to call in BeginExit()).
    void RequestCancel();

    // True when no job is queued or executing (safe non-blocking poll for IsReadyToRelease()).
    bool IsIdle() const;
    bool IsBusy() const { return !IsIdle(); }
    float Progress() const;
    std::uint32_t ThreadCount() const { return workerThreadCount_; }
    static bool IsOpenMpCompiled();

    // Joins worker thread(s) and frees internal buffers (call in ReleaseResources()).
    void StopAndJoin();

    // Pops completed image ONLY if generation, cameraRevision, and dimensions match; discards stale.
    bool TryPopCompletedResult(CpuImageResult &outResult, std::uint64_t expectedSceneGeneration,
                               std::uint64_t expectedCameraRevision, int expectedWidth,
                               int expectedHeight);

  private:
    struct PendingJob
    {
        CpuImageJobMeta meta{};
        CpuTileRenderFn renderFn;
        int tileSize = 32;
    };

    void CoordinatorLoop();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread coordinatorThread_;
    bool running_ = false;
    bool hasPendingJob_ = false;
    bool jobExecuting_ = false;
    PendingJob pendingJob_{};

    std::atomic<bool> cancelFlag_{false};
    std::atomic<int> completedTiles_{0};
    std::atomic<int> totalTiles_{1};

    bool hasCompletedResult_ = false;
    CpuImageResult completedResult_{};
    std::uint32_t workerThreadCount_ = 1;
};

} // namespace grav_models

