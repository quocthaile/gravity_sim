#pragma once

#include "gpu_memory_manager.hpp"
#include "render/resource_census.hpp"
#include "render/shader_cache.hpp"
#include <GL/glew.h>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace grav_render
{

// Host/shader contract for BufferRole::GridSources (std430 binding = 1 in grid passes).
struct GridSource
{
    glm::vec4 positionRs; // xyz: position in scene world units; w: Schwarzschild radius rs (meters)
};

static_assert(sizeof(GridSource) == 16, "GridSource must be 16 bytes to match std430 vec4");
static_assert(offsetof(GridSource, positionRs) == 0, "GridSource::positionRs must start at offset 0");

struct GridConfig
{
    int divisions = 200;
    float gridExtentWorld = 30000.0f;
    float basePlaneOffsetWorld = -800.0f;
    float metersPerWorldUnit = 1000.0f;
    float deformationAmplitudeWorld = 2.0e11f;
    float clampDelta = 0.0001f;
    glm::vec4 lineColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.25f);

    // Exact compatibility preset matching baseline F1 Gravity GPU grid.comp formula:
    // distance_m = max(length(toObject) * 1000.0, objRs * 1.0001)
    // displacement -= (2.0e11 * objRs) / distance_m
    static GridConfig GravityCompatibility()
    {
        GridConfig cfg;
        cfg.divisions = 200;
        cfg.gridExtentWorld = 30000.0f;
        cfg.basePlaneOffsetWorld = -800.0f;
        cfg.metersPerWorldUnit = 1000.0f;
        cfg.deformationAmplitudeWorld = 2.0e11f;
        cfg.clampDelta = 0.0001f;
        cfg.lineColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.25f);
        return cfg;
    }

    // Shared Black Hole 3D world configuration (1 world unit = 1.0e9 m = 1 Gm).
    // Uses divisions = 200 so topology and GL handles (BaseGrid, DeformedGrid, gridVAO, gridEBO)
    // remain invariant across F1 <-> F2 <-> F3 <-> F5 <-> F6 transitions.
    static GridConfig BlackHoleWorldGm()
    {
        GridConfig cfg;
        cfg.divisions = 200;
        cfg.gridExtentWorld = 300.0f;           // 3.0e11 m (~23.6 rs for SagA*)
        cfg.basePlaneOffsetWorld = -18.0f;      // -1.8e10 m (below equatorial plane y = 0)
        cfg.metersPerWorldUnit = 1.0e9f;        // 1 Gm per world unit
        cfg.deformationAmplitudeWorld = 28.0f;  // visual funnel depth proportional to rs/r
        cfg.clampDelta = 0.05f;
        cfg.lineColor = glm::vec4(0.82f, 0.88f, 1.0f, 0.28f);
        return cfg;
    }
};

struct GridResourceHandles
{
    GLuint baseGridBuffer = 0;
    GLuint deformedGridBuffer = 0;
    GLuint gridSourcesBuffer = 0;
    GLuint gridVAO = 0;
    GLuint gridEBO = 0;
};

class SharedGridService
{
  public:
    SharedGridService() = default;
    ~SharedGridService() = default;

    SharedGridService(const SharedGridService &) = delete;
    SharedGridService &operator=(const SharedGridService &) = delete;

    void Initialize(GpuMemoryManager &gpuManager, ShaderCache &shaderCache, ResourceCensus &census,
                    const GridConfig &initialConfig = GridConfig::GravityCompatibility());

    // Applies configuration; reallocates topology ONLY if divisions changed.
    // Re-uploads BaseGrid in-place if extent or basePlaneOffsetWorld changed.
    void Configure(const GridConfig &config);

    // GPU->GPU path for F1: dispatches grid_sources.comp reading CurrentState (0) + ObjectDerived (12)
    // and writing GridSources (1). Marks grid deformation dirty with sourceRevisionTag.
    void MaterializeFromGpuObjects(std::size_t objectCount, std::uint64_t sourceRevisionTag);

    // CPU->GPU path for F2/F3/F5/F6 or flat grid (sources empty -> sourceCount = 0).
    // Skips upload and dirtying if revisionTag matches last uploaded CPU source revision and count matches.
    void SetCpuSources(const std::vector<GridSource> &sources, std::uint64_t sourceRevisionTag = 0);

    // Forces deformation recomputation on next UpdateIfDirty().
    void MarkDirty();

    // Dispatches grid.comp ONLY when visible == true and deformation is dirty (or forceWhenHidden == true).
    // Returns true if grid.comp was dispatched this call.
    bool UpdateIfDirty(bool visible, bool forceWhenHidden = false);

    // Draws the deformed grid lines if visible == true. Automatically runs UpdateIfDirty(true) first
    // so a previously hidden dirty grid is always up-to-date before drawing.
    void Draw(const glm::mat4 &view, const glm::mat4 &projection, bool visible,
              bool depthTest = true);

    void Shutdown();

    bool IsInitialized() const { return initialized_; }
    bool IsDirty() const { return deformationDirty_; }
    const GridConfig &Config() const { return config_; }
    std::size_t ActiveSourceCount() const { return activeSourceCount_; }
    std::size_t NodeCount() const { return gridNodeCount_; }
    std::size_t IndexCount() const { return gridIndexCount_; }
    std::uint64_t ComputeDispatchCount() const { return computeDispatchCount_; }
    std::uint64_t MaterializationDispatchCount() const { return materializationDispatchCount_; }
    GridResourceHandles Handles() const;

  private:
    void EnsurePrograms();
    void EnsureTopology(int divisions);
    void UploadBaseGridGeometry();
    void EnsureSourceCapacity(std::size_t requiredCount);

    GpuMemoryManager *gpuManager_ = nullptr;
    ShaderCache *shaderCache_ = nullptr;
    ResourceCensus *census_ = nullptr;

    bool initialized_ = false;
    GridConfig config_{};

    GLuint gridVAO_ = 0;
    GLuint gridEBO_ = 0;
    std::uint64_t vaoCensusId_ = 0;
    std::uint64_t eboCensusId_ = 0;
    std::size_t gridNodeCount_ = 0;
    std::size_t gridIndexCount_ = 0;

    GLuint renderProgram_ = 0;
    GLint locModel_ = -1;
    GLint locView_ = -1;
    GLint locProjection_ = -1;
    GLint locObjectColor_ = -1;

    GLuint deformProgram_ = 0;
    GLint locSourceCount_ = -1;
    GLint locGridWidth_ = -1;
    GLint locGridHeight_ = -1;
    GLint locMetersPerUnit_ = -1;
    GLint locAmplitudeWorld_ = -1;
    GLint locClampDelta_ = -1;

    GLuint materializeProgram_ = 0;
    GLint locMaterializeCount_ = -1;

    std::size_t activeSourceCount_ = 0;
    bool lastSourceWasCpu_ = false;
    std::uint64_t lastCpuSourceTag_ = 0;
    bool deformationDirty_ = true;

    std::uint64_t computeDispatchCount_ = 0;
    std::uint64_t materializationDispatchCount_ = 0;
};

} // namespace grav_render

