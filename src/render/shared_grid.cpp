#include "render/shared_grid.hpp"

#include "gravity_sim_3Dgrid_function.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>

namespace grav_render
{
namespace
{
constexpr GLuint kGridWorkGroupX = 16;
constexpr GLuint kGridWorkGroupY = 16;
constexpr GLuint kMaterializeWorkGroupX = 64;

bool FloatDiffers(float a, float b) { return std::fabs(a - b) > 1e-6f; }
} // namespace

void SharedGridService::Initialize(GpuMemoryManager &gpuManager, ShaderCache &shaderCache,
                                   ResourceCensus &census, const GridConfig &initialConfig)
{
    gpuManager_ = &gpuManager;
    shaderCache_ = &shaderCache;
    census_ = &census;

    EnsurePrograms();
    config_ = initialConfig;
    EnsureTopology(config_.divisions);
    UploadBaseGridGeometry();
    EnsureSourceCapacity(16);

    activeSourceCount_ = 0;
    lastSourceWasCpu_ = false;
    lastCpuSourceTag_ = 0;
    deformationDirty_ = true;
    initialized_ = true;
}

void SharedGridService::EnsurePrograms()
{
    renderProgram_ = shaderCache_->GetOrCreateGraphics(
        "shared:grid_render", "shaders/grid.vert", "shaders/grid.frag", "shared:grid");
    locModel_ = glGetUniformLocation(renderProgram_, "model");
    locView_ = glGetUniformLocation(renderProgram_, "view");
    locProjection_ = glGetUniformLocation(renderProgram_, "projection");
    locObjectColor_ = glGetUniformLocation(renderProgram_, "objectColor");

    deformProgram_ = shaderCache_->GetOrCreateCompute("shared:grid_compute", "shaders/grid.comp",
                                                      "shared:grid");
    locSourceCount_ = glGetUniformLocation(deformProgram_, "u_sourceCount");
    locGridWidth_ = glGetUniformLocation(deformProgram_, "u_gridWidth");
    locGridHeight_ = glGetUniformLocation(deformProgram_, "u_gridHeight");
    locMetersPerUnit_ = glGetUniformLocation(deformProgram_, "u_metersPerWorldUnit");
    locAmplitudeWorld_ = glGetUniformLocation(deformProgram_, "u_deformationAmplitudeWorld");
    locClampDelta_ = glGetUniformLocation(deformProgram_, "u_clampDelta");

    materializeProgram_ = shaderCache_->GetOrCreateCompute(
        "shared:grid_sources", "shaders/grid_sources.comp", "shared:grid");
    locMaterializeCount_ = glGetUniformLocation(materializeProgram_, "u_objectCount");
}

void SharedGridService::EnsureTopology(int divisions)
{
    if (divisions <= 0)
    {
        throw std::invalid_argument("Grid divisions must be positive");
    }

    const std::size_t sideNodes = static_cast<std::size_t>(divisions + 1);
    const std::size_t requiredNodes = sideNodes * sideNodes;

    if (gridVAO_ != 0 && gridNodeCount_ == requiredNodes &&
        gpuManager_->Contains(BufferRole::BaseGrid) &&
        gpuManager_->Contains(BufferRole::DeformedGrid))
    {
        return;
    }

    if (gridVAO_ != 0)
    {
        if (census_ != nullptr)
        {
            census_->Unregister(vaoCensusId_);
            census_->Unregister(eboCensusId_);
        }
        glDeleteVertexArrays(1, &gridVAO_);
        glDeleteBuffers(1, &gridEBO_);
        gridVAO_ = 0;
        gridEBO_ = 0;
    }

    if (gpuManager_->Contains(BufferRole::BaseGrid))
    {
        gpuManager_->Resize(BufferRole::BaseGrid, requiredNodes, true);
    }
    else
    {
        gpuManager_->Create({BufferRole::BaseGrid, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(glm::vec4), requiredNodes});
    }
    gpuManager_->Get(BufferRole::BaseGrid).SetLogicalCount(requiredNodes);

    if (gpuManager_->Contains(BufferRole::DeformedGrid))
    {
        gpuManager_->Resize(BufferRole::DeformedGrid, requiredNodes, true);
    }
    else
    {
        gpuManager_->Create({BufferRole::DeformedGrid, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(glm::vec4), requiredNodes});
    }
    gpuManager_->Get(BufferRole::DeformedGrid).SetLogicalCount(requiredNodes);

    const std::vector<unsigned int> indices = CreateGridIndices(divisions);
    gridNodeCount_ = requiredNodes;
    gridIndexCount_ = indices.size();

    glGenVertexArrays(1, &gridVAO_);
    glGenBuffers(1, &gridEBO_);

    glBindVertexArray(gridVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, gpuManager_->Get(BufferRole::DeformedGrid).Handle());
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO_);
    const std::size_t indexBytes = indices.size() * sizeof(unsigned int);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indexBytes), indices.data(),
                 GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (census_ != nullptr)
    {
        vaoCensusId_ = census_->Register(ResourceCategory::VertexArray, gridVAO_, 0, 0,
                                         "shared:grid", "SharedGridVAO");
        eboCensusId_ = census_->Register(ResourceCategory::GeometryBuffer, gridEBO_, indexBytes,
                                         indexBytes, "shared:grid", "SharedGridEBO");
    }
}

void SharedGridService::UploadBaseGridGeometry()
{
    const int divisions = config_.divisions;
    const int side = divisions + 1;
    std::vector<glm::vec4> vertices;
    vertices.reserve(static_cast<std::size_t>(side) * static_cast<std::size_t>(side));

    const float step = config_.gridExtentWorld / static_cast<float>(divisions);
    const float halfSize = config_.gridExtentWorld * 0.5f;
    const float baseY = config_.basePlaneOffsetWorld;

    for (int zStep = 0; zStep <= divisions; ++zStep)
    {
        const float z = -halfSize + static_cast<float>(zStep) * step;
        for (int xStep = 0; xStep <= divisions; ++xStep)
        {
            const float x = -halfSize + static_cast<float>(xStep) * step;
            vertices.emplace_back(x, baseY, z, 1.0f);
        }
    }

    gpuManager_->Upload(BufferRole::BaseGrid, vertices.data(),
                        vertices.size() * sizeof(glm::vec4), vertices.size());
}

void SharedGridService::EnsureSourceCapacity(std::size_t requiredCount)
{
    const std::size_t targetMin = std::max<std::size_t>(requiredCount, 16);
    if (!gpuManager_->Contains(BufferRole::GridSources))
    {
        std::size_t cap = 16;
        while (cap < targetMin)
        {
            cap *= 2;
        }
        // bindImmediately = false so creating GridSources never clobbers NextState at binding 1
        // outside of explicit grid passes.
        gpuManager_->Create({BufferRole::GridSources, GL_SHADER_STORAGE_BUFFER,
                             GpuStorageMode::GpuResident, CpuAccess::None, MappingMode::None,
                             sizeof(GridSource), cap},
                            false);
        return;
    }

    const std::size_t currentCap = gpuManager_->Get(BufferRole::GridSources).Capacity();
    if (currentCap < targetMin)
    {
        std::size_t newCap = currentCap == 0 ? 16 : currentCap;
        while (newCap < targetMin)
        {
            newCap *= 2;
        }
        gpuManager_->Resize(BufferRole::GridSources, newCap, false);
    }
}

void SharedGridService::Configure(const GridConfig &config)
{
    if (!initialized_)
    {
        throw std::logic_error("SharedGridService::Configure called before Initialize");
    }

    const bool divisionsChanged = (config.divisions != config_.divisions);
    const bool baseGeomChanged =
        divisionsChanged || FloatDiffers(config.gridExtentWorld, config_.gridExtentWorld) ||
        FloatDiffers(config.basePlaneOffsetWorld, config_.basePlaneOffsetWorld);
    const bool deformFormulaChanged =
        FloatDiffers(config.metersPerWorldUnit, config_.metersPerWorldUnit) ||
        FloatDiffers(config.deformationAmplitudeWorld, config_.deformationAmplitudeWorld) ||
        FloatDiffers(config.clampDelta, config_.clampDelta);

    config_ = config;

    if (divisionsChanged)
    {
        EnsureTopology(config_.divisions);
    }
    if (baseGeomChanged)
    {
        UploadBaseGridGeometry();
    }
    if (baseGeomChanged || deformFormulaChanged)
    {
        deformationDirty_ = true;
    }
}

void SharedGridService::MaterializeFromGpuObjects(std::size_t objectCount,
                                                  std::uint64_t sourceRevisionTag)
{
    (void)sourceRevisionTag;
    if (!initialized_)
    {
        return;
    }

    EnsureSourceCapacity(objectCount);
    gpuManager_->Get(BufferRole::GridSources).SetLogicalCount(objectCount);
    activeSourceCount_ = objectCount;
    lastSourceWasCpu_ = false;
    lastCpuSourceTag_ = 0;

    if (objectCount > 0)
    {
        if (!gpuManager_->Contains(BufferRole::CurrentState) ||
            !gpuManager_->Contains(BufferRole::ObjectDerived))
        {
            throw std::runtime_error(
                "MaterializeFromGpuObjects requires CurrentState and ObjectDerived buffers");
        }

        gpuManager_->Bind(BufferRole::CurrentState, 0);
        gpuManager_->Bind(BufferRole::ObjectDerived, 12);
        gpuManager_->Bind(BufferRole::GridSources, 1);

        glUseProgram(materializeProgram_);
        glUniform1ui(locMaterializeCount_, static_cast<GLuint>(objectCount));
        const GLuint groupsX = static_cast<GLuint>(
            (objectCount + kMaterializeWorkGroupX - 1) / kMaterializeWorkGroupX);
        glDispatchCompute(groupsX, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        ++materializationDispatchCount_;
    }

    deformationDirty_ = true;
}

void SharedGridService::SetCpuSources(const std::vector<GridSource> &sources,
                                      std::uint64_t sourceRevisionTag)
{
    if (!initialized_)
    {
        return;
    }

    if (lastSourceWasCpu_ && sourceRevisionTag != 0 && sourceRevisionTag == lastCpuSourceTag_ &&
        sources.size() == activeSourceCount_)
    {
        return;
    }

    EnsureSourceCapacity(sources.size());
    if (!sources.empty())
    {
        gpuManager_->Upload(BufferRole::GridSources, sources.data(),
                            sources.size() * sizeof(GridSource), sources.size());
    }
    else
    {
        gpuManager_->Get(BufferRole::GridSources).SetLogicalCount(0);
    }

    activeSourceCount_ = sources.size();
    lastSourceWasCpu_ = true;
    lastCpuSourceTag_ = sourceRevisionTag;
    deformationDirty_ = true;
}

void SharedGridService::MarkDirty() { deformationDirty_ = true; }

bool SharedGridService::UpdateIfDirty(bool visible, bool forceWhenHidden)
{
    if (!initialized_ || !deformationDirty_)
    {
        return false;
    }
    if (!visible && !forceWhenHidden)
    {
        return false;
    }

    gpuManager_->Bind(BufferRole::GridSources, 1);
    gpuManager_->Bind(BufferRole::BaseGrid, 4);
    gpuManager_->Bind(BufferRole::DeformedGrid, 5);

    const GLuint gridWidth = static_cast<GLuint>(config_.divisions + 1);
    const GLuint gridHeight = static_cast<GLuint>(config_.divisions + 1);

    glUseProgram(deformProgram_);
    glUniform1ui(locSourceCount_, static_cast<GLuint>(activeSourceCount_));
    glUniform1ui(locGridWidth_, gridWidth);
    glUniform1ui(locGridHeight_, gridHeight);
    glUniform1f(locMetersPerUnit_, config_.metersPerWorldUnit);
    glUniform1f(locAmplitudeWorld_, config_.deformationAmplitudeWorld);
    glUniform1f(locClampDelta_, config_.clampDelta);

    const GLuint groupX = (gridWidth + kGridWorkGroupX - 1) / kGridWorkGroupX;
    const GLuint groupY = (gridHeight + kGridWorkGroupY - 1) / kGridWorkGroupY;
    glDispatchCompute(groupX, groupY, 1);
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    deformationDirty_ = false;
    ++computeDispatchCount_;
    return true;
}

void SharedGridService::Draw(const glm::mat4 &view, const glm::mat4 &projection, bool visible,
                             bool depthTest)
{
    if (!initialized_ || !visible || gridVAO_ == 0 || gridIndexCount_ == 0)
    {
        return;
    }

    UpdateIfDirty(true);

    const GLboolean prevDepth = glIsEnabled(GL_DEPTH_TEST);
    const GLboolean prevBlend = glIsEnabled(GL_BLEND);
    if (depthTest)
    {
        glEnable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(renderProgram_);
    const glm::mat4 model(1.0f);
    if (locModel_ >= 0)
    {
        glUniformMatrix4fv(locModel_, 1, GL_FALSE, glm::value_ptr(model));
    }
    if (locView_ >= 0)
    {
        glUniformMatrix4fv(locView_, 1, GL_FALSE, glm::value_ptr(view));
    }
    if (locProjection_ >= 0)
    {
        glUniformMatrix4fv(locProjection_, 1, GL_FALSE, glm::value_ptr(projection));
    }
    if (locObjectColor_ >= 0)
    {
        glUniform4fv(locObjectColor_, 1, glm::value_ptr(config_.lineColor));
    }

    glBindVertexArray(gridVAO_);
    glDrawElements(GL_LINES, static_cast<GLsizei>(gridIndexCount_), GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    if (prevDepth == GL_TRUE)
    {
        glEnable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }
    if (prevBlend == GL_FALSE)
    {
        glDisable(GL_BLEND);
    }
}

GridResourceHandles SharedGridService::Handles() const
{
    GridResourceHandles h;
    if (gpuManager_ != nullptr)
    {
        if (gpuManager_->Contains(BufferRole::BaseGrid))
        {
            h.baseGridBuffer = gpuManager_->Get(BufferRole::BaseGrid).Handle();
        }
        if (gpuManager_->Contains(BufferRole::DeformedGrid))
        {
            h.deformedGridBuffer = gpuManager_->Get(BufferRole::DeformedGrid).Handle();
        }
        if (gpuManager_->Contains(BufferRole::GridSources))
        {
            h.gridSourcesBuffer = gpuManager_->Get(BufferRole::GridSources).Handle();
        }
    }
    h.gridVAO = gridVAO_;
    h.gridEBO = gridEBO_;
    return h;
}

void SharedGridService::Shutdown()
{
    if (!initialized_)
    {
        return;
    }

    if (census_ != nullptr)
    {
        census_->Unregister(vaoCensusId_);
        census_->Unregister(eboCensusId_);
    }
    if (gridVAO_ != 0)
    {
        glDeleteVertexArrays(1, &gridVAO_);
        gridVAO_ = 0;
    }
    if (gridEBO_ != 0)
    {
        glDeleteBuffers(1, &gridEBO_);
        gridEBO_ = 0;
    }
    if (gpuManager_ != nullptr)
    {
        gpuManager_->DestroyRoles(
            {BufferRole::BaseGrid, BufferRole::DeformedGrid, BufferRole::GridSources}, false);
    }

    gridNodeCount_ = 0;
    gridIndexCount_ = 0;
    activeSourceCount_ = 0;
    initialized_ = false;
}

} // namespace grav_render
