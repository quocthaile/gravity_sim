#pragma once

#include "gpu_memory_manager.hpp"
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace grav_render
{

enum class ResourceCategory : std::uint8_t
{
    Ssbo = 0,
    Texture,
    Ubo,
    GeometryBuffer,
    VertexArray,
    Framebuffer,
    ShaderProgram,
};

const char *ResourceCategoryName(ResourceCategory category);

struct TrackedResource
{
    std::uint64_t registrationId = 0;
    ResourceCategory category = ResourceCategory::GeometryBuffer;
    GLuint glHandle = 0;
    std::size_t allocatedBytes = 0;
    std::size_t logicalBytes = 0;
    std::string ownerTag; // Prefix "shared:" indicates shared infrastructure; "scene:" indicates active module
    std::string label;
    std::uint64_t sceneGeneration = 0;
};

struct ResourceCensusSnapshot
{
    std::size_t ssboCount = 0;
    std::size_t textureCount = 0;
    std::size_t uboCount = 0;
    std::size_t geometryBufferCount = 0;
    std::size_t vertexArrayCount = 0;
    std::size_t framebufferCount = 0;
    std::size_t shaderProgramCount = 0;

    std::size_t totalLiveObjects = 0;
    std::size_t totalAllocatedBytes = 0;
    std::size_t totalLogicalBytes = 0;
    std::size_t sharedAllocatedBytes = 0;
    std::size_t sceneAllocatedBytes = 0;

    std::size_t appPeakAllocatedBytes = 0;
    std::size_t scenePeakAllocatedBytes = 0;
    std::size_t runPeakAllocatedBytes = 0;

    std::vector<TrackedResource> entries;
};

class ResourceCensus
{
  public:
    std::uint64_t Register(ResourceCategory category, GLuint glHandle, std::size_t allocatedBytes,
                           std::size_t logicalBytes, const std::string &ownerTag,
                           const std::string &label, std::uint64_t sceneGeneration = 0);

    void UpdateBytes(std::uint64_t registrationId, GLuint glHandle, std::size_t allocatedBytes,
                     std::size_t logicalBytes);

    void Unregister(std::uint64_t &registrationId);
    void UnregisterByOwnerPrefix(const std::string &ownerPrefix);

    void ResetScenePeak(const GpuMemoryManager *gpuManager = nullptr);
    void ResetRunPeak(const GpuMemoryManager *gpuManager = nullptr);

    ResourceCensusSnapshot Snapshot(const GpuMemoryManager *gpuManager = nullptr);

    std::size_t LiveSceneResourceCount() const;

  private:
    static bool IsSharedOwner(const std::string &ownerTag);
    std::size_t ComputeCurrentTotalBytes(const GpuMemoryManager *gpuManager) const;
    std::size_t ComputeCurrentSceneBytes(const GpuMemoryManager *gpuManager) const;
    void RefreshPeaks(const GpuMemoryManager *gpuManager);

    std::uint64_t nextId_ = 1;
    std::map<std::uint64_t, TrackedResource> tracked_;
    std::size_t appPeakAllocatedBytes_ = 0;
    std::size_t scenePeakAllocatedBytes_ = 0;
    std::size_t runPeakAllocatedBytes_ = 0;
};

} // namespace grav_render

