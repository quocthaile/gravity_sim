#include "render/resource_census.hpp"

#include <algorithm>

namespace grav_render
{

const char *ResourceCategoryName(ResourceCategory category)
{
    switch (category)
    {
    case ResourceCategory::Ssbo:
        return "SSBO";
    case ResourceCategory::Texture:
        return "Texture";
    case ResourceCategory::Ubo:
        return "UBO";
    case ResourceCategory::GeometryBuffer:
        return "VBO/EBO";
    case ResourceCategory::VertexArray:
        return "VAO";
    case ResourceCategory::Framebuffer:
        return "FBO";
    case ResourceCategory::ShaderProgram:
        return "ShaderProgram";
    }
    return "Unknown";
}

bool ResourceCensus::IsSharedOwner(const std::string &ownerTag)
{
    return ownerTag.rfind("shared:", 0) == 0;
}

std::uint64_t ResourceCensus::Register(ResourceCategory category, GLuint glHandle,
                                       std::size_t allocatedBytes, std::size_t logicalBytes,
                                       const std::string &ownerTag, const std::string &label,
                                       std::uint64_t sceneGeneration)
{
    const std::uint64_t id = nextId_++;
    TrackedResource entry;
    entry.registrationId = id;
    entry.category = category;
    entry.glHandle = glHandle;
    entry.allocatedBytes = allocatedBytes;
    entry.logicalBytes = logicalBytes;
    entry.ownerTag = ownerTag;
    entry.label = label;
    entry.sceneGeneration = sceneGeneration;
    tracked_.emplace(id, std::move(entry));
    RefreshPeaks(nullptr);
    return id;
}

void ResourceCensus::UpdateBytes(std::uint64_t registrationId, GLuint glHandle,
                                 std::size_t allocatedBytes, std::size_t logicalBytes)
{
    if (registrationId == 0)
    {
        return;
    }
    const auto it = tracked_.find(registrationId);
    if (it == tracked_.end())
    {
        return;
    }
    it->second.glHandle = glHandle;
    it->second.allocatedBytes = allocatedBytes;
    it->second.logicalBytes = logicalBytes;
    RefreshPeaks(nullptr);
}

void ResourceCensus::Unregister(std::uint64_t &registrationId)
{
    if (registrationId == 0)
    {
        return;
    }
    tracked_.erase(registrationId);
    registrationId = 0;
}

void ResourceCensus::UnregisterByOwnerPrefix(const std::string &ownerPrefix)
{
    for (auto it = tracked_.begin(); it != tracked_.end();)
    {
        if (it->second.ownerTag.rfind(ownerPrefix, 0) == 0)
        {
            it = tracked_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

std::size_t ResourceCensus::ComputeCurrentTotalBytes(const GpuMemoryManager *gpuManager) const
{
    std::size_t total = 0;
    for (const auto &[_, res] : tracked_)
    {
        total += res.allocatedBytes;
    }
    if (gpuManager != nullptr)
    {
        total += gpuManager->AllocatedBytes();
    }
    return total;
}

std::size_t ResourceCensus::ComputeCurrentSceneBytes(const GpuMemoryManager *gpuManager) const
{
    std::size_t sceneBytes = 0;
    for (const auto &[_, res] : tracked_)
    {
        if (!IsSharedOwner(res.ownerTag))
        {
            sceneBytes += res.allocatedBytes;
        }
    }
    if (gpuManager != nullptr)
    {
        for (const auto &[role, buffer] : gpuManager->Buffers())
        {
            const bool isSharedGridRole = (role == BufferRole::BaseGrid ||
                                           role == BufferRole::DeformedGrid ||
                                           role == BufferRole::GridSources);
            if (!isSharedGridRole)
            {
                sceneBytes += buffer.AllocatedBytes();
            }
        }
    }
    return sceneBytes;
}

void ResourceCensus::RefreshPeaks(const GpuMemoryManager *gpuManager)
{
    const std::size_t total = ComputeCurrentTotalBytes(gpuManager);
    const std::size_t scene = ComputeCurrentSceneBytes(gpuManager);
    appPeakAllocatedBytes_ = std::max(appPeakAllocatedBytes_, total);
    scenePeakAllocatedBytes_ = std::max(scenePeakAllocatedBytes_, scene);
    runPeakAllocatedBytes_ = std::max(runPeakAllocatedBytes_, total);
}

void ResourceCensus::ResetScenePeak(const GpuMemoryManager *gpuManager)
{
    scenePeakAllocatedBytes_ = ComputeCurrentSceneBytes(gpuManager);
    RefreshPeaks(gpuManager);
}

void ResourceCensus::ResetRunPeak(const GpuMemoryManager *gpuManager)
{
    runPeakAllocatedBytes_ = ComputeCurrentTotalBytes(gpuManager);
    RefreshPeaks(gpuManager);
}

std::size_t ResourceCensus::LiveSceneResourceCount() const
{
    std::size_t count = 0;
    for (const auto &[_, res] : tracked_)
    {
        if (!IsSharedOwner(res.ownerTag))
        {
            ++count;
        }
    }
    return count;
}

ResourceCensusSnapshot ResourceCensus::Snapshot(const GpuMemoryManager *gpuManager)
{
    RefreshPeaks(gpuManager);

    ResourceCensusSnapshot snap;
    snap.entries.reserve(tracked_.size() + (gpuManager ? gpuManager->ActiveBufferCount() : 0));

    if (gpuManager != nullptr)
    {
        for (const auto &[role, buffer] : gpuManager->Buffers())
        {
            const bool isSharedGridRole = (role == BufferRole::BaseGrid ||
                                           role == BufferRole::DeformedGrid ||
                                           role == BufferRole::GridSources);
            TrackedResource ssboEntry;
            ssboEntry.registrationId = 0;
            ssboEntry.category = ResourceCategory::Ssbo;
            ssboEntry.glHandle = buffer.Handle();
            ssboEntry.allocatedBytes = buffer.AllocatedBytes();
            ssboEntry.logicalBytes = buffer.LogicalBytes();
            ssboEntry.ownerTag = isSharedGridRole ? "shared:grid" : "scene:gravity_gpu";
            ssboEntry.label = BufferRoleName(role);
            ssboEntry.sceneGeneration = 0;

            ++snap.ssboCount;
            snap.totalAllocatedBytes += ssboEntry.allocatedBytes;
            snap.totalLogicalBytes += ssboEntry.logicalBytes;
            if (isSharedGridRole)
            {
                snap.sharedAllocatedBytes += ssboEntry.allocatedBytes;
            }
            else
            {
                snap.sceneAllocatedBytes += ssboEntry.allocatedBytes;
            }
            snap.entries.push_back(std::move(ssboEntry));
        }
    }

    for (const auto &[_, res] : tracked_)
    {
        switch (res.category)
        {
        case ResourceCategory::Ssbo:
            ++snap.ssboCount;
            break;
        case ResourceCategory::Texture:
            ++snap.textureCount;
            break;
        case ResourceCategory::Ubo:
            ++snap.uboCount;
            break;
        case ResourceCategory::GeometryBuffer:
            ++snap.geometryBufferCount;
            break;
        case ResourceCategory::VertexArray:
            ++snap.vertexArrayCount;
            break;
        case ResourceCategory::Framebuffer:
            ++snap.framebufferCount;
            break;
        case ResourceCategory::ShaderProgram:
            ++snap.shaderProgramCount;
            break;
        }
        snap.totalAllocatedBytes += res.allocatedBytes;
        snap.totalLogicalBytes += res.logicalBytes;
        if (IsSharedOwner(res.ownerTag))
        {
            snap.sharedAllocatedBytes += res.allocatedBytes;
        }
        else
        {
            snap.sceneAllocatedBytes += res.allocatedBytes;
        }
        snap.entries.push_back(res);
    }

    snap.totalLiveObjects = snap.entries.size();
    snap.appPeakAllocatedBytes = std::max(appPeakAllocatedBytes_, snap.totalAllocatedBytes);
    snap.scenePeakAllocatedBytes = std::max(scenePeakAllocatedBytes_, snap.sceneAllocatedBytes);
    snap.runPeakAllocatedBytes = std::max(runPeakAllocatedBytes_, snap.totalAllocatedBytes);
    return snap;
}

} // namespace grav_render

