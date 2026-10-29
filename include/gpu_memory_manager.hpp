#pragma once

#include <GL/glew.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>

enum class BufferRole : std::uint8_t
{
    CurrentState,
    NextState,
    ObjectPhysical,
    ObjectControl,
    BaseGrid,
    DeformedGrid,
    MassField,
    BlurFieldA,
    BlurFieldB,
    AccelerationField,
    LODMetadata,
    Metrics,
    ObjectDerived,
    ObjectAcceleration,
    ObjectRender,
    ObjectCollision,
    // Shared grid deformation source list (pass-scoped SSBO binding 1; shares slot 1 with
    // NextState across distinct passes so maximum SSBO binding remains <= 15).
    GridSources,
};

enum class GpuStorageMode : std::uint8_t
{
    GpuResident,
};

enum class CpuAccess : std::uint8_t
{
    None,
    Read,
    Write,
    ReadWrite,
};

enum class MappingMode : std::uint8_t
{
    None,
    Persistent,
    PersistentCoherent,
};

enum class FencePollStatus : std::uint8_t
{
    NoFence,
    Pending,
    Signaled,
    Failed,
};

constexpr GLuint BindingFor(BufferRole role)
{
    switch (role)
    {
    case BufferRole::CurrentState:
        return 0;
    case BufferRole::NextState:
        return 1;
    case BufferRole::ObjectPhysical:
        return 2;
    case BufferRole::ObjectControl:
        return 3;
    case BufferRole::BaseGrid:
        return 4;
    case BufferRole::DeformedGrid:
        return 5;
    case BufferRole::MassField:
        return 6;
    case BufferRole::BlurFieldA:
        return 7;
    case BufferRole::BlurFieldB:
        return 8;
    case BufferRole::AccelerationField:
        return 9;
    case BufferRole::LODMetadata:
        return 10;
    case BufferRole::Metrics:
        return 11;
    case BufferRole::ObjectDerived:
        return 12;
    case BufferRole::ObjectAcceleration:
        return 13;
    case BufferRole::ObjectRender:
        return 14;
    case BufferRole::ObjectCollision:
        return 15;
    case BufferRole::GridSources:
        return 1;
    }
    return 0;
}

constexpr const char *BufferRoleName(BufferRole role)
{
    switch (role)
    {
    case BufferRole::CurrentState:
        return "CurrentState";
    case BufferRole::NextState:
        return "NextState";
    case BufferRole::ObjectPhysical:
        return "ObjectPhysical";
    case BufferRole::ObjectControl:
        return "ObjectControl";
    case BufferRole::BaseGrid:
        return "BaseGrid";
    case BufferRole::DeformedGrid:
        return "DeformedGrid";
    case BufferRole::MassField:
        return "MassField";
    case BufferRole::BlurFieldA:
        return "BlurFieldA";
    case BufferRole::BlurFieldB:
        return "BlurFieldB";
    case BufferRole::AccelerationField:
        return "AccelerationField";
    case BufferRole::LODMetadata:
        return "LODMetadata";
    case BufferRole::Metrics:
        return "Metrics";
    case BufferRole::ObjectDerived:
        return "ObjectDerived";
    case BufferRole::ObjectAcceleration:
        return "ObjectAcceleration";
    case BufferRole::ObjectRender:
        return "ObjectRender";
    case BufferRole::ObjectCollision:
        return "ObjectCollision";
    case BufferRole::GridSources:
        return "GridSources";
    }
    return "Unknown";
}

struct GpuBufferDescription
{
    BufferRole role;
    GLenum target = GL_SHADER_STORAGE_BUFFER;
    GpuStorageMode storageMode = GpuStorageMode::GpuResident;
    CpuAccess cpuAccess = CpuAccess::None;
    MappingMode mappingMode = MappingMode::None;
    std::size_t elementSize = 0;
    std::size_t capacity = 0;
};

class GpuBuffer
{
  public:
    GpuBuffer() = default;
    explicit GpuBuffer(const GpuBufferDescription &bufferDescription);
    ~GpuBuffer();

    GpuBuffer(const GpuBuffer &) = delete;
    GpuBuffer &operator=(const GpuBuffer &) = delete;
    GpuBuffer(GpuBuffer &&other) noexcept;
    GpuBuffer &operator=(GpuBuffer &&other) noexcept;

    void Allocate(const GpuBufferDescription &bufferDescription);
    void Resize(std::size_t newCapacity);
    void Upload(const void *data, std::size_t byteCount, std::size_t byteOffset = 0);
    void BindBase() const;
    void BindBase(GLuint binding) const;
    void Reset();

    template <typename T> T *MappedPtr() { return static_cast<T *>(mappedPtr_); }
    template <typename T> const T *MappedPtr() const { return static_cast<const T *>(mappedPtr_); }

    GLuint Handle() const { return handle_; }
    const GpuBufferDescription &Desc() const { return bufferDescription_; }
    std::size_t LogicalCount() const { return logicalCount_; }
    std::size_t Capacity() const { return bufferDescription_.capacity; }
    std::size_t LogicalBytes() const { return logicalCount_ * bufferDescription_.elementSize; }
    std::size_t AllocatedBytes() const
    {
        return bufferDescription_.capacity * bufferDescription_.elementSize;
    }
    bool IsMapped() const { return mappedPtr_ != nullptr; }
    void SetLogicalCount(std::size_t count);

  private:
    friend class GpuMemoryManager;
    void SetRole(BufferRole role) { bufferDescription_.role = role; }
    void AllocateStorage();
    GpuBufferDescription bufferDescription_{};
    GLuint handle_ = 0;
    void *mappedPtr_ = nullptr;
    std::size_t logicalCount_ = 0;
};

class GpuSynchronization
{
  public:
    ~GpuSynchronization();
    void WaitForGpuCompletion();
    void FenceGpuCompletion();
    void CreateFreshFence();
    FencePollStatus PollCompletion();
    bool HasActiveFence() const { return fence_ != nullptr; }
    void Reset();

  private:
    GLsync fence_ = nullptr;
};

class GpuMemoryManager
{
  public:
    GpuBuffer &Create(const GpuBufferDescription &bufferDescription, bool bindImmediately = true);
    void Destroy(BufferRole role);
    void DestroyRoles(const BufferRole *roles, std::size_t count, bool alreadyQuiescent = false);
    void DestroyRoles(std::initializer_list<BufferRole> roles, bool alreadyQuiescent = false);
    void Resize(BufferRole role, std::size_t newCapacity, bool rebindImmediately = true);
    void Upload(BufferRole role, const void *data, std::size_t byteCount, std::size_t logicalCount,
                std::size_t byteOffset = 0);
    void Bind(BufferRole role) const;
    void Bind(BufferRole role, GLuint explicitBinding) const;
    void SwapCurrentNext();
    void WaitForGpuCompletion();
    void FenceGpuCompletion();
    void CreateFreshFence();
    FencePollStatus PollGpuCompletion();
    GpuBuffer &Get(BufferRole role);
    const GpuBuffer &Get(BufferRole role) const;
    bool Contains(BufferRole role) const;
    void Shutdown();

    std::size_t ActiveBufferCount() const { return buffers_.size(); }
    std::size_t LogicalBytes() const;
    std::size_t AllocatedBytes() const;
    std::size_t PeakAllocatedBytes() const { return peakAllocatedBytes_; }
    std::size_t ScenePeakAllocatedBytes() const { return scenePeakAllocatedBytes_; }
    void ResetScenePeakAllocation();
    std::size_t AllocationCount() const { return allocationCount_; }
    std::size_t ReallocationCount() const { return reallocationCount_; }
    std::size_t MappedBufferCount() const;
    std::size_t PersistentMappedBytes() const;

    const std::map<BufferRole, GpuBuffer> &Buffers() const { return buffers_; }

  private:
    void UnbindIfCurrentlyBound(const GpuBuffer &buffer) const;
    void UpdatePeakAllocation();
    std::map<BufferRole, GpuBuffer> buffers_;
    GpuSynchronization synchronization_;
    std::size_t peakAllocatedBytes_ = 0;
    std::size_t scenePeakAllocatedBytes_ = 0;
    std::size_t allocationCount_ = 0;
    std::size_t reallocationCount_ = 0;
};
