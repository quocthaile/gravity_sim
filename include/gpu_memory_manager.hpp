#pragma once

#include <GL/glew.h>

#include <cstddef>
#include <cstdint>
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
    }
    return 0;
}

struct GpuBufferDesc
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
    explicit GpuBuffer(const GpuBufferDesc &desc);
    ~GpuBuffer();

    GpuBuffer(const GpuBuffer &) = delete;
    GpuBuffer &operator=(const GpuBuffer &) = delete;
    GpuBuffer(GpuBuffer &&other) noexcept;
    GpuBuffer &operator=(GpuBuffer &&other) noexcept;

    void Allocate(const GpuBufferDesc &desc);
    void Resize(std::size_t newCapacity);
    void Upload(const void *data, std::size_t byteCount, std::size_t byteOffset = 0);
    void BindBase() const;
    void BindBase(GLuint binding) const;
    void Reset();

    template <typename T> T *MappedPtr() { return static_cast<T *>(mappedPtr_); }
    template <typename T> const T *MappedPtr() const { return static_cast<const T *>(mappedPtr_); }

    GLuint Handle() const { return handle_; }
    const GpuBufferDesc &Desc() const { return desc_; }
    std::size_t LogicalCount() const { return logicalCount_; }
    std::size_t Capacity() const { return desc_.capacity; }
    std::size_t LogicalBytes() const { return logicalCount_ * desc_.elementSize; }
    std::size_t AllocatedBytes() const { return desc_.capacity * desc_.elementSize; }
    bool IsMapped() const { return mappedPtr_ != nullptr; }

    void SetLogicalCount(std::size_t count);

  private:
    void AllocateStorage();

    GpuBufferDesc desc_{};
    GLuint handle_ = 0;
    void *mappedPtr_ = nullptr;
    std::size_t logicalCount_ = 0;
};

class GpuSynchronization
{
  public:
    ~GpuSynchronization();

    void WaitForCpuWrite();
    void FenceGpuCompletion();
    void Reset();

  private:
    GLsync fence_ = nullptr;
};

class GpuMemoryManager
{
  public:
    GpuBuffer &Create(const GpuBufferDesc &desc);
    void Destroy(BufferRole role);
    void Resize(BufferRole role, std::size_t newCapacity);
    void Upload(BufferRole role, const void *data, std::size_t byteCount, std::size_t logicalCount,
                std::size_t byteOffset = 0);
    void Bind(BufferRole role) const;
    void WaitForCpuWrite();
    void FenceGpuCompletion();
    GpuBuffer &Get(BufferRole role);
    const GpuBuffer &Get(BufferRole role) const;
    bool Contains(BufferRole role) const;
    void Shutdown();

    std::size_t LogicalBytes() const;
    std::size_t AllocatedBytes() const;
    std::size_t PeakAllocatedBytes() const { return peakAllocatedBytes_; }
    std::size_t AllocationCount() const { return allocationCount_; }
    std::size_t ReallocationCount() const { return reallocationCount_; }
    std::size_t MappedBufferCount() const;
    std::size_t PersistentMappedBytes() const;

  private:
    void UpdatePeakAllocation();

    std::map<BufferRole, GpuBuffer> buffers_;
    GpuSynchronization synchronization_;
    std::size_t peakAllocatedBytes_ = 0;
    std::size_t allocationCount_ = 0;
    std::size_t reallocationCount_ = 0;
};
