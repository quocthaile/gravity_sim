#pragma once

#include <GL/glew.h>

#include <cstddef>
#include <cstdint>
#include <map>

// Internal ABI for simulation resources shared by C++ and GLSL.
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
};

enum class GpuStorageMode : std::uint8_t
{
    GpuResident,
};

enum class GpuAccess : std::uint8_t
{
    ReadOnly,
    WriteOnly,
    ReadWrite,
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
    }
    return 0;
}

struct GpuBufferDesc
{
    BufferRole role;
    GLenum target = GL_SHADER_STORAGE_BUFFER;
    GLenum usage = GL_DYNAMIC_DRAW;
    GpuStorageMode storageMode = GpuStorageMode::GpuResident;
    GpuAccess access = GpuAccess::ReadWrite;
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

    GLuint Handle() const { return handle_; }
    const GpuBufferDesc &Desc() const { return desc_; }
    std::size_t LogicalCount() const { return logicalCount_; }
    std::size_t Capacity() const { return desc_.capacity; }
    std::size_t LogicalBytes() const { return logicalCount_ * desc_.elementSize; }
    std::size_t AllocatedBytes() const { return desc_.capacity * desc_.elementSize; }

    void SetLogicalCount(std::size_t count);

  private:
    void AllocateStorage();

    GpuBufferDesc desc_{};
    GLuint handle_ = 0;
    std::size_t logicalCount_ = 0;
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
    GpuBuffer &Get(BufferRole role);
    const GpuBuffer &Get(BufferRole role) const;
    bool Contains(BufferRole role) const;
    void Shutdown();

    std::size_t LogicalBytes() const;
    std::size_t AllocatedBytes() const;
    std::size_t PeakAllocatedBytes() const { return peakAllocatedBytes_; }
    std::size_t AllocationCount() const { return allocationCount_; }
    std::size_t ReallocationCount() const { return reallocationCount_; }

  private:
    void UpdatePeakAllocation();

    std::map<BufferRole, GpuBuffer> buffers_;
    std::size_t peakAllocatedBytes_ = 0;
    std::size_t allocationCount_ = 0;
    std::size_t reallocationCount_ = 0;
};
