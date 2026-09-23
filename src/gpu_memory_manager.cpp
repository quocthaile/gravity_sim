#include "gpu_memory_manager.hpp"

#include <stdexcept>
#include <utility>

GpuBuffer::GpuBuffer(const GpuBufferDesc &desc) { Allocate(desc); }

GpuBuffer::~GpuBuffer() { Reset(); }

GpuBuffer::GpuBuffer(GpuBuffer &&other) noexcept
    : desc_(other.desc_), handle_(other.handle_), logicalCount_(other.logicalCount_)
{
    other.handle_ = 0;
    other.logicalCount_ = 0;
}

GpuBuffer &GpuBuffer::operator=(GpuBuffer &&other) noexcept
{
    if (this != &other)
    {
        Reset();
        desc_ = other.desc_;
        handle_ = other.handle_;
        logicalCount_ = other.logicalCount_;
        other.handle_ = 0;
        other.logicalCount_ = 0;
    }
    return *this;
}

void GpuBuffer::Allocate(const GpuBufferDesc &desc)
{
    if (desc.elementSize == 0)
    {
        throw std::invalid_argument("GpuBuffer elementSize must be greater than zero");
    }
    if (desc.storageMode != GpuStorageMode::GpuResident)
    {
        throw std::invalid_argument("Unsupported GPU storage mode");
    }

    Reset();
    desc_ = desc;
    glGenBuffers(1, &handle_);
    AllocateStorage();
}

void GpuBuffer::AllocateStorage()
{
    glBindBuffer(desc_.target, handle_);
    glBufferData(desc_.target, static_cast<GLsizeiptr>(AllocatedBytes()), nullptr, desc_.usage);
    glBindBuffer(desc_.target, 0);
}

void GpuBuffer::Resize(std::size_t newCapacity)
{
    if (newCapacity == desc_.capacity)
    {
        return;
    }

    desc_.capacity = newCapacity;
    logicalCount_ = logicalCount_ > newCapacity ? newCapacity : logicalCount_;
    AllocateStorage();
}

void GpuBuffer::Upload(const void *data, std::size_t byteCount, std::size_t byteOffset)
{
    if (byteOffset + byteCount > AllocatedBytes())
    {
        throw std::out_of_range("GpuBuffer upload exceeds allocated capacity");
    }

    glBindBuffer(desc_.target, handle_);
    glBufferSubData(desc_.target, static_cast<GLintptr>(byteOffset),
                    static_cast<GLsizeiptr>(byteCount), data);
    glBindBuffer(desc_.target, 0);
}

void GpuBuffer::BindBase() const { BindBase(BindingFor(desc_.role)); }

void GpuBuffer::BindBase(GLuint binding) const { glBindBufferBase(desc_.target, binding, handle_); }

void GpuBuffer::Reset()
{
    if (handle_ != 0)
    {
        glDeleteBuffers(1, &handle_);
        handle_ = 0;
    }
    logicalCount_ = 0;
}

void GpuBuffer::SetLogicalCount(std::size_t count)
{
    if (count > desc_.capacity)
    {
        throw std::out_of_range("GpuBuffer logical count exceeds capacity");
    }
    logicalCount_ = count;
}

GpuBuffer &GpuMemoryManager::Create(const GpuBufferDesc &desc)
{
    if (buffers_.contains(desc.role))
    {
        throw std::invalid_argument("GPU buffer role already exists");
    }

    auto [it, inserted] = buffers_.emplace(desc.role, GpuBuffer(desc));
    if (!inserted)
    {
        throw std::runtime_error("Unable to create GPU buffer");
    }
    ++allocationCount_;
    UpdatePeakAllocation();
    it->second.BindBase();
    return it->second;
}

void GpuMemoryManager::Destroy(BufferRole role) { buffers_.erase(role); }

void GpuMemoryManager::Resize(BufferRole role, std::size_t newCapacity)
{
    GpuBuffer &buffer = Get(role);
    if (buffer.Capacity() == newCapacity)
    {
        return;
    }
    buffer.Resize(newCapacity);
    ++reallocationCount_;
    UpdatePeakAllocation();
}

void GpuMemoryManager::Upload(BufferRole role, const void *data, std::size_t byteCount,
                              std::size_t logicalCount, std::size_t byteOffset)
{
    GpuBuffer &buffer = Get(role);
    buffer.Upload(data, byteCount, byteOffset);
    buffer.SetLogicalCount(logicalCount);
}

void GpuMemoryManager::Bind(BufferRole role) const { Get(role).BindBase(); }

GpuBuffer &GpuMemoryManager::Get(BufferRole role)
{
    auto it = buffers_.find(role);
    if (it == buffers_.end())
    {
        throw std::out_of_range("GPU buffer role does not exist");
    }
    return it->second;
}

const GpuBuffer &GpuMemoryManager::Get(BufferRole role) const
{
    auto it = buffers_.find(role);
    if (it == buffers_.end())
    {
        throw std::out_of_range("GPU buffer role does not exist");
    }
    return it->second;
}

bool GpuMemoryManager::Contains(BufferRole role) const { return buffers_.contains(role); }

void GpuMemoryManager::Shutdown()
{
    buffers_.clear();
    peakAllocatedBytes_ = 0;
    allocationCount_ = 0;
    reallocationCount_ = 0;
}

std::size_t GpuMemoryManager::LogicalBytes() const
{
    std::size_t total = 0;
    for (const auto &[role, buffer] : buffers_)
    {
        (void)role;
        total += buffer.LogicalBytes();
    }
    return total;
}

std::size_t GpuMemoryManager::AllocatedBytes() const
{
    std::size_t total = 0;
    for (const auto &[role, buffer] : buffers_)
    {
        (void)role;
        total += buffer.AllocatedBytes();
    }
    return total;
}

void GpuMemoryManager::UpdatePeakAllocation()
{
    if (AllocatedBytes() > peakAllocatedBytes_)
    {
        peakAllocatedBytes_ = AllocatedBytes();
    }
}
