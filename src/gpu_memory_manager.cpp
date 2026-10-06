#include "gpu_memory_manager.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

GpuBuffer::GpuBuffer(const GpuBufferDesc &desc) { Allocate(desc); }

GpuBuffer::~GpuBuffer() { Reset(); }

GpuBuffer::GpuBuffer(GpuBuffer &&other) noexcept
    : desc_(other.desc_), handle_(other.handle_), mappedPtr_(other.mappedPtr_),
      logicalCount_(other.logicalCount_)
{
    other.handle_ = 0;
    other.mappedPtr_ = nullptr;
    other.logicalCount_ = 0;
}

GpuBuffer &GpuBuffer::operator=(GpuBuffer &&other) noexcept
{
    if (this != &other)
    {
        Reset();
        desc_ = other.desc_;
        handle_ = other.handle_;
        mappedPtr_ = other.mappedPtr_;
        logicalCount_ = other.logicalCount_;
        other.handle_ = 0;
        other.mappedPtr_ = nullptr;
        other.logicalCount_ = 0;
    }
    return *this;
}

void GpuBuffer::Allocate(const GpuBufferDesc &desc)
{
    if (desc.elementSize == 0 || desc.capacity == 0)
    {
        throw std::invalid_argument(
            "GpuBuffer element size and capacity must be greater than zero");
    }
    if (desc.storageMode != GpuStorageMode::GpuResident)
    {
        throw std::invalid_argument("Unsupported GPU storage mode");
    }
    if (desc.capacity > std::numeric_limits<std::size_t>::max() / desc.elementSize ||
        desc.capacity * desc.elementSize >
            static_cast<std::size_t>(std::numeric_limits<GLsizeiptr>::max()))
    {
        throw std::length_error("GpuBuffer allocation size exceeds OpenGL limits");
    }

    Reset();
    desc_ = desc;
    glGenBuffers(1, &handle_);
    AllocateStorage();
}

void GpuBuffer::AllocateStorage()
{
    GLbitfield storageFlags = GL_DYNAMIC_STORAGE_BIT;
    GLbitfield mappingFlags = 0;
    if (desc_.mappingMode != MappingMode::None)
    {
        if (desc_.cpuAccess == CpuAccess::None)
        {
            throw std::invalid_argument("Persistent mapping requires CPU access");
        }
        if (desc_.cpuAccess == CpuAccess::Read || desc_.cpuAccess == CpuAccess::ReadWrite)
        {
            storageFlags |= GL_MAP_READ_BIT;
            mappingFlags |= GL_MAP_READ_BIT;
        }
        if (desc_.cpuAccess == CpuAccess::Write || desc_.cpuAccess == CpuAccess::ReadWrite)
        {
            storageFlags |= GL_MAP_WRITE_BIT;
            mappingFlags |= GL_MAP_WRITE_BIT;
        }
        storageFlags |= GL_MAP_PERSISTENT_BIT;
        mappingFlags |= GL_MAP_PERSISTENT_BIT;
        if (desc_.mappingMode == MappingMode::PersistentCoherent)
        {
            storageFlags |= GL_MAP_COHERENT_BIT;
            mappingFlags |= GL_MAP_COHERENT_BIT;
        }
    }

    glBindBuffer(desc_.target, handle_);
    glBufferStorage(desc_.target, static_cast<GLsizeiptr>(AllocatedBytes()), nullptr, storageFlags);
    if (desc_.mappingMode != MappingMode::None)
    {
        mappedPtr_ = glMapBufferRange(desc_.target, 0, static_cast<GLsizeiptr>(AllocatedBytes()),
                                      mappingFlags);
        if (mappedPtr_ == nullptr)
        {
            glBindBuffer(desc_.target, 0);
            throw std::runtime_error("Unable to establish persistent GPU buffer mapping");
        }
    }
    glBindBuffer(desc_.target, 0);
}

void GpuBuffer::Resize(std::size_t newCapacity)
{
    if (newCapacity == 0)
    {
        throw std::invalid_argument("GpuBuffer capacity must be greater than zero");
    }
    if (newCapacity == desc_.capacity)
    {
        return;
    }

    GpuBufferDesc replacementDesc = desc_;
    replacementDesc.capacity = newCapacity;
    GpuBuffer replacement(replacementDesc);
    replacement.logicalCount_ = logicalCount_ > newCapacity ? newCapacity : logicalCount_;
    const std::size_t preservedBytes = replacement.LogicalBytes();
    if (preservedBytes != 0)
    {
        glBindBuffer(GL_COPY_READ_BUFFER, handle_);
        glBindBuffer(GL_COPY_WRITE_BUFFER, replacement.handle_);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                            static_cast<GLsizeiptr>(preservedBytes));
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }
    *this = std::move(replacement);
}

void GpuBuffer::Upload(const void *data, std::size_t byteCount, std::size_t byteOffset)
{
    const std::size_t allocatedBytes = AllocatedBytes();
    if (byteOffset > allocatedBytes || byteCount > allocatedBytes - byteOffset)
    {
        throw std::out_of_range("GpuBuffer upload exceeds allocated capacity");
    }
    if (byteCount == 0)
    {
        return;
    }
    if (data == nullptr)
    {
        throw std::invalid_argument("GpuBuffer upload data must not be null");
    }

    const bool cpuCanWrite =
        desc_.cpuAccess == CpuAccess::Write || desc_.cpuAccess == CpuAccess::ReadWrite;
    if (mappedPtr_ != nullptr && cpuCanWrite)
    {
        std::memcpy(static_cast<std::byte *>(mappedPtr_) + byteOffset, data, byteCount);
        return;
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
        if (mappedPtr_ != nullptr)
        {
            glBindBuffer(desc_.target, handle_);
            glUnmapBuffer(desc_.target);
            glBindBuffer(desc_.target, 0);
            mappedPtr_ = nullptr;
        }
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

GpuSynchronization::~GpuSynchronization() { Reset(); }

void GpuSynchronization::WaitForCpuWrite()
{
    if (fence_ == nullptr)
    {
        return;
    }

    GLenum result = glClientWaitSync(fence_, 0, 0);
    if (result == GL_TIMEOUT_EXPIRED)
    {
        result = glClientWaitSync(fence_, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
    }
    if (result == GL_WAIT_FAILED)
    {
        throw std::runtime_error("GPU fence wait failed");
    }
    glDeleteSync(fence_);
    fence_ = nullptr;
}

void GpuSynchronization::FenceGpuCompletion()
{
    Reset();
    glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
    fence_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (fence_ == nullptr)
    {
        throw std::runtime_error("Unable to create GPU completion fence");
    }
}

void GpuSynchronization::Reset()
{
    if (fence_ != nullptr)
    {
        glDeleteSync(fence_);
        fence_ = nullptr;
    }
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

void GpuMemoryManager::Destroy(BufferRole role)
{
    if (buffers_.contains(role))
    {
        WaitForCpuWrite();
        buffers_.erase(role);
    }
}

void GpuMemoryManager::Resize(BufferRole role, std::size_t newCapacity)
{
    GpuBuffer &buffer = Get(role);
    if (buffer.Capacity() == newCapacity)
    {
        return;
    }
    WaitForCpuWrite();
    buffer.Resize(newCapacity);
    buffer.BindBase();
    ++reallocationCount_;
    UpdatePeakAllocation();
}

void GpuMemoryManager::Upload(BufferRole role, const void *data, std::size_t byteCount,
                              std::size_t logicalCount, std::size_t byteOffset)
{
    GpuBuffer &buffer = Get(role);
    if (logicalCount > buffer.Capacity())
    {
        throw std::out_of_range("GPU buffer logical count exceeds capacity");
    }
    buffer.Upload(data, byteCount, byteOffset);
    buffer.SetLogicalCount(logicalCount);
}

void GpuMemoryManager::Bind(BufferRole role) const { Get(role).BindBase(); }

void GpuMemoryManager::WaitForCpuWrite() { synchronization_.WaitForCpuWrite(); }

void GpuMemoryManager::FenceGpuCompletion() { synchronization_.FenceGpuCompletion(); }

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
    synchronization_.WaitForCpuWrite();
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

std::size_t GpuMemoryManager::MappedBufferCount() const
{
    std::size_t count = 0;
    for (const auto &[role, buffer] : buffers_)
    {
        (void)role;
        count += buffer.IsMapped() ? 1 : 0;
    }
    return count;
}

std::size_t GpuMemoryManager::PersistentMappedBytes() const
{
    std::size_t total = 0;
    for (const auto &[role, buffer] : buffers_)
    {
        (void)role;
        total += buffer.IsMapped() ? buffer.AllocatedBytes() : 0;
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
