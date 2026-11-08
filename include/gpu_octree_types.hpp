#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <glm/glm.hpp>
#include <stdexcept>

constexpr std::uint32_t kOctreeBranchFactor = 8;
constexpr std::uint32_t kOctreeMaxDepth = 16;
constexpr std::uint32_t kOctreeLeafCapacity = 1;
constexpr std::uint32_t kInvalidOctreeIndex = 0xFFFFFFFFu;

constexpr std::uint32_t kOctreeNodeFlagOccupied = 1u << 0;
constexpr std::uint32_t kOctreeNodeFlagLeaf = 1u << 1;

constexpr std::size_t CalculateOctreeNodeBufferCapacity(std::size_t objectCapacity)
{
    const std::size_t maximumCapacity =
        (std::numeric_limits<std::uint32_t>::max() - 1ull) / kOctreeMaxDepth;
    if (objectCapacity == 0 || objectCapacity > maximumCapacity)
    {
        throw std::length_error("Octree node capacity exceeds the metadata/index range");
    }
    return 1 + objectCapacity * kOctreeMaxDepth;
}

enum class GpuOctreeBuildStatus : std::uint32_t
{
    Uninitialized,
    Building,
    Empty,
    Valid,
    NodeCapacityExceeded,
    InvalidBounds,
};

struct alignas(16) GpuMassAggregate
{
    glm::vec4 centerMass{};
};

struct alignas(16) GpuOctreeNode
{
    glm::uvec4 links{};
    glm::uvec4 range{};
    glm::vec4 boundsMinDiameter{};
    glm::vec4 boundsMax{};
};

struct alignas(16) GpuOctreeMetadata
{
    glm::uvec4 counts{};
    glm::uvec4 configuration{};
    glm::vec4 rootBoundsMinDiameter{};
    glm::vec4 rootBoundsMax{};
};

using GpuObjectIndex = std::uint32_t;

inline GpuOctreeMetadata MakeUninitializedOctreeMetadata(std::size_t nodeCapacity)
{
    if (nodeCapacity > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error("Octree node capacity does not fit the metadata layout");
    }
    return {glm::uvec4(0u, 0u, 0u, kInvalidOctreeIndex),
            glm::uvec4(kOctreeMaxDepth, kOctreeLeafCapacity,
                       static_cast<std::uint32_t>(nodeCapacity),
                       static_cast<std::uint32_t>(GpuOctreeBuildStatus::Uninitialized)),
            glm::vec4(0.0f), glm::vec4(0.0f)};
}

static_assert(sizeof(glm::uvec4) == 16);
static_assert(sizeof(glm::vec4) == 16);
static_assert(CalculateOctreeNodeBufferCapacity(256) == 4097);
static_assert(alignof(GpuMassAggregate) == 16);
static_assert(sizeof(GpuMassAggregate) == 16);
static_assert(offsetof(GpuMassAggregate, centerMass) == 0);
static_assert(alignof(GpuOctreeNode) == 16);
static_assert(sizeof(GpuOctreeNode) == 64);
static_assert(offsetof(GpuOctreeNode, links) == 0);
static_assert(offsetof(GpuOctreeNode, range) == 16);
static_assert(offsetof(GpuOctreeNode, boundsMinDiameter) == 32);
static_assert(offsetof(GpuOctreeNode, boundsMax) == 48);
static_assert(alignof(GpuOctreeMetadata) == 16);
static_assert(sizeof(GpuOctreeMetadata) == 64);
static_assert(offsetof(GpuOctreeMetadata, counts) == 0);
static_assert(offsetof(GpuOctreeMetadata, configuration) == 16);
static_assert(offsetof(GpuOctreeMetadata, rootBoundsMinDiameter) == 32);
static_assert(offsetof(GpuOctreeMetadata, rootBoundsMax) == 48);
