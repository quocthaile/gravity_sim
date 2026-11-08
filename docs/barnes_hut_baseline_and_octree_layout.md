# Barnes-Hut Baseline and Octree Layout

This note records the repository findings and data contract for the first two
stages of `KE_HOACH_TICH_HOP_GPU_BARNES_HUT_SCHWARZSCHILD.md`. Stage 2 allocates
Octree buffers, but no tree-building or force compute pass consumes them yet.

## Stage 0: Direct N-body baseline

### Confirmed implementation

| Item | Current behavior |
| --- | --- |
| Object state | `GpuObjectState` contains two `vec4`s (32 bytes). `CurrentState` and `NextState` are bound at 0 and 1. |
| Physics inputs | `ObjectPhysical` (mass/density) is binding 2; `ObjectControl` is binding 3; `ObjectDerived` (radius/Schwarzschild radius) is binding 12. |
| Force and collision | `nbody.comp` launches one invocation per target, loops over all sources, excludes self and initializing objects, and writes both `ObjectAcceleration` (binding 13) and `ObjectCollision` (binding 15). |
| Integration | `integrate.comp` reads acceleration and collision output, writes `NextState`, then the manager swaps the current/next buffers and rebinds 0/1 without copying. |
| Grid | `grid.comp` reads `CurrentState`, `ObjectDerived`, and `BaseGrid` (binding 4), writes `DeformedGrid` (binding 5), and retains the per-object Schwarzschild-radius displacement calculation. |
| Existing synchronization | N-body and integration dispatches are each followed by `GL_SHADER_STORAGE_BARRIER_BIT`; the grid dispatch is followed by `GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT`. |
| SSBO allocation | `ComputePipeline()` allocates the Octree roles at bindings 6-8, 10, and 11. Binding 9 (`AccelerationField`) stays unallocated. The new buffers currently have no compute consumer. |

Current frame order:

```text
RunObjectStateSync
  -> nbody.comp
  -> integrate.comp
  -> SwapCurrentNext
  -> grid.comp
  -> DrawGrid / DrawObjects
```

### Executed baseline

The current Debug executable was built and run in an isolated working directory;
existing `numerical_error` files in the repository were not overwritten.

| Measurement | Result |
| --- | --- |
| Build | `gravity_sim_3Dgrid` and `gravity_sim_3Dgrid_baseline` built successfully with MSVC / Visual Studio 18 2026. |
| Device string reported by OpenGL | `4.4.0 NVIDIA 610.62`; the application does not print the GPU model/renderer string. |
| Initial objects | 202 (200 random orbiters plus two default objects). |
| Timestep / force parameters | `dt = 1/96 s`, `epsilon = 10`, `G = 6.6743e-11`. |
| N-body dispatch | 202 objects, 4 workgroups, local size 64. |
| Captured positions | 11 snapshots at 96-frame intervals, 202 objects each (2,222 rows); the requested 10-second simulation interval was captured. |
| Reported whole-frame rate | 119.641-142 FPS across the observed one-second windows (about 7.042-8.358 ms/frame). These are application frame timings, not isolated GPU compute timings. |
| Isolated N-body GPU time | Not measured: the current implementation has no GPU timer query around the N-body dispatch. Do not treat frame time as compute time. |

The initial sample contains object slot 0 at `(3844, 0, 0)` and slot 1 at
`(0, 0, 0)`. The program does not record a random seed, so the remaining random
positions are not a reproducible baseline input.

### Integration points for later stages

- Add the builder and dispatch ordering in `gravity_sim_3Dgrid_function.cpp`,
  called after state synchronization and before the force solver.
- Keep collision results valid if the direct force loop is replaced or split.
- Keep `integrate.comp`, `SwapCurrentNext()`, and the existing Schwarzschild grid
  path unchanged in the initial integration.

An existing untracked `PLAN_MASS_AGGREGATION.md` describes an earlier, incompatible
scope in which aggregates drive grid deformation and force remains direct. This
Barnes-Hut plan instead requires Barnes-Hut force and preserves the current
per-object Schwarzschild grid. The older plan was left untouched; resolve that
document conflict before using it as implementation guidance.

## Stage 1: Octree data contract

The compiled C++ definitions are in `include/gpu_octree_types.hpp`. The GLSL
declarations must mirror the following four 16-byte fields, in the same order,
using `uvec4` and `vec4`:

### `GpuOctreeNode` — 64 bytes

| Offset | Field | Components |
| ---: | --- | --- |
| 0 | `uvec4 links` | `x`: first child; `y`: child count; `z`: occupied-octant mask; `w`: node flags |
| 16 | `uvec4 range` | `x`: first leaf object index; `y`: leaf object count; `z`: depth; `w`: reserved |
| 32 | `vec4 boundsMinDiameter` | `xyz`: minimum bound; `w`: cubic node side length |
| 48 | `vec4 boundsMax` | `xyz`: maximum bound; `w`: reserved |

Only occupied nodes are allocated. `kOctreeNodeFlagOccupied` marks an initialized
node and `kOctreeNodeFlagLeaf` marks a leaf. Internal nodes have no object range.
Leaf ranges refer to contiguous entries in `ObjectIndexList`; a leaf may contain
more than `kOctreeLeafCapacity` objects when objects remain coincident at maximum
depth.

Children of a node occupy one contiguous range, ordered by octant number. To map
an occupied octant to its child index, add the population count of the lower
occupied bits in `links.z` to `links.x`. The child count equals the population
count of the mask.

### `GpuOctreeMetadata` — 64 bytes

| Offset | Field | Components |
| ---: | --- | --- |
| 0 | `uvec4 counts` | active object count, node count, leaf count, root index |
| 16 | `uvec4 configuration` | maximum depth, leaf capacity, node capacity, build status |
| 32 | `vec4 rootBoundsMinDiameter` | root minimum bound and cube side length |
| 48 | `vec4 rootBoundsMax` | root maximum bound; `w` reserved |

Build status values are `Uninitialized`, `Building`, `Empty`, `Valid`,
`NodeCapacityExceeded`, and `InvalidBounds`. `kInvalidOctreeIndex` is
`0xFFFFFFFFu`; an empty tree has no root and zero node/leaf counts.

### Index arrays and construction policy

- `ObjectIndexList` stores original GPU object slots as `uint32_t`, not stable
  logical object IDs. Each active object appears exactly once, in its leaf's
  contiguous range. Octants are emitted in ascending order and slots within a
  leaf in ascending input order for deterministic output.
- `ObjectLeafIndex` stores one leaf index per original object slot and uses
  `kInvalidOctreeIndex` for inactive/unassigned slots. It lets traversal open
  every ancestor of the target leaf, preventing self-mass from being accepted as
  an aggregate.
- The root is a padded cube enclosing active `CurrentState` positions. It is not
  centered on or anchored to Earth. Split planes use a consistent half-open
  convention; the root's maximum face is clamped into its final octant.
- Root depth is 0. Split while a leaf exceeds capacity and depth is below 16.
  Coincident objects that still share a cell at the depth limit remain together
  in that leaf and are handled directly by the later traversal.
- Start with `ObjectCapacity = 256` for the current 202-object scene. Use the
  conservative sparse-tree capacity `1 + ObjectCapacity * MAX_OCTREE_DEPTH`,
  giving `4,097` nodes at depth 16. This is a design limit, not an allocated
  buffer yet.
- Allocate child blocks monotonically and check the whole block against capacity
  before writing any child. Any failed reservation sets `NodeCapacityExceeded`;
  the tree is unusable and dependent passes must not traverse it. Set `Valid`
  only after all build work has completed and the final dispatch dependency is
  visible.
- Bounds must be finite and have positive side length. Degenerate extents are
  padded before subdivision; invalid/non-finite bounds produce
  `InvalidBounds`, not a partially usable tree.

`GpuMassAggregate` is one `vec4` (COM xyz, mass w). `alignas(16)`, `sizeof`, and
field-offset assertions in the C++ header lock the ABI for the aggregate, node,
and metadata records. `CalculateOctreeNodeBufferCapacity()` returns
`1 + objectCapacity * 16` and rejects capacities that cannot fit the 32-bit
metadata/index layout.

Stage 2 allocates MassField and OctreeNodes at node capacity, ObjectIndexList
and ObjectLeafIndex at object capacity, and one OctreeMetadata element. For the
default object capacity of 256, that means 4,097 node slots. The OpenGL SSBO
block-size limit is checked before allocation and resize. Metadata starts in the
Uninitialized state; resizing resets metadata and invalidates the logical
contents of the tree arrays. The runtime verifies each role's actual binding
after creation and resize. A runtime growth test materialized 257 objects,
exercising object buffers' 256-to-512 resize, corresponding Octree buffer
resizes, and post-resize binding checks without errors. No GLSL shader currently
consumes these records.
