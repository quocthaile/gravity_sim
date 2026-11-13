# Barnes-Hut Baseline and Octree Layout

This note records the repository findings and data contract for
`KE_HOACH_TICH_HOP_GPU_BARNES_HUT_SCHWARZSCHILD.md`, including the implemented
GPU Octree, mass aggregation, and Barnes-Hut force stages.

## Stage 0: Direct N-body baseline

### Confirmed implementation

| Item | Current behavior |
| --- | --- |
| Object state | `GpuObjectState` contains two `vec4`s (32 bytes). `CurrentState` and `NextState` are bound at 0 and 1. |
| Physics inputs | `ObjectPhysical` (mass/density) is binding 2; `ObjectControl` is binding 3; `ObjectDerived` (radius/Schwarzschild radius) is binding 12. |
| Original force baseline | `nbody.comp` launches one invocation per target, loops over all sources, excludes self and initializing objects, and writes both `ObjectAcceleration` (binding 13) and `ObjectCollision` (binding 15). It remains available as a direct reference but is no longer the main simulation force pass. |
| Current force and collision | `barnes_hut.comp` writes `ObjectAcceleration` (binding 13); `collision.comp` writes `ObjectCollision` (binding 15) using the prior pairwise overlap rule. |
| Integration | `integrate.comp` reads acceleration and collision output, writes `NextState`, then the manager swaps the current/next buffers and rebinds 0/1 without copying. |
| Grid | `grid.comp` reads `CurrentState`, `ObjectDerived`, and `BaseGrid` (binding 4), writes `DeformedGrid` (binding 5), and retains the per-object Schwarzschild-radius displacement calculation. |
| Existing synchronization | Each dependent compute dispatch (bounds, construction, aggregation, Barnes-Hut, collision, integration) is separated with `GL_SHADER_STORAGE_BARRIER_BIT`; the grid dispatch is followed by `GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT`. |
| SSBO allocation | `ComputePipeline()` allocates the Octree roles at bindings 6-8, 10, and 11. Binding 9 (`AccelerationField`) stays unallocated. |

Current frame order:

```text
RunObjectStateSync
  -> bounding_box.comp
  -> build_octree.comp
  -> mass_aggregation.comp
  -> barnes_hut.comp
  -> collision.comp
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

- Keep the Octree construction after state synchronization and before the force
  solver so both consume the same `CurrentState` snapshot.
- Keep collision results valid if the direct force loop is replaced or split.
- Keep `integrate.comp`, `SwapCurrentNext()`, and the existing Schwarzschild grid
  path unchanged.

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
resizes, and post-resize binding checks without errors.

## Stage 3: GPU bounds and Octree construction

`bounding_box.comp` scans `CurrentState` and excludes objects whose
`ObjectControl.initializing` flag is set. It computes a finite padded cubic
root, reports empty or invalid input through `OctreeMetadata`, and handles
capacity mismatches without producing a usable tree. The root is offset from
the component-wise minimum by the padding; its side is the longest extent plus
twice that padding, ensuring the active AABB is enclosed.

`build_octree.comp` builds a sparse breadth-first tree and partitions original
object slots into contiguous leaf ranges in ascending octant order. Child blocks
are contiguous, `links.z` records the occupied-octant mask, and
`ObjectLeafIndex` maps active object slots to their leaf. Leaves stop splitting
at the configured capacity or maximum depth, so coincident objects remain in a
valid leaf at the depth limit. Empty input and object/node capacity failures
produce explicit metadata statuses. Each pass is dispatched before its
consumer with an SSBO memory barrier.

The current implementation uses one compute invocation for each entire pass
(`local_size_x = 1`). This favors a straightforward correctness baseline and is
not a parallel or performance-optimized construction algorithm. Debug builds
read back the result and validate object coverage and uniqueness, leaf mapping,
child ranges/depths, and node bounds.

The construction is called after object-state synchronization. Direct N-body
was retained as a reference, while the main frame now uses the tree for force
calculation; integration, collision behavior, buffer swapping, and
Schwarzschild grid deformation are preserved.

### Stage 3 runtime verification

The Debug application ran in an isolated working directory with the default
202-object scene. The debug Octree validator completed successfully while the
application continued through mass aggregation, Barnes-Hut, collision,
integration, and grid deformation for more than 240 frames and wrote its
position capture. The default-scene run does not independently exercise an
all-initializing scene or coincident-object input.

## Stage 4: Mass aggregation

`mass_aggregation.comp` runs one invocation over the tree in reverse node-index
order. Leaves sum source mass and mass-weighted positions from their contiguous
object ranges; internal nodes combine their already-computed children. Double
precision intermediates reduce accumulation error before storing each COM and
mass in the existing `vec4` `MassField` layout. Empty input writes a zero root
slot without dividing by zero, and non-finite/out-of-range input marks
`InvalidAggregate`.

Debug validation independently sums active CPU-mapped object masses and
mass-weighted positions, compares the root mass and COM, and checks every
aggregate for finite values. The checks use the same `CurrentState` snapshot as
the tree builder.

## Stage 5: GPU Barnes-Hut

`barnes_hut.comp` launches one invocation per target and traverses the sparse
Octree using a bounded local stack. The configured opening criterion is
`nodeSide / distance < theta`, with `theta = 0.5`; any node whose bounds contain
the target is always opened. Leaves evaluate direct source interactions, skip
self-interaction, and ignore initializing objects. The force retains the
existing `G`, position-to-meter conversion (`distance * 1000`), and epsilon
softening formula. A stack of 129 indices covers the configured depth-16
eight-way tree; overflow and malformed-tree/non-finite results set explicit
metadata failure statuses instead of silently continuing.

Collision remains a separate direct pairwise pass. It preserves the previous
strict radius-overlap condition and skips exact zero-distance pairs as the
original `nbody.comp` did. Integration, `SwapCurrentNext()`, and grid
deformation remain downstream and unchanged.

Debug validation checks finite accelerations and zero acceleration for
initializing objects; it also compares collision counts against a CPU reference.
The default 202-object scene ran through the construction, aggregation,
Barnes-Hut, collision, integration, and grid passes without validation errors.

## Stages 6-8: Integration, grid preservation, and measured validation

The production frame order is:

```text
RunObjectStateSync
  -> bounding_box.comp
  -> build_octree.comp
  -> mass_aggregation.comp
  -> barnes_hut.comp (theta = 0.3)
  -> collision.comp
  -> integrate.comp
  -> SwapCurrentNext
  -> grid.comp
  -> draw
```

Bounding, construction, aggregation, and force evaluation all consume the same
pre-integration `CurrentState`; SSBO barriers separate every producer and
consumer. Collision output is written before the unchanged integration shader.
The grid shader remains byte-for-byte unchanged from the Direct N-body baseline,
uses only bindings 0, 4, 5, and 12, and still executes after the state swap.

Run the Release validation sweep from an isolated working directory with:

```text
gravity_sim_3Dgrid.exe --benchmark
```

It generates `numerical_error/barnes_hut_benchmark.csv` and exits without
starting the interactive simulation. The deterministic sweep covers 32, 128,
and 202 objects, uniform/clustered/disk distributions, and theta values 0.3,
0.5, and 0.8. Each force comparison runs Direct N-body and Barnes-Hut on the
same frozen positions and masses. It records RMS/max absolute and relative
acceleration error. The clustered 32-object case also compares the final
positions after 24 fixed steps (`dt = 1/96 s`) for each theta against a Direct
N-body trajectory.

GPU timings use OpenGL `GL_TIME_ELAPSED` around each dispatch and its required
barrier, averaged over three repetitions after a warm-up. The direct total is
Direct N-body (which also writes collision counts) + integration + grid. The
Barnes-Hut total is bounds + construction + mass aggregation + Barnes-Hut +
separate collision + integration + grid. These are GPU compute totals, not full
rendered-frame times; they exclude CPU work, draw calls, and presentation.

### Executed validation results

Environment: NVIDIA GeForce RTX 3050 Laptop GPU, OpenGL 4.4.0
(`610.62` driver string), Release build, `G = 6.6743e-11`, `epsilon = 10`,
`dt = 1/96 s`, three timed repeats per case. Acceleration relative error is
RMS over objects, normalized by each Direct acceleration magnitude with a
`1e-12` floor. Times are milliseconds.

| Objects / distribution | RMS relative error (theta 0.3 / 0.5 / 0.8) | Direct total | Barnes-Hut total (theta 0.3 / 0.5 / 0.8) |
| --- | ---: | ---: | ---: |
| 32 / uniform | 0.002056 / 0.014000 / 0.039063 | 0.090 | 0.567 / 0.584 / 0.554 |
| 32 / clustered | 0.000239 / 0.001453 / 0.007635 | 0.089 | 0.651 / 0.635 / 0.620 |
| 32 / disk | 0.003126 / 0.008378 / 0.059375 | 0.099 | 0.621 / 0.618 / 0.593 |
| 128 / uniform | 0.001487 / 0.009312 / 0.038051 | 0.495 | 2.494 / 2.300 / 2.123 |
| 128 / clustered | 0.000497 / 0.004975 / 0.024193 | 0.454 | 2.578 / 2.557 / 2.495 |
| 128 / disk | 0.003200 / 0.014692 / 0.049816 | 0.471 | 2.606 / 2.444 / 2.344 |
| 202 / uniform | 0.001408 / 0.008112 / 0.031526 | 0.785 | 3.267 / 3.070 / 2.914 |
| 202 / clustered | 0.001029 / 0.006604 / 0.033276 | 0.666 | 3.193 / 3.123 / 3.049 |
| 202 / disk | 0.004476 / 0.013706 / 0.042400 | 0.689 | 3.572 / 3.279 / 3.102 |

For the representative uniform, 202-object case at theta 0.3, the measured
per-pass GPU times were: bounds 0.276 ms, construction 0.829 ms, aggregation
1.010 ms, direct N-body 0.656 ms, Barnes-Hut force 0.665 ms, collision 0.357
ms, integration 0.009 ms, and grid 0.120 ms. The direct total includes collision
calculation within its N-body shader; the Barnes-Hut total includes the separate
collision pass.

For the clustered 32-object, 24-step trajectory, RMS/max position error was
`0.000405 / 0.001007` at theta 0.3, `0.001003 / 0.002303` at theta 0.5, and
`0.010958 / 0.038148` at theta 0.8 (simulation units). Theta 0.3 is the
production default because it yielded the lowest sampled acceleration and
trajectory errors; it was not selected for frame rate. The measured Barnes-Hut
pipeline is slower than Direct N-body for every tested size and distribution
and theta.
With the current serial bounds/tree/aggregation passes and these small systems,
the implementation is a correctness milestone, not a performance improvement.

The sweep found no NaN/Inf, invalid-tree status, node-capacity overflow,
traversal-stack overflow, or buffer-range failure. The matrix does not establish
large-N behavior, an accuracy tolerance for all workloads, or the absence of
rare failures beyond the tested snapshots. Its raw per-case data is stored in
the generated CSV alongside per-pass timings and maximum acceleration errors.
