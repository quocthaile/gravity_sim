# GPU Memory Manager

The grid path requires an OpenGL 4.4 core context. `StartGLU` requests OpenGL 4.4, validates `GLEW_VERSION_4_4`, and prints the actual driver version before immutable buffer storage is used.

## Resource model

`BufferRole` is the semantic identity and the sole C++ source for binding numbers through `BindingFor`. `GpuMemoryManager` owns `GpuBuffer` instances and performs lookup, lifecycle, and application-visible accounting. A `GpuBuffer` owns its OpenGL buffer handle, immutable storage, capacity, logical count, and, where configured, its CPU mapping.

GPU-resident means an OpenGL buffer has immutable storage allocated with `glBufferStorage`. Persistent-mapped is a stricter subset: a GPU-resident buffer also has one `glMapBufferRange` mapping, retained as `mappedPtr` until resize, reset, or destruction. The pointer is a CPU virtual address for that buffer storage; it is not a GPU address, OpenGL handle, or `Object *`.

| Role | Binding | Mapping policy | Current use |
| --- | ---: | --- | --- |
| `CurrentState` | 0 | Persistent coherent mapping for selected CPU diagnostics | Logical current position/velocity consumed by compute and read directly by the object vertex shader |
| `NextState` | 1 | Persistent coherent CPU observation | Logical destination for the next state transition |
| `ObjectPhysical` | 2 | Persistent coherent CPU write | Mass and density |
| `ObjectControl` | 3 | Persistent coherent CPU write | Initializing, launched, and target flags plus stable object `id` |
| `BaseGrid` | 4 | GPU-only | Static grid input |
| `DeformedGrid` | 5 | GPU-only | Grid compute output; also the grid VAO vertex source |
| `MassField` | 6 | GPU-only | Reserved |
| `BlurFieldA` | 7 | GPU-only | Reserved |
| `BlurFieldB` | 8 | GPU-only | Reserved |
| `AccelerationField` | 9 | GPU-only | Reserved |
| `LODMetadata` | 10 | GPU-only | Reserved |
| `Metrics` | 11 | GPU-only | Reserved |
| `ObjectDerived` | 12 | Persistent coherent CPU write | Radius and Schwarzschild radius; the vertex shader uses radius as visual scale |
| `ObjectAcceleration` | 13 | Persistent coherent CPU observation | One `vec4 acceleration`: net gravity from the current snapshot; 16 bytes/body |
| `ObjectRender` | 14 | Persistent coherent CPU write | Per-object color read by the vertex shader |
| `ObjectCollision` | 15 | Persistent coherent CPU initialization and observation | One `uint32 overlapCount`; rewritten by N-body for every dispatched target; 4 bytes/body |

`CurrentState` and `NextState` are logical roles, not permanent physical buffer identities. `GpuMemoryManager::SwapCurrentNext()` exchanges the owned buffers without copying storage, updates each buffer's role, and rebinds SSBO slots 0 and 1. Consequently `Get(CurrentState)` and its `MappedPtr` always resolve to the current timestep after a swap.

The C++ and GLSL ABI uses matching std430 records for `GpuObjectState`, `GpuObjectPhysical`, `GpuObjectDerived`, `GpuObjectControl`, `GpuObjectAcceleration`, `GpuObjectCollision`, and `GpuObjectRender`. Rendering uses one shared unit-sphere VAO/VBO; `Object` retains logical identity, visual color, simulation/control metadata, and its GPU slot, but no longer owns duplicate mesh resources.

### Per-object layouts and accounting

| Structure / role | Layout | Bytes/body |
| --- | --- | ---: |
| `GpuObjectState` / `CurrentState` | `vec4 position` + `vec4 velocity` | 32 |
| `GpuObjectState` / `NextState` | `vec4 position` + `vec4 velocity` | 32 |
| `GpuObjectPhysical` | `vec4 massDensity` | 16 |
| `GpuObjectDerived` | `vec4 radiusRs` | 16 |
| `GpuObjectControl` | four `uint32` values | 16 |
| `GpuObjectAcceleration` | `vec4 acceleration` | 16 |
| `GpuObjectRender` | `vec4 color` | 16 |
| `GpuObjectCollision` | `uint32 overlapCount` | 4 |
| **Total** | Eight per-object buffers | **148** |

`GpuObjectAcceleration` contains only net gravitational acceleration for the immutable
`CurrentState` snapshot. With masses in kg, position/radius in the existing simulation-coordinate
units, and the shader's distance conversion by 1000, acceleration is expressed in m/s^2. The
collision record is independent: `overlapCount` counts other non-initializing objects for which
the existing nonzero-distance test passes and `target.radius + source.radius > distance`.
Self-interaction is excluded. Both records are overwritten for every valid target each N-body
dispatch; initializing targets write zero to both, and new slots are initialized to zero before
their first dispatch.

Integration first computes `vGravity = vCurrent + acceleration / 96`, then applies the separate
collision factor `(-0.2)^overlapCount` to that velocity, and advances position by `vNext / 94`.
An odd count reverses sign; an even count preserves sign; the magnitude is `0.2^count` and can
underflow to zero for sufficiently large counts. The response is now independent of source-loop
ordering, but is not trajectory-equivalent to the former ordered collision-weighted accumulation.
Paused and initializing records copy both state vectors exactly; while paused, N-body diagnostics
are still calculated, whereas initializing targets write zero acceleration and collision output.

The eight buffers therefore require 148 bytes per logical object and 148 bytes per capacity slot:
37,888 bytes at capacity 256 and 75,776 bytes at capacity 512. This excludes grid/field storage and
driver/allocation overhead and is not a measurement of physical VRAM residency.

### Object identity

`Object::id` is a stable logical identity used for validation and baseline/GPU result matching. It is allocated by `AllocateObjectId()` (monotonic, starting at 1; 0 is reserved as unassigned) inside the `Object` constructor, so every creation path (CSV, default bodies, random orbiters, mouse-created objects) receives a unique ID that is deterministic within one run and is preserved by normal vector copy/move.

`GpuObjectControl` carries the same value in its fourth `uint32` field (`id`, formerly padding). `objectGpuState` and `SynchronizeObjectGpuProperties` write `control[i].id = object.id`. The struct remains 16 bytes, so binding 3 layout and capacity accounting are unchanged.

`Object::gpuStateIndex` records the exact packed per-object buffer slot assigned by `objectGpuState`. The application maintains `objs` and the GPU arrays as the same dense order: runtime additions append and are materialized before the next draw; no object removal/reordering operation currently exists. `DrawObjects()` checks this invariant and draws `gl_InstanceID == gpuStateIndex`. The vertex shader reads position from binding 0, visual radius from binding 12, and color from binding 14. It never reads `Object::position` or `mappedPtr`.

The GPU slot is a packed storage location, not an identity. CPU validation matches objects by stable `id`; the rendering batch uses the verified dense slot order and rejects an order mismatch instead of drawing the wrong state.

## Runtime state transition

CPU `Object` data initializes GPU state and handles lifecycle/control events. The main program materializes its initial object list immediately after creating the state buffers. Each frame, `RunObjectStateSync()` processes initializing-object input, appends new objects when required, and synchronizes changed metadata before physics dispatch. During normal frames, `nbody.comp` reads logical `CurrentState`; `integrate.comp` writes logical `NextState`; its `GL_SHADER_STORAGE_BARRIER_BIT` orders compute writes before later shader access, then `SwapCurrentNext()` changes logical roles and rebinds bindings 0/1. `RunGridCompute()` reads the newly active state and keeps its existing vertex-fetch barrier for `DeformedGrid`. `DrawObjects()` then explicitly binds the current logical role at SSBO binding 0 (plus `ObjectDerived` at 12 and `ObjectRender` at 14) and submits one instanced draw. The shader reads the new current state directly; no CPU position read or wait occurs in the render path.

New objects are materialized by `objectGpuState` in their new slots. Parameter/control changes are detected and written only for the affected object by `SynchronizeObjectGpuProperties`. Position changes while the newest object is initializing use `UploadInitializingObjectState` for that one slot in both state buffers. `GpuBuffer::Upload` remains available for initialization uploads such as `BaseGrid`.

After materialization, CPU `Object::position` and `Object::velocity` are not continuously updated
from or used instead of GPU runtime state. The optional trajectory CSV reads `CurrentState` through
its mapped buffer after a completion checkpoint: one initial sample and one sample every 96
simulation transitions, not a full readback on each frame.

## Synchronization and resizing

Coherent mapping provides visibility but does not grant simultaneous ownership of a region. Instanced rendering requires no CPU fence: compute writes are made visible to subsequent SSBO consumers by the integration dispatch's `GL_SHADER_STORAGE_BARRIER_BIT`, and commands remain ordered on the GL context. `WaitForGpuCompletion()` is retained for CPU lifecycle writes, resize, and explicit mapped diagnostics; when no checkpoint exists, it creates a client-mapped barrier and fence at that point, then checks completion with finite waits. Failure or an unexpected wait result throws. `RunGridCompute()` no longer inserts a completion fence on every frame. Its `GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT` remains for the separate compute-write-to-grid-vertex-fetch dependency.

`glBufferStorage` is immutable. A capacity change creates a replacement `GpuBuffer`, establishes a replacement mapping when needed, preserves logical contents with a GPU copy, rebinds the semantic role, then releases the old resource. The old mapping is explicitly unmapped during replacement/reset/destruction.

### Grid data path

`DeformedGrid` is consumed directly as grid vertex input; there is no intermediate `gridVBO` copy target and no `glCopyBufferSubData`. The same OpenGL buffer object is bound as SSBO binding 5 for `grid.comp` writes and as `GL_ARRAY_BUFFER` when `InitializeGridRenderingResources` configures `gridVAO` (attribute 0, 3 floats, stride `sizeof(glm::vec4)`, offset 0). `gridEBO` remains a separate static index buffer. Because the VAO captures the `DeformedGrid` handle, `ComputePipeline` must create `DeformedGrid` before the VAO is configured, and a future `DeformedGrid` resize would require re-pointing the VAO.

```text
BaseGrid + CurrentState/ObjectDerived -> grid.comp -> DeformedGrid
    -> glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT) -> gridVAO -> DrawGrid()
```

`glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)` establishes the grid compute-write -> vertex-fetch dependency. `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)` after integration establishes visibility of state writes to subsequent grid and object vertex-shader SSBO reads. `GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT` plus a fence is used only before CPU access to persistently mapped data; it is not used by normal rendering. No `GL_ALL_BARRIER_BITS` or `glFinish` is used.

## Validation status

The `gravity_sim_3Dgrid_phase2_validation` executable creates a hidden OpenGL 4.4 context, compiles
and links the compute/render shaders, and renders into an offscreen framebuffer. It checks GPU
positions with deliberately stale CPU positions, scale/color, zero/one/two instances, and invalid
slots. Its independent snapshot CPU oracle covers two-body acceleration, zero interaction and
zero-object dispatch, acceleration and overlap-count output, 8-step ping-pong, single/multiple
collision parity, exact paused and initializing-state preservation, initializing exclusion and
launch, append without resize, and growth from 256 to 512 with preservation and
logical-count/capacity/binding checks across all eight per-object buffers.

The oracle is only for shader/state-contract correctness. No trajectory equivalence or matched
baseline result is claimed: the legacy baseline's force softening, integration, collision, and
update-order semantics differ. Baseline comparison and performance measurement are separate work.

## Phase 2 Rendering Status

- **Implemented:** one shared unit-sphere VAO/VBO, one `glDrawArraysInstanced()` batch, active-state SSBO lookup at binding 0, visual scale at binding 12, and color at binding 14.
- **Test coverage:** hidden-context FBO checks for GPU position, scale/color, stale CPU positions, invalid slots, and 0/1/2 instances; compute checks cover collision, pause/resume, init/launch, append, and capacity growth.
- **Executed:** CMake GCC Release validation target build succeeded (`g++` 16.2.0, `-O3 -DNDEBUG`). Exact commands: `cmake --build build/gcc-release --target gravity_sim_3Dgrid_phase2_validation` and `build/gcc-release/gravity_sim_3Dgrid_phase2_validation.exe`. The executable exited 0 on OpenGL `4.4.0 - Build 32.0.101.7085`, `Intel(R) UHD Graphics` / `Intel`; 14 transitions and 14 oracle checkpoints passed. Maximum errors: acceleration `6.51856e-5 m/s^2`, relative acceleration `3.62458e-7`, collision count `0`, position `6.98493e-10` simulation units, and velocity `1.30119e-7` units. Its final GL error check passed.
- **CTest blocker:** CMake Tools `RunCtest_CMakeTools` returned `Build failed` before test launch. Diagnostics identify the unchanged `src/gravity_sim_3Dgrid_baseline.cpp` error `'setprecision' is not a member of 'std'`; baseline remains unmodified. This does not invalidate the separately executed validation executable, but CTest itself is not marked passed.
- **Not verified:** interactive camera/control behavior, full-application visual inspection, large-scene performance, or a matched baseline comparison. FBO checks use a hidden context and 128x128 target.
- **Deferred:** Mass Accumulation, Gaussian Blur, Dynamic Grid/Field LOD, non-sphere geometry batches, compaction/indirect drawing, and culling/LOD.

## Accounting

Logical bytes, allocated bytes, peak allocated bytes, allocation count, and reallocation count refer to application-visible OpenGL resources, not physical VRAM residency. `MappedBufferCount` and `PersistentMappedBytes` report the subset with retained CPU mappings.
