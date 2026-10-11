# GPU Memory Manager

The grid path requires an OpenGL 4.4 core context. `StartGLU` requests OpenGL 4.4, validates `GLEW_VERSION_4_4`, and prints the actual driver version before immutable buffer storage is used.

## Resource model

`BufferRole` is the semantic identity and the sole C++ source for binding numbers through `BindingFor`. `GpuMemoryManager` owns `GpuBuffer` instances and performs lookup, lifecycle, and application-visible accounting. A `GpuBuffer` owns its OpenGL buffer handle, immutable storage, capacity, logical count, and, where configured, its CPU mapping.

GPU-resident means an OpenGL buffer has immutable storage allocated with `glBufferStorage`. Persistent-mapped is a stricter subset: a GPU-resident buffer also has one `glMapBufferRange` mapping, retained as `mappedPtr` until resize, reset, or destruction. The pointer is a CPU virtual address for that buffer storage; it is not a GPU address, OpenGL handle, or `Object *`.

| Role | Binding | Mapping policy | Current use |
| --- | ---: | --- | --- |
| `CurrentState` | 0 | Persistent coherent CPU observation | Logical current position and velocity consumed by compute |
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
| `ObjectDerived` | 12 | Persistent coherent CPU write | Radius and Schwarzschild radius |
| `ObjectAcceleration` | 13 | Persistent coherent CPU observation | GPU acceleration plus ordered velocity contribution for collision-compatible integration |
| `ObjectRender` | 14 | Persistent coherent CPU write | Object color |
| — | 15 | — | Reserved |

`CurrentState` and `NextState` are logical roles, not permanent physical buffer identities. `GpuMemoryManager::SwapCurrentNext()` exchanges the owned buffers without copying storage, updates each buffer's role, and rebinds SSBO slots 0 and 1. Consequently `Get(CurrentState)` and its `MappedPtr` always resolve to the current timestep after a swap.

The C++ and GLSL ABI uses vec4-aligned `GpuObjectState`, `GpuObjectPhysical`, `GpuObjectDerived`, `GpuObjectControl`, `GpuObjectAcceleration`, and `GpuObjectRender` structures. Rendering resources such as per-object VAOs, VBOs, and vertex count stay in `Object` and are not part of SSBO data.

### Object identity

`Object::id` is a stable logical identity used for validation and baseline/GPU result matching. It is allocated by `AllocateObjectId()` (monotonic, starting at 1; 0 is reserved as unassigned) inside the `Object` constructor, so every creation path (CSV, default bodies, random orbiters, mouse-created objects) receives a unique ID that is deterministic within one run and is preserved by normal vector copy/move.

`GpuObjectControl` carries the same value in its fourth `uint32` field (`id`, formerly padding). `objectGpuState` and `SynchronizeObjectControlState` write `control[i].id = object.id`. The struct remains 16 bytes, so binding 3 layout and capacity accounting are unchanged.

The GPU slot `i` (the index into the per-object buffers, which currently equals the `objs` vector index) is a storage location, not an identity. Consumers must match objects by `id`, never by slot.

## Runtime state transition

CPU `Object` data initializes GPU state and handles lifecycle/control events. During normal frames, `nbody.comp` reads logical `CurrentState`, physical data, control flags, and derived radii; `integrate.comp` reads the resulting acceleration and writes logical `NextState`; then `SwapCurrentNext()` changes the logical roles without copying state. The CPU does not mirror position or velocity each frame, and the normal loop does not upload the full object state.

New objects are materialized by `objectGpuState` in their new slots. Parameter/control changes are detected and written only for the affected object by `SynchronizeObjectControlState`. Position changes while the newest object is initializing use `UploadInitializingObjectState` for that one slot in both state buffers. `GpuBuffer::Upload` remains available for initialization uploads such as `BaseGrid`.

## Synchronization and resizing

Coherent mapping provides CPU/GPU visibility for mapped buffers; `GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT` orders CPU writes before the N-body dispatch. It does not grant simultaneous ownership of a region. `GpuSynchronization` inserts a `GLsync` fence after the grid GPU work. Before the CPU writes mapped object storage again, `WaitForCpuWrite` first performs a non-blocking readiness query and only waits if the GPU has not completed. Validation readback waits for GPU completion before reading `mappedPtr`; normal performance frames do not read back state.

`glBufferStorage` is immutable. A capacity change creates a replacement `GpuBuffer`, establishes a replacement mapping when needed, preserves logical contents with a GPU copy, rebinds the semantic role, then releases the old resource. The old mapping is explicitly unmapped during replacement/reset/destruction.

### Grid data path

`DeformedGrid` is consumed directly as grid vertex input; there is no intermediate `gridVBO` copy target and no `glCopyBufferSubData`. The same OpenGL buffer object is bound as SSBO binding 5 for `grid.comp` writes and as `GL_ARRAY_BUFFER` when `InitializeGridRenderingResources` configures `gridVAO` (attribute 0, 3 floats, stride `sizeof(glm::vec4)`, offset 0). `gridEBO` remains a separate static index buffer. Because the VAO captures the `DeformedGrid` handle, `ComputePipeline` must create `DeformedGrid` before the VAO is configured, and a future `DeformedGrid` resize would require re-pointing the VAO.

```text
BaseGrid + CurrentState/ObjectDerived -> grid.comp -> DeformedGrid
    -> glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT) -> gridVAO -> DrawGrid()
```

`glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)` establishes the GPU-side compute-write -> vertex-fetch dependency. `glFenceSync` (via `FenceGpuCompletion`/`WaitForCpuWrite`) is conceptually separate: it is a GPU completion checkpoint for CPU synchronization of mapped object storage, not a resource-visibility barrier, and neither replaces the other. No `GL_ALL_BARRIER_BITS` or `glFinish` is used.

## Validation status

`gravity_sim_3Dgrid_phase2_validation` creates a hidden OpenGL 4.4 context, compiles and links the N-body and integration shaders, checks acceleration and one-step/multi-step state against an independent snapshot-based CPU oracle by stable object ID, and verifies physical buffer handles and SSBO bindings exchange on every step. This validates the GPU transition and its synchronization path. It does not claim numerical equivalence to `src/gravity_sim_3Dgrid_baseline.cpp`: that legacy loop uses no epsilon softening and updates positions in vector order, while the GPU transition intentionally reads one immutable timestep snapshot. `DrawObjects()` remains CPU-driven and is outside this phase.

## Accounting

Logical bytes, allocated bytes, peak allocated bytes, allocation count, and reallocation count refer to application-visible OpenGL resources, not physical VRAM residency. `MappedBufferCount` and `PersistentMappedBytes` report the subset with retained CPU mappings.
