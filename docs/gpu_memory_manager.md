# GPU Memory Manager

The grid path requires an OpenGL 4.4 core context. `StartGLU` requests OpenGL 4.4, validates `GLEW_VERSION_4_4`, and prints the actual driver version before immutable buffer storage is used.

## Resource model

`BufferRole` is the semantic identity and the sole C++ source for binding numbers through `BindingFor`. `GpuMemoryManager` owns `GpuBuffer` instances and performs lookup, lifecycle, and application-visible accounting. A `GpuBuffer` owns its OpenGL buffer handle, immutable storage, capacity, logical count, and, where configured, its CPU mapping.

GPU-resident means an OpenGL buffer has immutable storage allocated with `glBufferStorage`. Persistent-mapped is a stricter subset: a GPU-resident buffer also has one `glMapBufferRange` mapping, retained as `mappedPtr` until resize, reset, or destruction. The pointer is a CPU virtual address for that buffer storage; it is not a GPU address, OpenGL handle, or `Object *`.

| Role | Binding | Mapping policy | Current use |
| --- | ---: | --- | --- |
| `CurrentState` | 0 | Persistent coherent CPU read/write | Current position and velocity consumed by `grid.comp` |
| `NextState` | 1 | Persistent coherent CPU read | Reserved for future state ping-pong |
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
| `ObjectAcceleration` | 13 | Persistent coherent CPU read | Reserved for future GPU computation |
| `ObjectRender` | 14 | Persistent coherent CPU write | Object color |
| — | 15 | — | Reserved |

The C++ and GLSL ABI uses vec4-aligned `GpuObjectState`, `GpuObjectPhysical`, `GpuObjectDerived`, `GpuObjectControl`, `GpuObjectAcceleration`, and `GpuObjectRender` structures. Rendering resources such as per-object VAOs, VBOs, and vertex count stay in `Object` and are not part of SSBO data.

### Object identity

`Object::id` is a stable logical identity used for validation and baseline/GPU result matching. It is allocated by `AllocateObjectId()` (monotonic, starting at 1; 0 is reserved as unassigned) inside the `Object` constructor, so every creation path (CSV, default bodies, random orbiters, mouse-created objects) receives a unique ID that is deterministic within one run and is preserved by normal vector copy/move.

`GpuObjectControl` carries the same value in its fourth `uint32` field (`id`, formerly padding). `objectGpuState` and `UploadObjectState` write `control[i].id = object.id`. The struct remains 16 bytes, so binding 3 layout and capacity accounting are unchanged.

The GPU slot `i` (the index into the per-object buffers, which currently equals the `objs` vector index) is a storage location, not an identity. Consumers must match objects by `id`, never by slot.

## CPU to GPU path

CPU `Object` instances remain authoritative. After the unchanged CPU Direct N-body, collision, and integration work, the CPU writes selected buffers through typed mappings:

```cpp
auto *state = gpuMemoryManager.Get(BufferRole::CurrentState).MappedPtr<GpuObjectState>();
state[index].position = glm::vec4(object.GetPosition(), 0.0f);
```

The runtime also writes physical, derived, and control data to their dedicated mappings. There is no temporary object-state upload array and no per-frame map/unmap cycle. `GpuBuffer::Upload` remains available for GPU-only initialization uploads such as `BaseGrid`.

## Synchronization and resizing

Coherent mapping provides CPU/GPU visibility for the selected CPU-write buffers. It does not grant simultaneous ownership of a region. `GpuSynchronization` inserts a `GLsync` fence after the grid GPU work. Before the CPU writes mapped object storage again, `WaitForCpuWrite` first performs a non-blocking readiness query and only waits if the GPU has not completed. This avoids CPU writes while the compute dispatch may still read the protected data. GPU-to-CPU consumption is not currently needed; a future reader must wait for its producer fence before reading mapped bytes.

`glBufferStorage` is immutable. A capacity change creates a replacement `GpuBuffer`, establishes a replacement mapping when needed, preserves logical contents with a GPU copy, rebinds the semantic role, then releases the old resource. The old mapping is explicitly unmapped during replacement/reset/destruction.

### Grid data path

`DeformedGrid` is consumed directly as grid vertex input; there is no intermediate `gridVBO` copy target and no `glCopyBufferSubData`. The same OpenGL buffer object is bound as SSBO binding 5 for `grid.comp` writes and as `GL_ARRAY_BUFFER` when `InitializeGridRenderingResources` configures `gridVAO` (attribute 0, 3 floats, stride `sizeof(glm::vec4)`, offset 0). `gridEBO` remains a separate static index buffer. Because the VAO captures the `DeformedGrid` handle, `ComputePipeline` must create `DeformedGrid` before the VAO is configured, and a future `DeformedGrid` resize would require re-pointing the VAO.

```text
BaseGrid + CurrentState/ObjectDerived -> grid.comp -> DeformedGrid
    -> glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT) -> gridVAO -> DrawGrid()
```

`glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)` establishes the GPU-side compute-write -> vertex-fetch dependency. `glFenceSync` (via `FenceGpuCompletion`/`WaitForCpuWrite`) is conceptually separate: it is a GPU completion checkpoint for CPU synchronization of mapped object storage, not a resource-visibility barrier, and neither replaces the other. No `GL_ALL_BARRIER_BITS` or `glFinish` is used.

## Scope boundary

This layer does not migrate physics. CPU Direct N-body, collision, integration, timestep behavior, and the grid deformation equation remain unchanged. `grid.comp` is still the existing grid-deformation consumer, not a N-body, mass accumulation, field, blur, or integration compute pass.

## Accounting

Logical bytes, allocated bytes, peak allocated bytes, allocation count, and reallocation count refer to application-visible OpenGL resources, not physical VRAM residency. `MappedBufferCount` and `PersistentMappedBytes` report the subset with retained CPU mappings.
