# Object-State Management

This document describes how per-object simulation state moves from the CPU to the GPU in `gravity_sim_3Dgrid`.

## 1. Runtime flow

Each frame follows this order:

1. The CPU updates every `Object` in `objs` (velocity, position, mass, radius, and Schwarzschild radius `rs`).
2. `EnsureObjectStateCapacity(objectCount)` makes sure the object-state SSBO has enough storage.
3. `sphreStateData` is resized to the current object count.
4. The main loop packs each `Object` into one `SphreStateCpu` record.
5. `UploadObjectState(objectCount)` uploads the active records with `glBufferSubData`.
6. `RunGridCompute(objectCount)` dispatches `grid.comp` with the active object count.
7. The compute shader reads the object records and deforms the grid.

The upload and compute stages use only the first `objectCount` records. Unused capacity at the end of the SSBO is not processed.

## 2. CPU representation

The CPU-side record is defined in `include/gravity_sim_3Dgrid_function.hpp`:

```cpp
struct SphreStateCpu
{
    glm::vec4 position_mass;
    glm::vec4 velocity_radius;
};
```

Each record contains:

| Field | Components | Meaning |
| --- | --- | --- |
| `position_mass` | `xyz` | Object position in simulation coordinates |
| `position_mass` | `w` | Object mass |
| `velocity_radius` | `xyz` | Object velocity |
| `velocity_radius` | `w` | Schwarzschild radius `rs` |

The static assertions require the record to have the same size and alignment as two `glm::vec4` values. This keeps the CPU layout compatible with the compute shader's `std430` layout.

## 3. GPU storage

Object state is stored in `sphreStateSSBO` and bound to SSBO binding point `0`:

```glsl
layout(std430, binding = 0) readonly buffer SphereStateBuffer
{
    SphereState spheres[];
};
```

The matching GLSL structure is:

```glsl
struct SphereState
{
    vec4 position_mass;
    vec4 velocity_radius;
};
```

The other compute buffers are bound as follows:

| Binding | Buffer | Access | Purpose |
| --- | --- | --- | --- |
| `0` | `sphreStateSSBO` | Read-only | Active object states |
| `1` | `baseGridSSBO` | Read-only | Original grid positions |
| `2` | `deformedGridSSBO` | Write-only | Grid positions after deformation |

## 4. Capacity management

`objectStateCapacity` tracks the number of records allocated in the object-state SSBO. The allocation policy is:

- A non-empty buffer starts with capacity `256`.
- When the current object count does not fit, capacity doubles until it fits.
- Existing capacity is reused when the object count decreases or remains unchanged.
- Reallocation uses `glBufferData` with `GL_DYNAMIC_DRAW`.

This avoids reallocating the GPU buffer every frame when objects are added or removed. The CPU vector `sphreStateData` reserves the same new capacity.

## 5. Per-frame packing

The main loop converts the simulation objects into GPU records:

```cpp
sphreStateData[i].position_mass = glm::vec4(objs[i].GetPosition(), objs[i].mass);
sphreStateData[i].velocity_radius = glm::vec4(objs[i].velocity, objs[i].rs);
```

The vector is resized to `objectCount` before packing, so `UploadObjectState` sends a contiguous array of active records starting at offset `0`.

## 6. Compute-shader consumption

`RunGridCompute` sends `objectCount` to the compute shader through `u_objectCount`. For each grid cell, `grid.comp` loops over the active objects:

```glsl
for (uint i = 0; i < u_objectCount; ++i)
{
    vec3 objPos = spheres[i].position_mass.xyz;
    float objRs = spheres[i].velocity_radius.w;
    // Calculate the object's contribution to grid displacement.
}
```

The compute shader uses position and `rs` for grid deformation. Mass and velocity are preserved in the object-state record for the simulation contract, even though this shader currently consumes only position and `rs`.

## 7. Invariants

- `sphreStateData.size()` equals the number of active objects before upload.
- `objectStateCapacity >= objectCount` before `UploadObjectState`.
- `sizeof(SphreStateCpu) == 2 * sizeof(glm::vec4)`.
- Object-state buffer binding `0` must remain consistent between C++ and `grid.comp`.
- `u_objectCount` must never exceed the number of records uploaded for the current frame.

## 8. Ownership and cleanup

The CPU owns the authoritative `Object` values in `objs`. `sphreStateData` is a per-frame transfer snapshot, while `sphreStateSSBO` is the GPU copy used by the compute pass. The SSBO is released by `Cleanup` together with the other OpenGL buffers.
