# GravitySim GPU-Resident Object State Architecture

## 1. Mục đích

Tài liệu này mô tả kiến trúc mục tiêu cho `gravity_sim` trên nhánh `feat/gpu-memory`, tập trung vào việc chuyển dữ liệu object từ CPU sang GPU-resident storage và từng bước chuyển quyền sở hữu simulation state sang GPU.

Kiến trúc được làm rõ theo ba giai đoạn chính:

1. Hoàn thiện GPU memory và object data materialization.
2. Chuyển `CurrentState -> NextState` thành state transition thực sự trên GPU.
3. Chuyển `DrawObjects()` sang GPU instanced rendering.

`src/gravity_sim_3Dgrid_baseline.cpp` được giữ độc lập làm baseline/reference implementation cho benchmark và validation. Baseline không phải là CPU mirror runtime của GPU implementation.

---

## 2. Phạm vi

### Trong phạm vi

- Object data layout trên CPU và GPU.
- GPU-resident buffers.
- `CurrentState` / `NextState`.
- Object lifecycle và upload object mới.
- `ObjectControl` làm control plane.
- `mappedPtr` cho CPU observation/debug/measurement.
- GPU N-body state transition.
- GPU instanced rendering cho objects.
- Quan hệ giữa GPU implementation và baseline.
- Synchronization và validation checkpoints.
- Khả năng mở rộng theo số lượng object.

### Ngoài phạm vi của bước hiện tại

- Thay đổi physical model của baseline.
- Refactor toàn bộ `Object` class.
- Ngay lập tức chuyển toàn bộ collision/solver architecture.
- Thay đổi các optimization algorithm như Mass Accumulation, Gaussian Blur hoặc Dynamic LOD.
- Kết luận hiệu năng trước khi có benchmark thực nghiệm.

---

## 3. Nguyên tắc ownership

Nguyên tắc trung tâm:

> CPU khởi tạo và điều khiển lifecycle; GPU giữ và tiến hóa runtime simulation state; CPU chỉ quan sát GPU state tại các checkpoint cần thiết.

Do đó cần phân biệt ba khái niệm:

```text
Initial object data
        |
        +--> GPU runtime state
        |
        +--> baseline/reference input
```

`CPU Object` không phải là một runtime shadow copy bắt buộc phải được cập nhật theo GPU ở mỗi frame.

Sau khi object được materialize lên GPU:

```text
GPU runtime state = runtime authority
```

Còn baseline:

```text
src/gravity_sim_3Dgrid_baseline.cpp
    = independent reference implementation
```

được dùng trong benchmark/validation.

---

## 4. Object data model

Object hiện tại chứa nhiều loại thông tin:

```text
Object
├── Identity
│   └── id
│
├── Simulation state
│   ├── position
│   └── velocity
│
├── Physical parameter
│   ├── mass
│   └── density
│
├── Derived state
│   ├── radius
│   └── rs
│
├── Interaction/control
│   ├── initializing
│   ├── launched
│   └── target
│
├── Acceleration
│   └── acceleration
│
├── Rendering
│   └── color
│
└── CPU/OpenGL resources
    ├── VAO
    ├── VBO
    └── vertexCount
```

GPU runtime representation:

```text
CurrentState
    position
    velocity

NextState
    position
    velocity

ObjectPhysical
    mass
    density

ObjectDerived
    radius
    rs

ObjectAcceleration
    acceleration

ObjectControl
    initializing
    launched
    target
    id

ObjectRender
    color
```

`VAO`, `VBO` và các CPU/OpenGL resource handle không phải simulation data và không thuộc GPU object-state contract.

### Object identity

`Object::id` là logical identity ổn định của object, dùng cho validation và để ghép kết quả baseline với kết quả GPU khi tính sai số.

```text
Object.id  (logical identity)
    |
    v
GpuObjectControl.id  (ObjectControl, binding 3)
```

- `id` được cấp bởi `AllocateObjectId()` trong constructor của `Object`: monotonic, bắt đầu từ 1 (0 = unassigned), deterministic trong một lần chạy, không tái sử dụng.
- Mọi đường tạo object (CSV, default objects, random orbiters, object tạo bằng chuột) đều đi qua constructor nên đều có `id` duy nhất; copy/move của `std::vector` giữ nguyên `id`.
- `GpuObjectControl.id` tái sử dụng field `uint32` thứ tư (trước đây là padding), nên struct vẫn 16 bytes và std430 layout không đổi.
- GPU slot / vector index `i` chỉ là vị trí lưu trữ, **không** phải identity. Validation phải ghép theo `id`, không theo slot.

---

## 5. GPU buffer roles

### CurrentState

Runtime input state của timestep hiện tại.

```text
position
velocity
```

Producer:

```text
GPU compute / initialization upload
```

Consumers:

```text
GPU compute
GPU rendering
CPU observation at checkpoint
```

### NextState

Destination state của timestep kế tiếp.

```text
position
velocity
```

Producer:

```text
GPU compute
```

Consumer:

```text
GPU compute after ping-pong swap
CPU observation if explicitly required
```

`NextState` không cần CPU full upload mỗi frame.

### ObjectPhysical

```text
mass
density
```

Được khởi tạo từ CPU và giữ resident trên GPU.

Chỉ cập nhật lại khi physical parameter thực sự thay đổi.

### ObjectDerived

```text
radius
rs
```

Được materialize từ CPU trong giai đoạn hiện tại.

Về sau có thể được tính/duy trì trên GPU nếu cần, nhưng đây là bước mở rộng riêng.

### ObjectAcceleration

Kết quả acceleration của GPU N-body.

```text
acceleration
```

Có thể được dùng:

```text
GPU integration
CPU validation/debug via mappedPtr
```

### ObjectControl

```text
initializing
launched
target
id
```

Đóng vai trò control plane. `id` là logical identity của object (xem Object identity), không phải slot index.

Control event từ CPU chỉ cập nhật phần object bị thay đổi, không cần upload toàn bộ object array.

### ObjectRender

```text
color
```

Phục vụ GPU rendering.

Buffer này không tham gia physics computation trừ khi một rendering rule cần control state.

### BaseGrid / DeformedGrid

`BaseGrid` (binding 4) là static grid input. `DeformedGrid` (binding 5, `glm::vec4`, GPU-only, immutable storage) là output của `grid.comp` và đồng thời là vertex source trực tiếp của `gridVAO`. Không còn `gridVBO` trung gian và không còn `glCopyBufferSubData`.

```text
CurrentState/ObjectDerived/BaseGrid
        |
        v
    grid.comp
        |
        v
   DeformedGrid
        |
        v
glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        |
        v
     gridVAO
        |
        v
    DrawGrid()
```

- `glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)` thiết lập dependency compute write -> vertex attribute fetch trên GPU.
- `glFenceSync` (`FenceGpuCompletion` / `WaitForGpuCompletion`) là GPU completion checkpoint cho CPU synchronization, tách biệt về khái niệm với memory barrier; cái này không thay thế cái kia.
- Per-object `Object::VAO/VBO` không liên quan và vẫn giữ nguyên.

---

## 6. Initial object materialization

Khi chương trình khởi tạo:

```text
CPU
 |
 | CreateObjects()
 v
Object[0..N-1]
 |
 | materialization
 v
GpuMemoryManager
 |
 +--> CurrentState
 +--> NextState
 +--> ObjectPhysical
 +--> ObjectDerived
 +--> ObjectControl
 +--> ObjectAcceleration
 +--> ObjectRender
```

Initial upload là operation theo object set.

Sau operation này:

```text
CPU -> GPU full-state upload
```

không còn là hoạt động bình thường của mỗi frame.

---

## 7. Upload policy

### Initial upload

Toàn bộ object set được upload một lần:

```text
CPU Object[0..N-1]
        |
        v
GPU buffers
```

### New object

Khi tạo object mới:

```text
CPU creates Object[N]
        |
        v
allocate/find GPU slot N
        |
        +--> CurrentState[N]
        +--> NextState[N]
        +--> ObjectPhysical[N]
        +--> ObjectDerived[N]
        +--> ObjectControl[N]
        +--> ObjectAcceleration[N]
        +--> ObjectRender[N]
```

Chỉ object mới được materialize.

Không upload lại object `0..N-1`.

### Object parameter/control update

Nếu CPU thay đổi:

```text
mass
density
color
control flag
```

thì chỉ phần tương ứng của object đó được cập nhật.

### Full upload mỗi frame

Không thuộc architecture mục tiêu.

---

## 8. Capacity management

GPU buffers cần có `capacity` lớn hơn hoặc bằng `logical object count`.

Ví dụ:

```text
capacity = 256
objectCount = 202
```

Khi thêm object:

```text
objectCount = 203
```

không cần resize.

Khi:

```text
objectCount = 257
```

mới cần resize, ví dụ:

```text
256 -> 512
```

Resize phải thực hiện khi GPU không còn sử dụng resource cũ.

Các buffer per-object phải duy trì cùng capacity:

```text
CurrentState
NextState
ObjectPhysical
ObjectDerived
ObjectControl
ObjectAcceleration
ObjectRender
```

---

## 9. GPU N-body state transition

Direct N-body vẫn có độ phức tạp:

\[
O(N^2)
\]

GPU-resident architecture không thay đổi complexity này; nó thay đổi nơi execution và data movement diễn ra.

Mỗi invocation xử lý một target object:

```text
invocation i
```

GPU đọc:

```text
CurrentState[i]
ObjectPhysical[i]
ObjectControl[i]

CurrentState[j]
ObjectPhysical[j]
ObjectControl[j]
for j = 0..N-1
```

và tính:

\[
\mathbf{a}_i =
\sum_{j\ne i}
Gm_j
\frac{\mathbf{r}_j-\mathbf{r}_i}
{\left\|\mathbf{r}_j-\mathbf{r}_i\right\|^3}
\]

Sau đó state transition:

```text
CurrentState
      |
      v
GPU N-body / integration
      |
      v
NextState
```

Không ghi trực tiếp vào `CurrentState` trong cùng state transition nếu các invocation khác vẫn cần đọc timestep hiện tại.

---

## 10. Ping-pong state

Tại timestep `t`:

```text
CurrentState = S(t)
NextState    = destination
```

GPU thực hiện:

```text
S(t) -> compute -> S(t+1)
```

ghi vào `NextState`.

Sau khi compute hoàn tất:

```text
CurrentState <-> NextState
```

Timestep tiếp theo:

```text
CurrentState = S(t+1)
NextState    = destination
```

Luồng:

```mermaid
flowchart LR
    A["CurrentState(t)"] --> B["GPU N-body + Integration"]
    B --> C["NextState(t+1)"]
    C --> D["Ping-Pong Swap"]
    D --> E["CurrentState(t+1)"]
    E --> B
```

Invariant:

> `CurrentState` luôn đại diện cho state của timestep đang được sử dụng làm input; `NextState` là destination của state kế tiếp.

---

## 11. N-body nên được migration theo từng bước

### Step A — Data path validation

Mục tiêu:

```text
CPU Object
    -> GPU CurrentState / Physical / Derived / Control
```

Kiểm chứng:

```text
position
velocity
mass
density
radius
rs
control
```

không bị sai layout/value.

### Step B — GPU acceleration

GPU tính:

```text
CurrentState + ObjectPhysical + ObjectControl
        |
        v
ObjectAcceleration
```

Baseline/reference được dùng để so sánh acceleration.

### Step C — GPU integration

GPU thực hiện:

```text
CurrentState + ObjectAcceleration
        |
        v
NextState
```

### Step D — Ping-pong ownership

```text
NextState
    |
    +--> becomes CurrentState
```

Khi bước này hoàn tất, GPU trở thành runtime owner của position/velocity.

---

## 12. CPU observation bằng mappedPtr

`mappedPtr` không phải là CPU simulation mirror.

Nó là observation/debug interface.

Conceptual flow:

```text
GPU buffer
    |
    | persistent mapped
    v
mappedPtr
    |
    v
CPU observation
```

CPU có thể đọc:

```text
CurrentState
NextState
ObjectAcceleration
```

khi buffer được cấu hình cho CPU read access và synchronization bảo đảm GPU write đã hoàn thành.

Ví dụ:

```cpp
const GpuObjectState* state =
    gpuMemoryManager
        .Get(BufferRole::CurrentState)
        .MappedPtr<GpuObjectState>();
```

CPU có thể đọc:

```text
state[i].position
state[i].velocity
```

mà không cần tạo một bản copy đầy đủ của GPU state.

---

## 13. Synchronization rule

Persistent mapping không loại bỏ synchronization.

CPU chỉ nên đọc GPU-produced data sau checkpoint:

```text
GPU compute
    |
    v
GPU completion / fence
    |
    v
CPU wait
    |
    v
mappedPtr read
```

Không nên:

```text
GPU compute
    |
    v
CPU read immediately
```

Vì vậy `mappedPtr` không làm cho GPU→CPU observation trở thành zero-cost; synchronization vẫn có thể tạo stall.

Architecture mục tiêu là:

```text
frequent:
    GPU -> GPU

infrequent:
    GPU -> CPU observation
```

---

## 14. Baseline và validation

Baseline nằm độc lập:

```text
src/gravity_sim_3Dgrid_baseline.cpp
```

Baseline thực hiện Direct N-body theo implementation gốc của project.

GPU path:

```text
feat/gpu-memory
```

thực hiện implementation đang được phát triển.

Hai configuration cần sử dụng cùng:

```text
initial conditions
simulation parameters
object count N
timestep configuration
hardware/software configuration
```

Validation concept:

```text
same initial condition
        |
        +--------------------+
        |                    |
        v                    v
baseline implementation   GPU implementation
        |                    |
        v                    v
reference result         GPU result
        |                    |
        +---------+----------+
                  |
                  v
             error metrics
```

Baseline không cần chạy đồng thời như một CPU shadow copy của GPU simulation.

Kết quả baseline và GPU được ghép theo `Object.id` / `GpuObjectControl.id`, không theo vector index hoặc GPU slot.

---

## 15. Error metrics

Các metric có thể dùng cho validation:

### Acceleration error

\[
e_a =
\left\|
\mathbf{a}^{GPU}
-
\mathbf{a}^{baseline}
\right\|
\]

### Position error

\[
e_p =
\left\|
\mathbf{x}^{GPU}
-
\mathbf{x}^{baseline}
\right\|
\]

### Velocity error

\[
e_v =
\left\|
\mathbf{v}^{GPU}
-
\mathbf{v}^{baseline}
\right\|
\]

Có thể sử dụng thêm relative error:

\[
e_{rel} =
\frac{e}{\max(\|reference\|,\epsilon)}
\]

và tổng-energy drift theo methodology của đề tài.

Không yêu cầu bitwise equality giữa CPU và GPU vì floating-point operation ordering có thể khác.

---

## 16. GPU object rendering

**Implemented and tested:** object rendering now uses a single unit-sphere VAO/VBO and one instanced batch. The old per-object radius-scaled meshes and CPU-generated model matrices are removed.

`objects.vert` reads the active `CurrentState` at SSBO binding 0, `ObjectDerived.radiusRs.x` at binding 12, and `ObjectRender.color` at binding 14. It transforms the shared local unit-sphere vertex by the per-object radius and GPU position. `Object::rs` remains physical metadata and was not used by the old object draw path; it is not substituted for visual radius.

```text
integration writes NextState
    -> GL_SHADER_STORAGE_BARRIER_BIT
    -> SwapCurrentNext() rebinds the active physical buffer at binding 0
    -> RunGridCompute() and its unchanged grid vertex-fetch barrier
    -> DrawObjects() binds the current logical roles
    -> objects.vert reads state/scale/color using gl_InstanceID
    -> glDrawArraysInstanced()
```

There is no fence or CPU mapped-position read in the normal render path. The integration shader-storage barrier makes compute writes visible to later SSBO shader consumers; context command ordering keeps the draw after the state transition. The grid's separate `GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT` remains unchanged.

## 17. DrawObjects() implementation

CPU validation checks the dense slot invariant and instance count, binds the current logical roles (`CurrentState` is not a fixed physical buffer), binds the shared sphere VAO, and submits one `glDrawArraysInstanced()` call. The vertex shader maps `gl_InstanceID` to the corresponding packed slot. `Object::gpuStateIndex` must equal its position in the live `objs` array; append-only creation materializes the new slot before the next draw. No removal or reorder path currently exists. Empty input produces no draw; invalid counts and slot ordering throw before submission.

`ObjectRender` and `ObjectDerived` remain separate from simulation state: color and scale are CPU-maintained mapped metadata, while position/velocity remain in ping-ponged `GpuObjectState`. Initializing objects render after their GPU slot has been initialized, with the same unit-sphere visual radius and color behavior.

---

## 18. ObjectControl trong rendering

`ObjectControl` is not bound by the current object rendering shader. All objects in the valid packed `objs` range, including initializing objects, are rendered. Active-object filtering, compaction, indirect drawing, and free-list management are **deferred**; there is no removal workflow to validate today.

---

## 19. Main loop hiện tại

```cpp
while (!windowShouldClose)
{
    BeginFrame();

    ProcessInput();

    ProcessObjectControlEvents();

    RunNBodyCompute();

    RunIntegration();

    SwapCurrentNext();

    RunGridCompute();

    DrawGrid(...);

    DrawObjects(objectRenderingShaderProgram, objs);

    SwapBuffers();
    PollEvents();
}
```

Đặc trưng:

```text
Normal frame: CPU control -> GPU compute and ping-pong -> grid compute -> grid and instanced object draws.

Validation: the hidden-context test renders to an FBO, then synchronizes only for explicit mapped-state oracle reads.
```

---

## 20. Object lifecycle

### Initial object

```text
Create Object
    |
    v
Materialize to GPU
    |
    v
GPU slot active
```

### Simulation

```text
CurrentState
    |
    v
GPU compute
    |
    v
NextState
    |
    v
swap
```

### New object

```text
Create Object on CPU
    |
    v
find/allocate GPU slot
    |
    v
upload only new object
    |
    v
Object joins GPU simulation
```

### Parameter/control change

```text
CPU event
    |
    +--> update specific GPU field/range
```

### Debug/measurement

```text
GPU state
    |
    v
fence
    |
    v
mappedPtr
    |
    v
CPU observation
```

---

## 21. Data movement model

Architecture mục tiêu:

```text
                  CPU
                   |
       initialization/control
                   |
                   v
              GPU buffers
                   |
          +--------+---------+
          |                  |
          v                  v
     GPU simulation      GPU rendering
          |
          v
    Current <-> Next
          |
          |
          +---- checkpoint ----> CPU mappedPtr
```

Không có runtime loop:

```text
GPU state
   -> CPU full object array
   -> CPU update
   -> GPU full upload
```

mỗi frame.

---

## 22. Memory scaling

Theo data layout mục tiêu:

```text
CurrentState        32 B/object
NextState           32 B/object
ObjectPhysical      16 B/object
ObjectDerived       16 B/object
ObjectControl       16 B/object
ObjectAcceleration  16 B/object
```

Tổng:

\[
128\ \text{bytes/object}
\]

Nếu bổ sung:

```text
ObjectRender = 16 B/object
```

thì:

\[
144\ \text{bytes/object}
\]

Ví dụ, nếu chỉ xét các buffers này:

```text
N = 1,000,000
```

thì khoảng:

\[
144\times10^6 \approx 144\text{ MB}
\]

per-object allocation theo data layout trên, chưa bao gồm grid/field buffers, OpenGL resources và allocator/driver overhead.

Đây là phép tính lý thuyết từ struct layout, không phải VRAM measurement.

---

## 23. Performance model

Tách riêng các thành phần:

\[
T_{frame}
=
T_{CPU-control}
+
T_{GPU-compute}
+
T_{GPU-render}
+
T_{sync}
\]

Trong GPU-resident mode, mục tiêu là giảm:

\[
T_{CPU\rightarrow GPU\ full\ upload}
\]

và giảm CPU-side per-object rendering overhead.

Không giả định rằng GPU residency tự động làm Direct N-body thành `O(N)`. Solver complexity vẫn là:

\[
O(N^2)
\]

cho Direct N-body.

Các kỹ thuật Mass Accumulation, Gaussian Blur và Dynamic Grid/Field LOD mới là các hướng thay đổi computational workload/approximation được đề cương đặt ra.

---

## 24. Scalability implications

Khi tăng N:

### CPU upload

Initial:

\[
O(N)
\]

Object event:

\[
O(1)
\]

cho một object.

Normal frame:

\[
O(0)
\]

đối với full object upload.

### GPU Direct N-body

\[
O(N^2)
\]

### GPU rendering

Với instancing, số draw submissions không tăng tuyến tính theo N như CPU per-object draw-call loop; workload vertex/rasterization vẫn phụ thuộc N.

---

## 25. Validation modes

Nên tách ít nhất hai mode:

### Performance mode

```text
GPU simulation
GPU rendering
no full CPU state observation
```

Dùng để đo:

```text
Frame Time
FPS
GPU execution time
CPU overhead
RAM
VRAM
```

### Validation mode

```text
GPU simulation
periodic CPU observation
mappedPtr + synchronization
comparison with baseline/reference
```

Dùng để đo:

```text
acceleration error
position error
velocity error
energy drift
```

Không gộp chi phí validation/readback vào mọi frame của performance benchmark nếu mục tiêu là đo GPU-resident runtime path.

---

## 26. Compatibility với đề cương KLTN

Kiến trúc này trực tiếp hỗ trợ các phần đã có trong đề cương:

### Object State Management / GPU buffer management

Đề cương đặt mục tiêu hoàn thiện Object State Management, GPU buffer management, SSBO/data layout và Compute Shader execution path.

Architecture này hiện thực hóa các thành phần đó bằng:

```text
CurrentState
NextState
ObjectPhysical
ObjectDerived
ObjectControl
ObjectAcceleration
ObjectRender
GpuMemoryManager
```

### Baseline

`src/gravity_sim_3Dgrid_baseline.cpp` tiếp tục là baseline/reference độc lập.

### Accuracy

GPU state được kiểm tra tại checkpoint với reference/baseline bằng:

```text
acceleration error
field/state error
position error
velocity error
energy drift
```

### Ablation

Cùng state contract có thể được tái sử dụng khi thay solver:

```text
Direct N-body
Mass Accumulation
Mass + Gaussian Blur
Mass + Gaussian Blur + Dynamic LOD
```

### Scalability

Có thể benchmark theo:

```text
N
G
frame time
GPU time
CPU overhead
RAM / VRAM
```

và tách validation overhead khỏi normal runtime path.

---

## 27. Relationship với methodology

Luồng nghiên cứu:

```text
Baseline
    |
    v
Profile
    |
    v
GPU memory/data path
    |
    v
GPU compute
    |
    v
GPU state transition
    |
    v
GPU rendering
    |
    v
Validation
    |
    v
Benchmark
    |
    v
Ablation / Scalability
```

Mỗi giai đoạn có invariant riêng, giảm số lượng thay đổi đồng thời.

---

## 28. Implementation roadmap

Status note: the roadmap below uses repository-wide architecture stage numbers. GPU object rendering (stage 3 here) is the Phase 2 rendering milestone and is implemented in this revision.

### Phase 1 — Complete GPU Memory — **Implemented and tested**

Mục tiêu:

```text
Object -> GPU buffers
```

Công việc:

- Hoàn thiện buffer roles.
- Hoàn thiện initial materialization.
- Hoàn thiện per-object update.
- Hoàn thiện capacity management.
- Hoàn thiện CPU access/mapped pointer semantics.
- Xác định synchronization contract.

Verification:

```text
CPU Object fields
=
GPU mapped fields
```

### Phase 2 — Real CurrentState -> NextState transition — **Implemented and tested**

Mục tiêu:

```text
GPU owns position/velocity evolution
```

Công việc:

- GPU N-body.
- GPU integration.
- `CurrentState -> NextState`.
- Memory barriers.
- ping-pong swap.
- Remove full `UploadObjectState()` from normal frame loop.

Verification:

```text
GPU trajectory
vs
baseline trajectory
```

### Phase 3 — GPU Object Rendering — **Implemented and tested; rendering Phase 2**

Mục tiêu:

```text
GPU CurrentState -> screen
```

Công việc đã triển khai:

- Shared unit-sphere VAO/VBO and one instanced draw batch.
- `ObjectRender` color and `ObjectDerived` visual radius.
- Object vertex shader reading active `CurrentState` directly.
- No per-object CPU draw loop or mapped position read in normal rendering.

Verification — **tested** in the hidden OpenGL 4.4 framebuffer test for stale CPU positions, GPU scale/color, zero/one/multiple instances, ping-pong swaps, pause/resume, invalid slot rejection, and appended objects. Interactive camera inspection and large-scene comparison — **not tested**.

```text
GPU-resident position correctness
+
instanced draw performance
```

### Phase 4 — Optimization Algorithms — **Deferred**

Sau infrastructure:

```text
Direct N-body
    |
    v
Mass Accumulation
    |
    v
Gaussian Blur
    |
    v
Dynamic Grid / Field LOD
```

Các optimization algorithm được đánh giá riêng bằng ablation study.

---

## 29. Invariants cần giữ

### State invariant

```text
CurrentState = current simulation timestep
NextState    = next simulation timestep destination
```

### Ownership invariant

```text
GPU owns runtime position/velocity
```

### Upload invariant

```text
No full-state upload every frame
```

### Lifecycle invariant

```text
new object -> only new object/range uploaded
```

### Rendering invariant

```text
GPU renderer reads GPU-resident state
```

### Validation invariant

```text
CPU observation does not become runtime simulation authority
```

### Baseline invariant

```text
baseline.cpp remains independent
```

---

## 30. Architecture summary

Kiến trúc mục tiêu:

```text
CPU
├── Object creation
├── Input/control
├── Object lifecycle
├── Initial upload
└── Optional measurement/debug
          |
          v
GPU-RESIDENT OBJECT RUNTIME
├── CurrentState
├── NextState
├── ObjectPhysical
├── ObjectDerived
├── ObjectAcceleration
├── ObjectControl
└── ObjectRender
          |
          +--> GPU N-body
          |       |
          |       v
          |   NextState
          |       |
          |     swap
          |       |
          |       v
          |   CurrentState
          |
          +--> GPU Grid Compute -> DeformedGrid -> gridVAO (direct vertex input)
          |
          +--> GPU Instanced Rendering
          |
          +--> checkpoint -> mappedPtr -> CPU
```

Core rule:

\[
\boxed{
CPU\ initialization/control
\rightarrow
GPU\ resident\ state
\rightarrow
GPU\ compute
\rightarrow
GPU\ render
}
\]

và riêng validation:

\[
\boxed{
GPU\ state
\rightarrow
synchronization
\rightarrow
mappedPtr
\rightarrow
CPU\ observation
}
\]

Baseline:

\[
\boxed{
gravity\_sim\_3Dgrid\_baseline.cpp
\rightarrow
benchmark/reference
}
\]

được giữ độc lập, không trở thành CPU mirror của GPU runtime.

---

## 31. Source basis

### Project

- Repository: `https://github.com/quocthaile/gravity_sim`
- Branch: `feat/gpu-memory`
- `src/gravity_sim_3Dgrid.cpp`
- `src/gravity_sim_3Dgrid_function.cpp`
- `include/gravity_sim_3Dgrid_function.hpp`
- `include/gpu_memory_manager.hpp`
- `src/gpu_memory_manager.cpp`
- `src/gravity_sim_3Dgrid_baseline.cpp`

### Thesis proposal

- `DECUONGKLTN.pdf`
- Relevant sections:
  - Objectives and methodology.
  - Object State Management / GPU buffer management.
  - SSBO/data layout and Compute Shader execution.
  - Direct N-body, Mass Accumulation, Gaussian Blur, Dynamic Grid/Field LOD.
  - Numerical error and validation.
  - Benchmark, ablation study and scalability.

This document describes the target architecture and distinguishes it from the currently implemented transitional state. Performance or accuracy conclusions remain experimental until measured.
