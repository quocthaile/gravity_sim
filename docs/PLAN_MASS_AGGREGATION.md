# Kế hoạch mass aggregation cho deformed grid

## 1. Mục tiêu và phạm vi

Xây dựng octree trên GPU để gom khối lượng các object theo vị trí, rồi dùng
mass và tâm khối lượng của các node để tính độ biến dạng của lưới (`DeformedGrid`).
Cây được xây lại theo state và mass hiện hành để biến dạng phản ánh việc object
di chuyển hoặc thay đổi khối lượng.

Octree và aggregation **chỉ phục vụ tính deformed grid**. Không dùng aggregate
để tính gia tốc, lực hấp dẫn hoặc cập nhật vị trí; không triển khai Stochastic
Barnes–Hut hay Barnes–Hut traversal cho các bước đó. Giữ nguyên đường tính lực
trực tiếp hiện tại trong [`nbody.comp`](../shaders/nbody.comp) và integration
trong [`integrate.comp`](../shaders/integrate.comp). Earth vẫn là tâm/tham chiếu
của tính toán deformed grid; lực hấp dẫn giữa Earth và các object tiếp tục được
tính trực tiếp theo từng cặp như pipeline hiện tại, theo cả hai chiều. Không
được thay Earth hay các object bằng một khối aggregate trong mô phỏng lực.

Các nguyên tắc đã chốt:

- CPU quản lý cấu hình, cấp phát và điều phối; GPU xây cây và tính aggregate từ
  state hiện hành.
- Không đọc toàn bộ vị trí về CPU mỗi frame để xây/cập nhật cây.
- Dùng binding 6, hiện là role `MassField` dự phòng, cho `NodeAggregate` nếu
  kiểm tra layout và vòng đời buffer xác nhận phù hợp. Node aggregate chỉ được
  shader deformed-grid đọc.
- Giữ nguyên chính xác hợp đồng lực, collision, integration, ping-pong state và
  vị trí của Earth trong mô phỏng. Không sửa
  [`gravity_sim_3Dgrid_baseline.cpp`](../src/gravity_sim_3Dgrid_baseline.cpp).
- Không triển khai CPU tree traversal; chỉ dùng phép tính tham chiếu đơn giản
  trong test để xác minh aggregate và deformed grid.

## 2. Căn cứ trong mã nguồn và ranh giới tích hợp

| Thành phần | Hiện tại | Hướng thay đổi |
|---|---|---|
| State | `CurrentState`/`NextState` nằm trong GPU buffer và được swap sau integration. | Xây lại cây từ snapshot `CurrentState` cùng mass hiện hành; không đọc toàn bộ state về CPU. |
| Lực hấp dẫn | [`nbody.comp`](../shaders/nbody.comp) tính trực tiếp từng cặp object, bao gồm cả Earth và các object đang hoạt động. | Giữ nguyên direct pairwise force; không đọc `NodeAggregate`, không đổi gia tốc hoặc vị trí theo cây. |
| Integration | [`integrate.comp`](../shaders/integrate.comp) áp dụng acceleration/collision và ghi `NextState`. | Giữ nguyên công thức, thứ tự state transition và ping-pong. |
| Lưới | [`grid.comp`](../shaders/grid.comp) hiện lặp qua mọi object cho mỗi vertex, dùng `radiusRs` và khoảng cách để tính displacement theo trục Y. | Dùng aggregate của octree cho tính displacement; chốt quan hệ mass-to-deformation và xác minh kết quả với phép tính direct. |
| Bộ nhớ | [`gpu_memory_manager.hpp`](../include/gpu_memory_manager.hpp) quản lý SSBO theo role/capacity; binding 6 đang dự phòng cho `MassField`. | Có thể đổi role binding 6 thành `NodeAggregate`; bổ sung topology, object index, scratch và metadata theo giới hạn thiết bị. |
| Baseline | `gravity_sim_3Dgrid_baseline.cpp` là chương trình CPU riêng. | Giữ nguyên file; chỉ dùng khi cần đối chiếu hành vi mô phỏng hiện có. |

`spatial_partition.comp` là prototype chưa được nối vào pipeline; không xem nó
là octree builder hoàn chỉnh. Kiểm tra `ComputePipeline` và
`RunObjectStateSync` trong [`gravity_sim_3Dgrid_function.cpp`](../src/gravity_sim_3Dgrid_function.cpp)
để tích hợp các dispatch mới mà không thay đổi thứ tự tính lực và integration.
Các shader mới phải được thêm vào tạo program, required uniforms, CMake và bước
copy shader cạnh executable.

## 3. Octree và mass aggregation

### 3.1. Dữ liệu node

Ưu tiên octree tuyến tính trong SSBO, dùng index/offset thay con trỏ CPU. Mỗi
node cần bounds/kích thước, liên kết node con, thông tin leaf và aggregate mass
cùng COM. Buffer index ánh xạ thành viên leaf về đúng slot trong
`CurrentState`/`ObjectPhysical`; không giả định thứ tự không gian trùng thứ tự
object.

Binding 6 hiện là role `MassField` dự phòng, chưa allocate; không phải buffer
mass aggregation đang hoạt động. Đổi thành `NodeAggregate` chỉ sau khi kiểm tra
mọi consumer và layout CPU/GLSL. `spatial_partition.comp` cũng khai báo binding
6 cho `ObjectCellID`, nhưng chưa nối vào pipeline; không dùng đồng thời hai ý
nghĩa trong cùng pass. Nếu cần cả hai, bố trí binding riêng hoặc rebind theo
pass với role/vòng đời rõ ràng. Binding aggregate không thay thế topology,
object index hay metadata.

CPU cấp phát theo capacity; GPU cấp node trong buffer đã cấp phát, dùng counter
nguyên tử hoặc cơ chế tương đương để tránh index trùng và ghi ngoài capacity.
Định nghĩa layout `std430`, alignment, logical count/capacity và kiểm tra giới
hạn SSBO/binding của thiết bị.

### 3.2. Xây lại cây theo state mới

Mỗi lần cần cập nhật deformed grid:

1. Reset metadata/counter và xác định bounds bao phủ các object hợp lệ theo hệ
   tọa độ lấy Earth làm tâm tham chiếu; không thay đổi tọa độ vật lý trong
   `CurrentState`.
2. Xây topology và object index từ vị trí hiện hành, mass từ
   `ObjectPhysical`; loại source `initializing` nhất quán với pipeline hiện tại.
3. Bảo đảm mỗi object hợp lệ xuất hiện đúng một lần. Xử lý rõ object ngoài
   bounds, vị trí trùng nhau, leaf vượt capacity, độ sâu tối đa và cây mất cân
   bằng.
4. Tính aggregate của leaf rồi các node cha từ dưới lên. Dùng các dispatch có
   dependency và `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)` phù hợp; không
   giả định đồng bộ toàn device giữa workgroup trong cùng dispatch.

Trong MVP, xây lại cây thay vì cập nhật topology tăng dần. Khi object di chuyển
hoặc mass đổi, phải tính lại aggregate trước lần dùng deformed grid tiếp theo.
Không tái cấp phát SSBO mỗi frame; tái sử dụng buffer theo capacity.

### 3.3. Mass và COM

Với object trong leaf:

```text
M = sum(m_i)
COM = sum(m_i * position_i) / M
```

Với node trong:

```text
M_node = sum(M_child)
COM_node = sum(M_child * COM_child) / M_node
```

Node rỗng hoặc mass không hợp lệ không được chia cho 0 hay truyền NaN/Inf sang
shader lưới. Chốt chính sách với mass không dương; không âm thầm loại object
hợp lệ theo semantics hiện hành.

## 4. Dùng aggregate để tính deformed grid

### 4.1. Phạm vi tính toán

`NodeAggregate` chỉ là đầu vào cho phép tính displacement của các vertex/cell
lưới trong [`grid.comp`](../shaders/grid.comp). Với mỗi điểm lưới, truy vấn cây
để lấy ảnh hưởng từ mass phân bố trong không gian; dùng bounds, COM và mass của
node, mở node khi cần độ chi tiết cao và xử lý trực tiếp object ở leaf. Không
traversal node nào được chuyển sang `nbody.comp`, acceleration hay integration.

Earth là tâm/tham chiếu của vùng tính deformed grid. Giữ Earth trong dữ liệu
object và trong aggregate như một nguồn khối lượng; object khác cũng đóng góp
theo vị trí của chúng. Không chuyển COM toàn hệ thành tâm mô phỏng và không
dùng aggregate để thay đổi lực Earth-object. Lực giữa Earth và từng object
tiếp tục được tính theo cặp bằng đường direct hiện tại.

### 4.2. Quy tắc deformation cần xác định

`grid.comp` hiện cộng displacement theo `radiusRs` từng object và khoảng cách
tới điểm lưới. Trước khi thay vòng lặp này, phải chốt:

- Cách quy đổi mass aggregate thành đại lượng deformation, đơn vị và dấu của
  displacement. Không mặc định rằng công thức hiện tại tương đương với một
  công thức theo mass nếu chưa chứng minh quan hệ `radiusRs`/mass.
- Cách node aggregate đóng góp tại điểm lưới và tiêu chí mở node (bounds,
  khoảng cách, sai số). Khi mở đến leaf, kết quả phải đối chiếu được với phép
  tính direct cùng input.
- Cách giữ Earth làm tâm/tham chiếu, bao gồm trường hợp Earth di chuyển hoặc
  không có trong tập object hợp lệ.
- Hành vi ở khoảng cách gần, node rỗng, bounds không hợp lệ và mass không dương.

Không đổi dữ liệu vertex khác ngoài displacement đã định nghĩa; không đưa lực
hay cập nhật vị trí object vào shader lưới. Nếu sai số deformation không đạt
ngưỡng, mở node sâu hơn hoặc dùng direct contribution cho vùng đó; không dùng
stochastic sampling.

### 4.3. Luồng dự kiến

```text
GPU: nbody.comp direct pairwise -> ObjectAcceleration + ObjectCollision
GPU: integrate.comp -> NextState
GPU: kiểm tra lỗi bước và swap CurrentState/NextState như hiện tại
GPU: reset/build octree từ state + mass mới
GPU: bottom-up mass/COM aggregation
GPU: grid.comp truy vấn aggregate -> DeformedGrid
GPU: render grid và objects
```

Thứ tự cập nhật grid có thể phù hợp với lịch render hiện hành, miễn grid dùng
snapshot mới nhất đã được swap và các barrier bảo đảm dữ liệu aggregate sẵn
sàng. Việc build tree không được làm thay đổi kết quả hoặc thứ tự của bước
force/integration.

### 4.4. Overflow và trạng thái lỗi

GPU ghi cờ khi node/scratch capacity bị vượt hoặc dữ liệu không thể biểu diễn;
không ghi ngoài buffer và không dùng cây thiếu node như cây hợp lệ. Nếu build
lỗi, không chạy truy vấn aggregate lỗi. Giữ deformed grid hợp lệ gần nhất hoặc
báo lỗi rõ ràng theo chính sách được chốt; không chặn hay cập nhật sai
`CurrentState`/`NextState` chỉ vì lỗi phần lưới.

CPU nhận thông báo và báo lỗi rõ ràng, hoặc tăng capacity rồi thử lại theo
chính sách; kiểm tra giới hạn thiết bị và lỗi cấp phát. Nếu đọc status bất đồng
bộ, GPU vẫn phải ngăn grid pass dùng cây lỗi trước khi CPU nhận thông báo. Việc
đọc status không được biến thành readback toàn bộ vị trí mỗi frame.

## 5. Kiểm thử và so sánh

### 5.1. Kiểm thử aggregate

- Dùng fixture có Earth, nhiều object, mass khác nhau, object khởi tạo và các
  vị trí biên để so sánh mass/COM của node với phép tính tham chiếu trực tiếp.
- Xác minh mỗi object hợp lệ có đúng một vị trí trong cây; kiểm tra node rỗng,
  trùng vị trí, bounds, độ sâu và overflow.
- Di chuyển object hoặc thay mass giữa hai lần chạy; xác minh topology/aggregate
  được cập nhật theo state mới, không dùng dữ liệu stale.
- Đối chiếu layout SSBO CPU/GLSL, logical count, capacity và barrier.

Không cần triển khai CPU octree hay CPU tree traversal. Tham chiếu trực tiếp
trong test chỉ dùng để xác minh kết quả.

### 5.2. Kiểm thử deformed grid

So sánh `grid.comp` dùng aggregate với phép tính direct trên cùng snapshot,
mass, vị trí Earth và vertex lưới. Khi mở node đến leaf, kết quả displacement
phải khớp phép tính direct trong tolerance. Với tiêu chí mở node được chọn,
báo sai số tuyệt đối/tương đối, RMS, percentile và max trên nhiều phân bố
object, đồng thời kiểm tra Earth-only, object-only, Earth cùng objects, lưới
gần nguồn và trường hợp object di chuyển.

Kiểm thử riêng rằng lỗi build/overflow không khiến grid dùng aggregate thiếu
hoặc làm sai state vật lý. Không dùng sai số acceleration hay trajectory để
đánh giá octree: octree không tham gia tính lực.

### 5.3. Bảo toàn lực và cập nhật vị trí

Xác nhận `nbody.comp` vẫn tính trực tiếp từng cặp source/target hợp lệ, gồm lực
object lên Earth và Earth lên object, theo semantics hiện hành của
`initializing`, `distance <= 0`, epsilon, đơn vị và collision. Xác nhận
`integrate.comp`, dt, collision và ping-pong state không đổi bởi việc thêm
octree. Không thêm stochastic/deterministic Barnes–Hut cho acceleration hoặc
position update; không sửa baseline CPU.

Baseline CPU có khác biệt về công thức, thứ tự cập nhật và lấy mẫu như mô tả
trong tài liệu hiện hành; chỉ dùng để đối chiếu mô phỏng nếu cần, không dùng làm
oracle chính xác cho aggregate hoặc displacement.

## 6. Lộ trình triển khai

### Giai đoạn 0 — Hợp đồng deformation và dữ liệu

1. Ghi fixture/test direct cho aggregate và phép tính `grid.comp` hiện tại.
2. Chốt hệ tọa độ Earth-centered, bounds, policy cho Earth không hợp lệ, mass,
   depth/leaf capacity và hành vi overflow.
3. Chốt quan hệ mass-to-deformation, tiêu chí mở node, tolerance và layout
   CPU/GLSL; kiểm tra hardware limits, role và binding.

**Đầu ra:** hợp đồng dữ liệu và phép tính deformed grid có thể kiểm chứng.

### Giai đoạn 1 — GPU octree và aggregation

1. Đổi role dự phòng `MassField` ở binding 6 thành `NodeAggregate` nếu layout/
   consumer đã được xác nhận; thêm buffer topology, object indices, scratch và
   metadata.
2. Xây lại octree từ state/mass hiện hành, với kiểm tra ownership, bounds và
   capacity.
3. Tính mass/COM bottom-up, đặt barrier hợp lệ và phát hiện overflow.
4. Đối chiếu aggregate chọn lọc với phép tính tham chiếu trực tiếp.

**Đầu ra:** hierarchy GPU và aggregate được kiểm chứng, chưa thay đổi lực hay
integration.

### Giai đoạn 2 — Tích hợp vào deformed grid

1. Sửa `grid.comp` để truy vấn aggregate và tính displacement theo hợp đồng đã
   chốt.
2. Tích hợp dispatch/build/aggregation trước grid pass; bảo đảm grid dùng state
   mới và có buffer/barrier phù hợp.
3. So sánh deformed grid với phép tính direct; kiểm thử sai số, chuyển động
   object, thay mass và Earth làm tâm tham chiếu.
4. Kiểm thử build lỗi/overflow để grid không đọc cây thiếu; xác nhận
   `nbody.comp`, `integrate.comp` và state transition giữ nguyên.

**Đầu ra:** mass aggregation điều khiển deformed grid; mô phỏng lực và chuyển
động vẫn dùng đường hiện tại.

### Giai đoạn 3 — Đánh giá và tài liệu

1. Đo riêng build, aggregation, grid query và tổng chi phí cập nhật deformed
   grid trên nhiều `N`/phân bố; báo cáo trường hợp cây chậm hơn direct.
2. Chọn tiêu chí mở node theo sai số deformation/hiệu năng đo được.
3. Cập nhật tài liệu kiến trúc sau khi code hoạt động. Không mở rộng scope sang
   stochastic sampling hay tree-based force/integration.

## 7. Tiêu chí nghiệm thu

- Mỗi object hợp lệ xuất hiện đúng một lần trong cây; mọi node/index/bounds hợp
  lệ và nằm trong capacity.
- Mass và COM khớp phép tính tham chiếu trong tolerance; node rỗng/không hợp lệ
  không gây NaN, Inf hoặc chia cho 0.
- Khi mở đến leaf, displacement của deformed grid khớp direct test; sai số khi
  dùng aggregate được đo và báo cáo trên nhiều input.
- Cây/aggregate được xây lại hoặc cập nhật đúng khi vị trí hay mass thay đổi;
  grid không dùng aggregate stale.
- Earth tiếp tục là tâm/tham chiếu deformed grid và các object cùng Earth tiếp
  tục tham gia tính lực trực tiếp từng cặp theo semantics hiện hành.
- Octree chỉ được đọc bởi đường tính deformed grid; không dùng để tính
  acceleration, force hay cập nhật vị trí. Không triển khai Stochastic
  Barnes–Hut cho các bước này.
- Overflow, bounds không hợp lệ, vượt hardware limits và cấp phát thất bại
  được phát hiện; không ghi ngoài buffer hoặc âm thầm dùng aggregate thiếu.
- Không đọc toàn bộ vị trí về CPU mỗi frame trong pipeline octree production.
- Baseline CPU không bị sửa. Benchmark ghi rõ thiết bị, đầu vào/seed, số object,
  tiêu chí mở node, sai số deformation và thời gian từng stage.

## 8. Các quyết định kỹ thuật cần chốt

1. Bounds octree theo hệ tọa độ Earth-centered; cách xử lý Earth di chuyển hoặc
   không hiện diện/đang khởi tạo.
2. Biểu diễn cây, phương pháp build/sort/group, độ sâu tối đa và leaf capacity.
3. Công thức mass-to-deformation có đơn vị rõ ràng và quan hệ với displacement
   hiện tại dựa trên `radiusRs`.
4. Quy tắc mở node theo bounds/khoảng cách và tolerance của displacement.
5. Capacity node/scratch, phát hiện overflow, thông báo, retry/dừng grid an
   toàn và giới hạn phần cứng.
6. Binding/role, layout, barrier và chính sách mass không dương.
7. Lịch build/aggregation/grid so với integration để luôn dùng snapshot mới
   mà không can thiệp vào lực, collision hoặc state transition.
