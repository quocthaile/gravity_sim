# KẾ HOẠCH TÍCH HỢP GPU OCTREE / MASS AGGREGATION / BARNES–HUT

> **Tài liệu giao việc cho GitHub Copilot Agent**  
> **Mục tiêu:** thay phần tính tương tác hấp dẫn N-body trực tiếp bằng GPU Barnes–Hut sử dụng Octree và Mass Aggregation; giữ nguyên mô hình biến dạng lưới dựa trên bán kính Schwarzschild.

---

## 1. Mục tiêu và nguyên tắc triển khai

### 1.1. Mục tiêu

Tích hợp một pipeline tính gia tốc hấp dẫn gần đúng trên GPU theo thuật toán **Barnes–Hut**. Cây không gian **Octree** phân chia không gian 3D thành các vùng nhỏ hơn. Giai đoạn **Mass Aggregation** tổng hợp khối lượng và tâm khối của các node. Sau đó, mỗi vật thể duyệt cây để tính gia tốc: tương tác với vật thể ở xa có thể được xấp xỉ bằng một node tổng hợp, còn những vùng ở gần phải tiếp tục được phân rã để tính chi tiết.

Kết quả của Barnes–Hut phải được ghi vào `ObjectAcceleration`, sau đó pipeline tích phân hiện tại sử dụng gia tốc này để cập nhật trạng thái vật thể.

**Không thay đổi mô hình biến dạng lưới hiện tại.** `grid.comp` tiếp tục sử dụng vị trí vật thể trong `CurrentState` và bán kính Schwarzschild trong `ObjectDerived` để tính độ lõm. Octree/Mass Aggregation không được dùng thay thế công thức Schwarzschild trong bước này.

### 1.2. Nguyên tắc bắt buộc

1. Đọc và hiểu các file hiện có trước khi sửa. Không viết lại toàn bộ `gravity_sim_3Dgrid.cpp`, `gravity_sim_3Dgrid_function.cpp` hoặc `GpuMemoryManager` nếu chỉ cần mở rộng chúng.
2. Thực hiện theo từng giai đoạn nhỏ. Sau mỗi giai đoạn phải build và kiểm tra trước khi chuyển sang giai đoạn tiếp theo.
3. `CurrentState` là nguồn dữ liệu chuẩn cho vị trí và vận tốc đang mô phỏng trên GPU. Không dùng bản sao dữ liệu CPU cũ để xây cây trong mỗi frame.
4. Octree và dữ liệu Mass Aggregation phải được tạo từ đúng phiên bản `CurrentState` mà lần tính Barnes–Hut đang sử dụng.
5. Giữ nguyên thứ tự và ý nghĩa cập nhật trạng thái `CurrentState` / `NextState`, trừ phần cần thiết để chèn các compute pass mới.
6. Không ghim Earth tại gốc tọa độ, không mặc định Earth là root của Octree và không thay đổi quy tắc chuyển động của các vật thể trong phạm vi công việc này.
7. Không thay đổi công thức Schwarzschild, đơn vị đo hoặc cách cộng độ lõm của `grid.comp` trong lúc tích hợp Barnes–Hut.
8. Không tự ý thay đổi timestep hoặc thuật toán tích phân hiện có. Hệ số `u_dt` và `u_dt` trong `integrate.comp` phải được giữ nguyên trong lần tích hợp đầu tiên; ghi nhận riêng nếu phát hiện vấn đề cần xem xét.
9. Không tuyên bố hiệu năng đã cải thiện chỉ dựa trên việc có Octree. Phải đo thời gian GPU và kiểm tra sai số so với Direct N-body. Tạo functional programming test để so sánh kết quả giữa hai thuật toán trên cùng snapshot, không xen kẽ tích phân giữa hai lần đo. Mã nguồn    `numerical_error.cpp` dùng để kiểm tra sai số giữa `gravity_sim_3Dgrid_baseline.cpp` và các file liên quan với `gravity_sim_3Dgrid.cpp`, tinh chỉnh `numerical_error.cpp` để gọi Barnes–Hut và Direct N-body trên cùng snapshot. Dùng mã nguồn này để tính CPU time, GPU time, fps, frame time, thời gian compute, sai số gia tốc/vị trí, lực hấp dẫn, và các sai số có sẵn trong file đó. Báo cáo kết quả kiểm thử trong tài liệu kiến trúc. Hỏi lại khi chưa hiểu yêu cầu này.
10. Trong báo cáo kết quả kiểm thử, nêu rõ số lượng vật thể, cấu hình phân bố, giá trị `theta`, thiết bị GPU, phương pháp đo và các kết quả đo từ nguyên tắc `9`. Không đưa ra kết luận về hiệu năng hoặc độ đúng nếu chưa có dữ liệu thực tế.
11. Không thay đổi logic va chạm hiện tại trong lần tích hợp đầu tiên. Barnes–Hut không đủ thông tin để xác định va chạm chính xác.

---

## 2. Hiện trạng kiến trúc cần giữ tương thích

Các thông tin dưới đây là điểm xuất phát từ cấu trúc hiện tại. Copilot Agent phải xác nhận lại trong working tree trước khi sửa, vì mã nguồn có thể đã thay đổi.

### 2.1. Các file chính

| File | Vai trò hiện tại / vai trò cần mở rộng |
|---|---|
| `src/gravity_sim_3Dgrid.cpp` | Khởi tạo đối tượng, tài nguyên render và điều phối các bước trong main loop. Chèn các compute pass xây cây, tổng hợp khối lượng và Barnes–Hut vào đúng vị trí. |
| `src/gravity_sim_3Dgrid_function.cpp` | Chứa các hàm khởi tạo pipeline, đồng bộ trạng thái và gọi compute. Mở rộng tại đây nếu phù hợp với cách tổ chức hiện có. |
| `include/gravity_sim_3Dgrid_function.hpp` | Khai báo các hàm pipeline và kiểu dữ liệu dùng chung nếu cần. |
| `include/gpu_memory_manager.hpp` | Khai báo `BufferRole`, binding point và quản lý buffer. |
| `src/gpu_memory_manager.cpp` | Cấp phát, resize, bind và quản lý vòng đời SSBO. |
| `shaders/nbody.comp` | Hiện tính tương tác hấp dẫn trực tiếp giữa các vật thể. Sẽ được thay đổi hoặc tách thành shader Barnes–Hut. |
| `shaders/integrate.comp` | Đọc `CurrentState` và `ObjectAcceleration`, ghi trạng thái kế tiếp vào `NextState`. Giữ nguyên logic trong giai đoạn tích hợp đầu tiên. |
| `shaders/grid.comp` | Tính độ lõm lưới bằng bán kính Schwarzschild của từng vật thể. Giữ nguyên mô hình và công thức. |
| `docs/gravity_sim_gpu_resident_architecture.md` | Tài liệu kiến trúc GPU-resident hiện có; cập nhật khi pipeline mới ổn định. |

### 2.2. Các buffer trạng thái đang có

- `CurrentState`: vị trí và vận tốc hiện tại của các vật thể; là nguồn trạng thái runtime chính trên GPU.
- `NextState`: đích ghi trạng thái sau bước tích phân.
- `ObjectPhysical`: dữ liệu vật lý, ví dụ khối lượng và mật độ.
- `ObjectControl`: cờ trạng thái và thông tin điều khiển vật thể.
- `ObjectDerived`: dữ liệu suy ra từ thuộc tính vật lý, trong đó có bán kính và bán kính Schwarzschild.
- `ObjectAcceleration`: gia tốc hấp dẫn tổng hợp của từng vật thể.
- `ObjectCollision`: dữ liệu đếm va chạm hiện tại.
- `BaseGrid`: lưới cơ sở chưa biến dạng.
- `DeformedGrid`: lưới sau khi áp dụng độ lõm Schwarzschild.

`CurrentState` và `NextState` được hoán đổi vai trò sau bước tích phân bằng `SwapCurrentNext()`. Cần giữ đúng cơ chế hoán đổi này; không sao chép toàn bộ dữ liệu chỉ để đổi vai trò nếu manager hiện tại đã hỗ trợ swap buffer.

### 2.3. Luồng GPU hiện tại

Hiện tại `nbody.comp` thực hiện Direct N-body: mỗi vật thể mục tiêu duyệt qua các vật thể còn lại để tính tổng gia tốc. Độ phức tạp tính toán xấp xỉ `O(N²)`. Shader cũng đang xử lý thông tin chồng lấn/va chạm theo từng cặp vật thể.

`integrate.comp` đọc gia tốc, cập nhật vận tốc và vị trí, rồi ghi vào `NextState`. Sau đó chương trình gọi `SwapCurrentNext()`.

`grid.comp` đọc vị trí mới từ `CurrentState`, đọc bán kính Schwarzschild từ `ObjectDerived`, đọc `BaseGrid` và ghi kết quả vào `DeformedGrid`. Shader này hiện duyệt từng vật thể cho mỗi đỉnh lưới.

---

## 3. Luồng xử lý đích

### 3.1. Khởi tạo một lần

1. **CPU tạo danh sách vật thể.** Khởi tạo các thuộc tính cần thiết như vị trí, vận tốc, khối lượng, mật độ, bán kính và bán kính Schwarzschild.
2. **Khởi tạo và cấp phát SSBO.** Cấp phát các buffer trạng thái vật thể hiện có và các buffer mới phục vụ Octree/Mass Aggregation.
3. **Upload dữ liệu ban đầu lên GPU.** Nạp vị trí, vận tốc và thuộc tính vật lý vào các buffer tương ứng.
4. **Khởi tạo tài nguyên lưới và render như hiện tại.** Không đưa Octree vào đường xử lý biến dạng lưới.
5. **Trước lần tính Barnes–Hut đầu tiên, phải xây Octree và Mass Aggregation từ `CurrentState` ban đầu.** Không được đọc dữ liệu cây chưa khởi tạo.

### 3.2. Luồng xử lý mỗi frame

Thứ tự logic bắt buộc:

```text
CPU: xử lý input / camera / thay đổi điều khiển vật thể
                 |
                 v
        Đồng bộ thay đổi vật thể lên GPU
                 |
                 v
       Đọc CurrentState trên GPU
                 |
                 v
       Tính bounding box của hệ vật thể
                 |
                 v
          Xây dựng GPU Octree
                 |
                 v
       GPU Mass Aggregation (bottom-up)
       - tổng khối lượng mỗi node
       - tâm khối mỗi node
                 |
                 v
       GPU Barnes-Hut traversal
       - mỗi vật thể duyệt Octree
       - tính gia tốc gần đúng
       - ghi ObjectAcceleration
                 |
                 v
       GPU Integration: CurrentState -> NextState
                 |
                 v
             SwapCurrentNext
                 |
                 v
       GPU Schwarzschild Grid Deformation
       - đọc CurrentState sau swap
       - đọc ObjectDerived (Schwarzschild radius)
       - đọc BaseGrid
       - ghi DeformedGrid
                 |
                 v
              Draw Grid
                 |
                 v
             Draw Objects
                 |
                 v
             End Frame
```

### 3.3. Vai trò của từng giai đoạn

**A. Đồng bộ thay đổi vật thể**  
Giữ cơ chế đồng bộ hiện tại. Những thay đổi do CPU thực hiện đối với trạng thái điều khiển, khởi tạo hoặc thuộc tính vật lý phải được cập nhật lên GPU trước khi xây cây và tính gia tốc. Không upload lại toàn bộ vị trí/vận tốc từ CPU mỗi frame nếu GPU đang là nơi quản lý trạng thái runtime.

**B. Bounding Box**  
Tính miền không gian chứa tất cả vật thể hợp lệ dựa trên vị trí trong `CurrentState`. Bounding box là miền bao không gian của toàn hệ, không phải vị trí của Earth. Với Octree 3D, cần chuyển miền bao thành miền gốc phù hợp để chia thành 8 octant; xử lý trường hợp một hoặc nhiều chiều có kích thước bằng 0 bằng padding hợp lý.

**C. Octree Construction**  
Xây cấu trúc cây 8 nhánh từ bounding box. Mỗi node cần có đủ thông tin để xác định vùng không gian, quan hệ node cha/con và phạm vi hoặc danh sách vật thể thuộc node. Cần xác định rõ cách biểu diễn node, cách cấp phát node trên GPU, giới hạn số node và cách xử lý trường hợp vượt dung lượng.

**D. Mass Aggregation**  
Tính tổng khối lượng và tâm khối cho từng node theo hướng bottom-up, nghĩa là tổng hợp từ node lá lên node cha.

Với node có các vật thể hoặc node con có khối lượng `m_k` và tâm khối `c_k`:

\[
M = \sum_k m_k
\]

\[
\mathbf{COM} = \frac{\sum_k m_k\mathbf{c}_k}{\sum_k m_k}
\]

Nếu tổng khối lượng bằng 0, không được chia cho 0; node phải được đánh dấu rỗng hoặc có trạng thái hợp lệ tương đương.

**E. Barnes–Hut Traversal**  
Mỗi invocation xử lý một vật thể mục tiêu hoặc một nhóm vật thể theo thiết kế shader. Khi duyệt cây:

- Bỏ qua node rỗng.
- Không tính vật thể mục tiêu tương tác với chính nó.
- Nếu node đủ xa, dùng tổng khối lượng và tâm khối của node để xấp xỉ đóng góp hấp dẫn.
- Nếu node quá gần hoặc chứa vật thể mục tiêu, mở node và tiếp tục duyệt các node con để tránh tính cả khối lượng của chính vật thể mục tiêu như một nguồn khối lượng tổng hợp.
- Với node lá, xử lý các vật thể thành viên theo cách tính trực tiếp.
- Giữ quy tắc softening và quy đổi đơn vị phù hợp với solver hiện tại.

Tiêu chí mở node Barnes–Hut thường có dạng:

\[
\frac{s}{d} < \theta
\]

Trong đó `s` là kích thước node, `d` là khoảng cách từ vật thể mục tiêu đến tâm khối node, còn `theta` là tham số điều khiển mức xấp xỉ. `theta` nhỏ thường cho kết quả gần Direct N-body hơn nhưng tăng số node phải mở; `theta` lớn giảm công việc duyệt cây nhưng có thể tăng sai số. Không chọn giá trị `theta` tùy ý mà không kiểm thử sai số.

Gia tốc kết quả của mỗi vật thể phải được ghi vào đúng phần tử tương ứng trong `ObjectAcceleration` để `integrate.comp` có thể đọc.

**F. Integration và Swap**  
Giữ luồng `CurrentState` -> `NextState` -> `SwapCurrentNext()`. Chỉ thay nguồn tạo `ObjectAcceleration` từ Direct N-body sang Barnes–Hut. Trong giai đoạn đầu, không thay đổi công thức cập nhật vận tốc/vị trí, thứ tự cập nhật hoặc hệ số hiện có trong `integrate.comp`.

**G. Schwarzschild Grid Deformation**  
Chạy sau `SwapCurrentNext()` để lưới hiển thị trạng thái vị trí mới nhất. Giữ nguyên các đầu vào logic:

- `CurrentState`: vị trí mới của vật thể.
- `ObjectDerived`: bán kính Schwarzschild `r_s`.
- `BaseGrid`: vị trí các đỉnh lưới ban đầu.
- `DeformedGrid`: vị trí các đỉnh lưới sau biến dạng.

Với mỗi đỉnh lưới và mỗi vật thể, giữ cách tính khoảng cách và độ dịch chuyển hiện tại:

\[
d_i = \max\left(1000\,\lVert\mathbf{x}_i-\mathbf{x}_{grid}\rVert,\;1.0001r_{s,i}\right)
\]

\[
\Delta y = -\sum_i \frac{2.0\times10^{11}\,r_{s,i}}{d_i}
\]

Trong đó khoảng cách được quy đổi sang mét theo quy ước hiện tại của mã nguồn. Công thức trên phải được giữ theo implementation hiện có, không tự ý thay bằng thế năng tổng hợp từ `MassField`, tâm khối Octree hoặc một trường hấp dẫn mới.

**H. Render**  
Tiếp tục gọi `DrawGrid()` và `DrawObjects()` theo trình tự hiện tại.

---

## 4. Quy tắc đồng bộ thời gian và dữ liệu

Đây là điều kiện quan trọng để tránh dùng cây được xây từ trạng thái cũ.

1. Octree phải được xây từ `CurrentState` trước lần dispatch Barnes–Hut tương ứng.
2. Mass Aggregation phải hoàn tất trước khi Barnes–Hut đọc khối lượng và tâm khối node.
3. Barnes–Hut phải hoàn tất trước khi `integrate.comp` đọc `ObjectAcceleration`.
4. Integration phải hoàn tất trước `SwapCurrentNext()`.
5. Sau swap, `grid.comp` đọc `CurrentState` theo vai trò logic mới đã được manager bind lại.
6. Không cần xây lại Octree sau swap chỉ để chạy `grid.comp`, vì grid shader không đọc Octree. Octree của frame kế tiếp sẽ được xây lại từ `CurrentState` mới trước lần Barnes–Hut kế tiếp.
7. Sau mỗi compute dispatch có ghi dữ liệu mà compute pass tiếp theo đọc, phải sử dụng memory barrier phù hợp. Với các SSBO compute hiện tại, kiểm tra và duy trì `GL_SHADER_STORAGE_BARRIER_BIT` ở đúng ranh giới dispatch. Nếu có dữ liệu được dùng bởi vertex shader hoặc draw call, bổ sung barrier phù hợp với cách truy cập thực tế.
8. Khi thêm/xóa vật thể hoặc thay đổi khối lượng/vị trí từ CPU, phải bảo đảm dữ liệu GPU được cập nhật và cây được xây lại trước lần Barnes–Hut kế tiếp.

---

## 5. Định hướng SSBO binding

Các binding point hiện có cần được kiểm tra trực tiếp trong `BufferRole`, `BindingFor()`, `ComputePipeline()` và các khai báo `layout(std430, binding = ...)` của shader.

Bảng dưới đây là **đề xuất phân vai cho phần mở rộng**, không phải khẳng định rằng các buffer này đã được cấp phát:

| Binding | Vai trò đề xuất | Mục đích |
|---:|---|---|
| 0 | `CurrentState` | Vị trí/vận tốc hiện tại. |
| 1 | `NextState` | Vị trí/vận tốc sau tích phân. |
| 2 | `ObjectPhysical` | Khối lượng, mật độ và thuộc tính vật lý. |
| 3 | `ObjectControl` | Trạng thái điều khiển vật thể. |
| 4 | `BaseGrid` | Lưới cơ sở. |
| 5 | `DeformedGrid` | Lưới sau biến dạng. |
| 6 | `MassField` | Khối lượng tổng và tâm khối của các node sau Mass Aggregation. Không dùng cho grid deformation trong phương án này. |
| 7 | `OctreeNodes` | Cấu trúc node: bounding region, liên kết node con, trạng thái node và thông tin cần cho traversal. Có thể đổi tên role hiện đang mang tên `BlurFieldA` nếu phù hợp. |
| 8 | `ObjectIndexList` | Danh sách chỉ số vật thể được sắp xếp hoặc nhóm theo node lá. Có thể đổi tên role hiện đang mang tên `BlurFieldB` nếu phù hợp. |
| 9 | Để dành | Chưa sử dụng trong phương án cơ sở. Không cấp phát nếu chưa có nhu cầu cụ thể. |
| 10 | `OctreeMetadata` | Metadata phục vụ xây cây/traversal, ví dụ số node, trạng thái xây dựng, cấp phát hoặc thông tin phạm vi. Có thể đổi tên role hiện đang mang tên `LODMetadata`. |
| 11 | `ObjectLeafIndex` (tùy chọn) | Ánh xạ một vật thể tới leaf node hoặc metadata phụ trợ, chỉ thêm nếu thuật toán thực tế cần. Có thể đổi tên role hiện đang mang tên `Metrics`. |
| 12 | `ObjectDerived` | Bán kính và bán kính Schwarzschild của vật thể. |
| 13 | `ObjectAcceleration` | Gia tốc tổng hợp cho từng vật thể. |
| 14 | `ObjectRender` | Dữ liệu hiển thị vật thể. |
| 15 | `ObjectCollision` | Dữ liệu va chạm/chồng lấn. |

### Yêu cầu khi triển khai binding

- Binding point trong C++ phải trùng với `layout(std430, binding = N)` trong GLSL.
- Định nghĩa struct C++ và GLSL phải tương thích về layout, alignment, padding và kiểu dữ liệu theo `std430`.
- Không được giả định rằng role đã khai báo trong enum đồng nghĩa buffer đã được cấp phát. Kiểm tra `ComputePipeline()` và đường gọi tạo buffer thực tế.
- Chỉ cấp phát những buffer thực sự được dùng. Không tạo `AccelerationField` riêng nếu `ObjectAcceleration` hiện tại đã đảm nhiệm đúng dữ liệu gia tốc.
- Không đưa Octree/MassField vào `grid.comp` trong phương án Schwarzschild.
- Nếu thay tên role, cập nhật đồng bộ enum, `BindingFor()`, khởi tạo/cấp phát/resize, shader và tài liệu.

---

## 6. Phạm vi xử lý va chạm

Barnes–Hut dùng tổng khối lượng và tâm khối để xấp xỉ lực hấp dẫn ở xa. Hai đại lượng tổng hợp này **không đủ để xác định chính xác va chạm giữa các vật thể riêng lẻ**.

Vì vậy, trong phiên bản tích hợp đầu tiên:

1. Không dùng một Octree node để thay thế kiểm tra va chạm chi tiết.
2. Giữ logic collision hiện tại hoặc tách collision thành một pass riêng nhưng phải bảo toàn hành vi hiện có.
3. Nếu `nbody.comp` được thay thế hoàn toàn bằng shader Barnes–Hut, phải bảo đảm `ObjectCollision` vẫn được cập nhật đúng trước `integrate.comp`.
4. Chỉ tối ưu collision bằng broad phase dựa trên không gian trong một công việc riêng, có kiểm thử độc lập. Không gộp thay đổi collision vào bước xác nhận Barnes–Hut ban đầu.

---

## 7. Các giai đoạn thực hiện cho Copilot Agent

### Giai đoạn 0 — Khảo sát và chốt baseline

- Đọc các file được liệt kê ở Mục 2.
- Xác nhận chính xác binding hiện tại, cấu trúc buffer, kích thước struct và trình tự gọi compute trong main loop.
- Xác nhận cách `SwapCurrentNext()` đổi vai trò binding.
- Xác nhận cách cập nhật `ObjectAcceleration` và `ObjectCollision` hiện tại.
- Build và chạy Direct N-body hiện tại để ghi nhận baseline: thời gian compute, số vật thể, timestep/cấu hình chạy và dữ liệu vị trí mẫu.
- Không sửa thuật toán ở giai đoạn này.

**Đầu ra:** ghi chú baseline và sơ đồ pipeline hiện tại; danh sách những điểm cần sửa được xác định bằng file/hàm cụ thể.

### Giai đoạn 1 — Thiết kế cấu trúc dữ liệu Octree

- Định nghĩa node Octree và metadata cần thiết.
- Xác định cách biểu diễn 8 node con, node rỗng, node lá, node nội bộ và liên kết giữa node.
- Xác định cách lưu danh sách object index trong leaf node.
- Xác định giới hạn số node, số vật thể mỗi leaf, độ sâu tối đa và cách xử lý vị trí trùng nhau.
- Xác định chiến lược cấp phát/counter trên GPU và hành vi khi vượt capacity.
- Chốt layout C++/GLSL trước khi viết shader.

**Đầu ra:** tài liệu/struct thống nhất, không có hai định nghĩa layout khác nhau giữa CPU và GPU.

### Giai đoạn 2 — Mở rộng GPU Memory Manager và xác định cấu trúc buffer cho Octree

**Mục tiêu:** Mở rộng `GpuMemoryManager` để quản lý dữ liệu Octree và Mass Aggregation trên GPU, bảo đảm cấu trúc C++/GLSL tương thích `std430`, capacity được tính toán an toàn và các SSBO được bind chính xác.

**2.1. Phân bổ BufferRole và BindingFor()**

Sử dụng các binding sau:

| BufferRole | Binding | Element Size |
|---|---:|---:|
| MassField | 6 | 16 bytes |
| OctreeNodes | 7 | 64 bytes |
| ObjectIndexList | 8 | 4 bytes |
| OctreeMetadata | 10 | 64 bytes |
| ObjectLeafIndex | 11 | 4 bytes |

Tái sử dụng các binding hiện có: `BlurFieldA` thành `OctreeNodes`, `BlurFieldB` thành `ObjectIndexList`, `LODMetadata` thành `OctreeMetadata`, `Metrics` thành `ObjectLeafIndex`.

Giữ binding 9 (`AccelerationField`) chưa cấp phát vì chưa được sử dụng trong kiến trúc biến dạng lưới Schwarzschild hiện tại.

Không thay đổi binding 0–5 và 12–15.

**2.2. Khai báo cấu trúc dữ liệu**

- `GpuMassAggregate`: một `glm::vec4`, trong đó xyz là Center of Mass và w là Total Mass. Kích thước 16 bytes.
- `GpuOctreeNode`: gồm `links`, `range`, `boundsMinDiameter`, `boundsMax`; mỗi trường là `glm::uvec4` hoặc `glm::vec4`. Kích thước 64 bytes.
- `ObjectIndexList`: mảng `uint32_t`, ánh xạ sorted index sang original object ID.
- `ObjectLeafIndex`: mảng `uint32_t`, ánh xạ original object ID sang leaf node index. Dùng `0xFFFFFFFF` làm INVALID_NODE.
- `GpuOctreeMetadata`: gồm 4 vector 16 bytes, lưu số lượng vật thể hoạt động, số node, số node lá, root index, độ sâu tối đa, leaf capacity, node capacity, trạng thái xây dựng cây và root bounding box.

Tất cả cấu trúc phải có `alignas(16)` ở C++ khi cần thiết và phải có `static_assert(sizeof(...))` để xác minh kích thước.

**2.3. Tham số Octree ban đầu**

- Số chiều: 3.
- Branch factor: 8 (Octree cổ điển).
- MAX_OCTREE_DEPTH: 16, là giá trị cấu hình ban đầu cần kiểm thử.
- LEAF_CAPACITY: 1.
- Root index: 0.
- Chỉ cấp phát node không rỗng.
- Node con được cấp phát liên tục theo thứ tự octant để `firstChildIndex`, `childCount` và `childMask` hoạt động đúng.

Không sao chép nguyên `DIM_SPLIT_LOG = 3` của repository tham khảo, vì cấu hình đó tạo tối đa 512 vùng con mỗi node, không phải Octree cổ điển 8 octant.

**2.4. Capacity**

Tiếp tục sử dụng `CalculateObjectStateBufferCapacity()` hiện tại: capacity tối thiểu 256, tăng theo hệ số 2.

Với 202 vật thể ban đầu, ObjectCapacity = 256.

Sử dụng công thức capacity bảo thủ cho cây sparse có thể có node một-con:

`OctreeNodeCapacity = 1 + ObjectCapacity * MAX_OCTREE_DEPTH`.

Với ObjectCapacity = 256 và MAX_OCTREE_DEPTH = 16, OctreeNodeCapacity = 4097.

Capacity của các buffer:
- MassField: OctreeNodeCapacity.
- OctreeNodes: OctreeNodeCapacity.
- ObjectIndexList: ObjectCapacity.
- ObjectLeafIndex: ObjectCapacity.
- OctreeMetadata: 1.

Không sử dụng công thức `2*N-1` nếu cây vẫn giữ các node nội bộ chỉ có một node con. Chỉ áp dụng giới hạn đó khi đã bảo đảm nén các đường dẫn một-con và không lưu node rỗng.

**2.5. Cấp phát buffer**

Cấp phát các buffer trong `ComputePipeline()` hoặc hàm khởi tạo tài nguyên tương ứng.

Đối với dữ liệu chỉ được GPU đọc/ghi:
- Storage mode: `GpuResident`.
- Target: `GL_SHADER_STORAGE_BUFFER`.
- CPU access: `CpuAccess::None`.
- Mapping mode: `MappingMode::None`.

Chỉ dùng persistent mapping khi thực sự có yêu cầu CPU truy cập dữ liệu.

**2.6. Resize**

- Object capacity thay đổi: resize `ObjectIndexList` và `ObjectLeafIndex` cùng capacity với nhóm object buffers.
- Node capacity thay đổi: resize `OctreeNodes` và `MassField` đồng thời.
- Chờ GPU hoàn tất trước khi resize.
- Giữ dữ liệu hợp lệ khi thay buffer.
- Bind lại đúng binding sau resize.
- Reset hoặc cập nhật metadata sau khi capacity thay đổi.
- Khi GPU phát hiện node count vượt node capacity, phải đặt overflow flag và dừng các dispatch phụ thuộc vào cây chưa hợp lệ.

`OctreeMetadata.counts.y` là nguồn xác định số node thực tế do GPU xây dựng; không giả định shader tự cập nhật `GpuBuffer::LogicalCount()` ở CPU.

**2.7. Kiểm tra và nghiệm thu**

1. Xác nhận tất cả BufferRole có đúng BindingFor().
2. Xác nhận binding 0–15 không xung đột.
3. Kiểm tra `GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS`.
4. Kiểm tra `GL_MAX_SHADER_STORAGE_BLOCK_SIZE` cho từng buffer.
5. Xác nhận kích thước C++ trùng với GLSL std430.
6. Xác nhận capacity ObjectIndexList và ObjectLeafIndex đủ cho object capacity.
7. Xác nhận MassField và OctreeNodes đủ cho node capacity.
8. Xác nhận binding đúng sau khi tạo và sau khi resize.
9. Không cấp phát AccelerationField hoặc buffer chưa có shader sử dụng.
10. Không thay đổi CurrentState, NextState, ObjectAcceleration và công thức Schwarzschild Grid Deformation.

**Điều kiện hoàn thành:** GPU Memory Manager có thể tạo, quản lý, resize và bind đầy đủ buffer cần thiết cho Octree và Mass Aggregation; chưa yêu cầu kết quả Mass Aggregation chính xác ở giai đoạn này vì phần tính toán thuộc giai đoạn shader tương ứng.

### Giai đoạn 3 — Bounding Box và Octree Construction

- Tạo compute pass tính bounding box từ `CurrentState`.
- Tạo compute pass xây Octree theo thiết kế đã chốt.
- Xử lý object initializing theo đúng quy tắc của solver hiện tại; không đưa vật thể không hợp lệ vào aggregation/traversal.
- Xử lý node rỗng, node chỉ có một vật thể, nhiều vật thể trùng vị trí, giới hạn độ sâu và hết dung lượng node.
- Thêm memory barrier giữa các pass có phụ thuộc dữ liệu.
- Có kiểm tra debug để phát hiện object bị mất, xuất hiện trùng hoặc không thuộc leaf nào.

**Đầu ra:** Octree hợp lệ cho một snapshot `CurrentState`.

### Giai đoạn 4 — Mass Aggregation

- Tổng hợp khối lượng và tâm khối cho node lá.
- Tổng hợp node cha từ các node con theo hướng bottom-up.
- Bảo đảm node rỗng không gây chia cho 0 hoặc tạo giá trị NaN/Inf.
- Kiểm chứng tổng khối lượng ở root xấp xỉ tổng khối lượng của các vật thể hợp lệ trong cùng snapshot.
- Kiểm chứng tâm khối root bằng phép tính tham chiếu độc lập ở CPU trong chế độ kiểm thử.

**Đầu ra:** `MassField` chứa tổng khối lượng và tâm khối hợp lệ cho các node.

### Giai đoạn 5 — GPU Barnes–Hut

- Viết hoặc điều chỉnh compute shader để mỗi vật thể tính gia tốc bằng cách duyệt Octree.
- Dùng tiêu chí mở node `s / d < theta` hoặc biến thể được định nghĩa rõ ràng trong thiết kế.
- Bắt buộc mở node chứa vật thể mục tiêu; không dùng khối lượng tổng hợp của node đó để tự tương tác với chính nó.
- Node lá tính tương tác trực tiếp với các object index trong node.
- Giữ softening và đơn vị vật lý nhất quán với solver Direct N-body.
- Ghi gia tốc vào đúng index của `ObjectAcceleration`.
- Kiểm tra giới hạn stack/depth của traversal; có cách phát hiện overflow thay vì âm thầm bỏ qua node.

**Đầu ra:** Barnes–Hut sinh được gia tốc hữu hạn, đúng index và không bỏ sót vùng không gian.

### Giai đoạn 6 — Tích hợp vào frame pipeline

- Chèn đúng thứ tự: Bounding Box -> Octree Construction -> Mass Aggregation -> Barnes–Hut -> Integration -> Swap.
- Bảo đảm tree và mass aggregate cùng snapshot với `CurrentState` mà Barnes–Hut đọc.
- Bảo đảm collision output vẫn hợp lệ cho `integrate.comp`.
- Giữ nguyên `integrate.comp` trong giai đoạn này.
- Bảo đảm lần frame đầu tiên có tree hợp lệ trước lần gọi Barnes–Hut đầu tiên.

**Đầu ra:** chương trình chạy end-to-end bằng GPU Barnes–Hut.

### Giai đoạn 7 — Giữ nguyên Schwarzschild Grid Deformation

- Giữ các binding logic của `grid.comp`: `CurrentState` (0), `BaseGrid` (4), `DeformedGrid` (5), `ObjectDerived` (12), theo binding thực tế đã xác nhận.
- Giữ công thức khoảng cách, clamp theo `r_s` và tổng độ dịch chuyển hiện tại.
- Chạy grid compute sau `SwapCurrentNext()`.
- Không thêm dependency của `grid.comp` vào `MassField`, `OctreeNodes` hoặc `ObjectIndexList`.
- So sánh hình ảnh/kết quả grid trước và sau thay đổi khi cấp cùng một snapshot vị trí và cùng dữ liệu `ObjectDerived`; kết quả grid phải tương đương.

**Đầu ra:** biến dạng lưới vẫn dựa trên Schwarzschild như phiên bản trước; thay đổi quỹ đạo chỉ đến từ solver gia tốc mới.

### Giai đoạn 8 — Kiểm thử độ đúng và hiệu năng

- So sánh gia tốc Barnes–Hut với Direct N-body trên cùng snapshot, không tích phân xen kẽ giữa hai lần đo.
- Kiểm thử nhiều số lượng vật thể và nhiều cấu hình phân bố không gian.
- Đánh giá sai số gia tốc và sai số vị trí sau một khoảng mô phỏng cố định.
- Thử nhiều giá trị `theta`; không chọn giá trị chỉ vì frame rate cao.
- Đo riêng thời gian Bounding Box, Octree Construction, Mass Aggregation, Barnes–Hut, Integration và Grid Deformation.
- So sánh tổng thời gian mỗi frame với Direct N-body. Với số vật thể nhỏ, chi phí xây cây có thể làm Barnes–Hut chậm hơn; phải ghi nhận kết quả thực tế.
- Kiểm tra NaN/Inf, node overflow, traversal overflow, buffer out-of-bounds và lỗi đồng bộ.

**Đầu ra:** bảng kết quả độ đúng và hiệu năng; nêu rõ số vật thể, cấu hình, `theta`, thiết bị và phương pháp đo.

### Giai đoạn 9 — Cập nhật tài liệu

- Cập nhật `docs/gravity_sim_gpu_resident_architecture.md` với pipeline mới.
- Cập nhật sơ đồ buffer/binding và thứ tự compute dispatch.
- Ghi rõ `grid.comp` vẫn sử dụng Schwarzschild per-object và không phụ thuộc Octree.
- Ghi rõ giới hạn hiện tại, ví dụ collision vẫn xử lý trực tiếp, capacity Octree và giới hạn traversal.

**Đầu ra:** tài liệu phản ánh đúng implementation đã build và kiểm thử, không ghi các tính năng mới khi chúng chưa được triển khai.

---

## 8. Tiêu chí nghiệm thu

Chỉ xem việc tích hợp hoàn tất khi đáp ứng tất cả tiêu chí sau:

- [ ] Project build thành công bằng cấu hình đang dùng.
- [ ] Không có lỗi OpenGL/GLSL compile hoặc lỗi binding SSBO.
- [ ] Octree được xây từ `CurrentState` trước mỗi lần Barnes–Hut traversal.
- [ ] Tổng khối lượng node root khớp với tổng khối lượng các vật thể hợp lệ trong sai số số học cho phép.
- [ ] Tâm khối node được tính đúng; node rỗng không tạo NaN/Inf.
- [ ] Barnes–Hut không tính vật thể tự tương tác thông qua node chứa chính nó.
- [ ] `ObjectAcceleration` được ghi đầy đủ cho các vật thể cần mô phỏng.
- [ ] `ObjectCollision` vẫn có dữ liệu hợp lệ nếu `integrate.comp` tiếp tục đọc buffer này.
- [ ] `CurrentState` / `NextState` swap đúng; không đọc trạng thái cũ ngoài ý muốn.
- [ ] `grid.comp` không đọc Octree hoặc `MassField`.
- [ ] Công thức Schwarzschild và luồng render lưới được giữ nguyên.
- [ ] Có số liệu sai số so với Direct N-body trên cùng snapshot.
- [ ] Có số liệu hiệu năng theo từng compute pass và tổng frame.
- [ ] Tài liệu kiến trúc được cập nhật theo mã đã triển khai.

---

## 9. Các lỗi thiết kế cần tránh

1. **Xây Octree sau khi đã tính Barnes–Hut:** khiến solver đọc cây của frame trước hoặc dữ liệu chưa được khởi tạo.
2. **Xây cây từ dữ liệu CPU cũ:** gây sai khác giữa vị trí trong cây và vị trí mà shader đang tính lực.
3. **Không mở node chứa vật thể mục tiêu:** có thể đưa chính vật thể đó vào tổng khối lượng node và tạo tự tương tác sai.
4. **Dùng tâm hình học root làm tâm khối:** tâm hình học của bounding box không phải tâm khối lượng của hệ.
5. **Coi Earth là root hoặc tâm bắt buộc của Octree:** Octree phải bao phủ toàn bộ phân bố vật thể; Earth chỉ là một vật thể trong hệ.
6. **Coi LOD đồng nghĩa với độ sâu cây:** độ sâu là vị trí node trong cây; LOD là mức chi tiết/xấp xỉ. Tiêu chí Barnes–Hut phải dựa trên kích thước node và khoảng cách, không gán cứng mỗi tầng cây thành một LOD.
7. **Dùng Mass Aggregation để thay collision chính xác:** khối lượng và tâm khối node không đủ để xác định cặp vật thể va chạm.
8. **Cho `grid.comp` đọc `MassField` để tính độ lõm:** trái với phương án đã chọn; grid tiếp tục dùng bán kính Schwarzschild từng vật thể.
9. **Quên memory barrier:** có thể khiến pass sau đọc dữ liệu chưa hoàn tất từ pass trước.
10. **Không xử lý capacity/overflow:** có thể làm mất vật thể hoặc node mà không báo lỗi.
11. **Thay timestep trong cùng commit tích hợp Barnes–Hut:** làm khó phân biệt sai số do solver mới với sai số do tích phân thay đổi.
12. **Kết luận Barnes–Hut nhanh hơn chỉ từ độ phức tạp lý thuyết:** phải tính cả chi phí xây cây, aggregation và traversal trên GPU.

---

## 10. Chỉ dẫn trực tiếp cho Copilot Agent

Hãy thực hiện công việc theo thứ tự ở Mục 7. Trước khi sửa mã, hãy kiểm tra working tree, nhánh hiện tại, nội dung các file và xác nhận các binding đang được sử dụng. Không giả định các buffer chỉ mới được khai báo trong enum đã được cấp phát.

Mỗi giai đoạn cần:

1. Nêu ngắn gọn các file dự kiến thay đổi và mục tiêu thay đổi.
2. Chỉ sửa phạm vi cần thiết; không refactor ngoài mục tiêu.
3. Build hoặc chạy kiểm thử liên quan ngay sau thay đổi.
4. Báo cáo file/hàm/shader đã sửa, kết quả kiểm thử và phần chưa hoàn thành.
5. Không đánh dấu một giai đoạn hoàn tất nếu chưa có bằng chứng build/test tương ứng.

Ưu tiên tạo một phiên bản có thể kiểm chứng ở từng giai đoạn hơn là triển khai toàn bộ Octree, aggregation và traversal cùng lúc. Trong mọi thay đổi, phải giữ nguyên đường xử lý Schwarzschild Grid Deformation và phải chỉ ra rõ nếu có bất kỳ thay đổi nào ảnh hưởng đến nó.

**Kết quả cuối cùng cần bàn giao:** mã nguồn GPU Barnes–Hut tích hợp vào pipeline hiện tại, kiểm thử so sánh với Direct N-body của `gravity_sim_3Dgrid_baseline.cpp`, số liệu hiệu năng, và tài liệu kiến trúc cập nhật. Không được thay thế hoặc loại bỏ Direct N-body trong `gravity_sim_3Dgrid_baseline.cpp`, mà chỉnh sửa trực tiếp trong các file `gravity_sim_3Dgrid.cpp`, `gravity_sim_3Dgrid_function.cpp`, và các file liên quan khác.
