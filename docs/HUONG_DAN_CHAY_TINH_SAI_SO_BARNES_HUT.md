# Hướng dẫn chạy kiểm thử sai số Barnes–Hut

## 1. Chuẩn bị và chạy

Mở PowerShell tại thư mục gốc repository. Nếu CMake yêu cầu `VCPKG_ROOT`, đặt
biến môi trường đến vcpkg đi cùng Visual Studio:

```powershell
$env:VCPKG_ROOT = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\vcpkg"
```

Build Release, chạy unit test cho các công thức sai số, rồi chạy benchmark:

```powershell
cmake --build .\build\msvc-vs2026 --config Release --target gravity_sim_3Dgrid error_utils_test -- /m
ctest --test-dir .\build\msvc-vs2026 -C Release -R error_utils_test --output-on-failure
.\build\msvc-vs2026\Release\gravity_sim_3Dgrid.exe --benchmark
```

Phải chạy lệnh cuối từ thư mục gốc repository vì đường dẫn kết quả là tương đối.
`--benchmark` kết thúc sau khi xuất kết quả, không vào vòng lặp mô phỏng tương tác.
Mỗi lần chạy tạo thư mục riêng `numerical_error/run_YYYYMMDD_HHMMSS_mmm/`;
nếu trùng tên, chương trình thêm hậu tố số. Các CSV cũ trong `numerical_error/`
không bị ghi đè.

## 2. Cấu hình benchmark

- Với cảnh mặc định 202 vật thể, sweep dùng 32, 128 và 202 vật thể; ba phân bố
  `uniform`, `clustered`, `disk`; và `theta = 0.3, 0.5, 0.8` (27 dòng tổng hợp).
- Snapshot vị trí/khối lượng được tạo xác định từ seed ghi trong CSV. Direct
  N-body và Barnes–Hut tính gia tốc trên cùng snapshot, không xen tích phân giữa
  hai lần tính lực.
- So sánh quỹ đạo dùng snapshot `clustered` riêng gồm tối đa 32 vật thể, chạy
  24 bước với `dt = 1/96 s`, rồi ghi sai số vị trí và vận tốc ở từng bước.
- Mass Aggregation được đối chiếu độc lập với phép tính trực tiếp trên các node
  lá và node cha. Ngưỡng là sai số khối lượng tương đối `1e-5`; sai số tâm khối
  `1e-3 + 1e-5 * |COM_reference|`.
- GPU timing dùng OpenGL `GL_TIME_ELAPSED`, có warm-up và ba lượt đo. Báo cáo
  trung bình, trung vị và độ lệch chuẩn tổng thể. CPU submission time, thời gian
  truyền dữ liệu riêng, frame time và FPS được ghi `N/A` vì chế độ benchmark
  không đo các đại lượng này.

Đây là phép so sánh Direct N-body shader và Barnes–Hut trong cùng tiến trình,
không đọc `baseline_positions.csv` hay chạy executable baseline độc lập. Nó
đánh giá sai số gia tốc trên snapshot giống nhau và sai số quỹ đạo sau các bước
mô phỏng riêng. `src/numerical_error.cpp` là chương trình cũ để so sánh CSV vị
trí; không cần chạy file đó cho benchmark mới.

## 3. File kết quả

Mỗi thư mục `run_*` gồm:

| File | Nội dung |
| --- | --- |
| `run_metadata.json` | GPU/OpenGL, G, epsilon, timestep, seed policy, số lượt đo và ngưỡng sai số. |
| `comparison/summary.csv` | 27 tổ hợp (với cảnh mặc định): sai số gia tốc/lực, thời gian từng pass, tổng GPU compute, và tổng hợp quỹ đạo cho case clustered 32 vật thể. |
| `comparison/acceleration_error.csv` | Sai số gia tốc và lực theo vật thể, theta, và lượt lặp. |
| `comparison/trajectory_error.csv` | Sai số vị trí và vận tốc theo vật thể, theta và từng bước của quỹ đạo 24 bước. |
| `comparison/per_step_error.csv` | RMSE và sai số lớn nhất vị trí/vận tốc theo từng bước. |
| `mass_aggregation/node_validation.csv` | Khối lượng và tâm khối tham chiếu/đã tổng hợp cho từng node. |
| `mass_aggregation/summary.csv` | Số node đạt/trượt và sai số lớn nhất theo phân bố/số vật thể. |
| `performance.csv` | GPU mean/median/stddev theo từng pass và tổng pipeline. Các trường không đo được là `N/A`. |

`summary.csv` ghi các sai số vector theo độ lớn vector sai khác:

- Với mỗi vật thể, `e_i = |value_BH - value_Direct|`.
- Vector RMSE: `sqrt(sum(e_i^2) / N)`.
- MAPE: `100 * mean(e_i / |value_Direct_i|)` trên các mẫu có
  `|value_Direct_i| > 1e-12`.
- Relative error tổng hợp: `sum(e_i) / sum(|value_Direct_i|)` trên cùng các mẫu
  hợp lệ. `*_excluded_samples` ghi số mẫu bị loại do tham chiếu bằng/gần 0.
- Max error là `max(e_i)`. MAPE được tính theo phần trăm; relative error là tỷ
  lệ không thứ nguyên. Nếu không có mẫu hợp lệ, CSV để trống giá trị metric đó.

Với quỹ đạo, RMSE mỗi bước lấy trên số vật thể; RMSE tổng hợp lấy trên toàn bộ
các cặp vật thể-bước. Các đơn vị của sai số vị trí/vận tốc theo đơn vị mô phỏng.
Sai số lực trong CSV được suy ra từ sai số gia tốc nhân với khối lượng vật thể.

## 4. Đọc và kiểm tra kết quả

Từ PowerShell, dùng đường dẫn thư mục `run_*` vừa được in ở Terminal:

```powershell
$run = ".\numerical_error\run_YYYYMMDD_HHMMSS_mmm"
Import-Csv "$run\comparison\summary.csv" |
    Format-Table distribution, object_count, theta, acceleration_relative_error, direct_total_ms, barnes_hut_total_ms
Import-Csv "$run\mass_aggregation\summary.csv" |
    Format-Table distribution, object_count, node_count, passed_nodes, failed_nodes
```

Chỉ kết luận hiệu năng từ các cột GPU compute có phép đo tương ứng; không xem
`N/A` là 0. Nếu benchmark dừng do lỗi trạng thái cây, kiểm tra file stderr và
`mass_aggregation/summary.csv`; không coi một kết quả chạy dở là benchmark đạt.

## 5. Phạm vi kết luận

Sweep hiện tại dùng các kích thước/phân bố nhỏ nêu trên, ba giá trị theta và ba
lượt đo. Nó chưa đánh giá CPU time, truyền dữ liệu riêng, FPS/frame time, hoặc
hiệu năng với hệ thống lớn hơn. Kết quả không thay thế kiểm thử trên snapshot
được lưu từ một lần chạy baseline độc lập.
