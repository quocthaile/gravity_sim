# GPU octree và mass aggregation

## Pipeline và hợp đồng

Mỗi bước dựng lại cây từ `CurrentState`, không tải toàn bộ vị trí về CPU:

1. GPU tính bounds từ các object không `initializing`. Một root lập phương bao
   phủ bounds, mở rộng 0,001%; trạng thái không hữu hạn hoặc bounds không biểu
   diễn được báo lỗi. Mọi object hợp lệ đi vào một lá.
2. Tám cấp octree được dựng bằng các pass đánh dấu octant và cấp phát node.
   Dispatch giữa các cấp có `GL_SHADER_STORAGE_BARRIER_BIT`; node con được
   công bố trước pass dùng nó. Lá chứa danh sách slot object, kể cả nhiều object
   trùng vị trí. Danh sách được merge-sort theo slot để thứ tự cộng không phụ
   thuộc thứ tự atomic. Không cần sort Morton.
3. Aggregation chạy từ độ sâu 8 về root, một dispatch và barrier cho mỗi cấp.
   Mỗi node dùng tổng mass và trung bình có trọng số; tổng trung gian là double,
   dữ liệu lưu là float. Node không thể aggregate có cờ riêng.
4. CPU đọc **16 byte metadata đồng bộ**. Nếu lỗi, dừng rõ ràng, không swap và
   không xuất trajectory dở dang. Force/collision tự kiểm tra lỗi trên GPU;
   integration cũng tự copy nguyên state khi lỗi, kể cả trước khi CPU đọc.
5. `nbody.comp` duyệt cây với `theta = 0.5`; node chứa target luôn được mở
   theo đường octant của target. Lá duyệt từng source, bỏ qua target. Node không
   chứa target chỉ được xấp xỉ khi `2 * halfSize < theta * distanceToCOM`.
   `theta = 0` mở hết lá. Công thức lực, epsilon, hệ số 1000 và hướng giữ nguyên.
6. `collision.comp` kiểm tra chính xác tất cả cặp, bỏ qua initializing và
   khoảng cách bằng 0; `integrate.comp` áp dụng `dt`, pause và số overlap như
   trước, sau đó swap. Collision vẫn O(N²).

Mass hữu hạn không dương vẫn tham gia tính trực tiếp ở lá. Bất kỳ node nào
chứa mass không dương, hoặc có tổng mass/COM không biểu diễn được bằng float,
không được dùng làm aggregate và phải mở xuống dưới. Giá trị mass hoặc vị trí
không hữu hạn của source hợp lệ khiến build thất bại. Với target mass bằng 0,
phép chia `force / targetMass` vẫn giữ semantics NaN của shader direct cũ;
đây không phải NaN do tính COM. Các object initializing bị loại khỏi cây.

## Buffer, giới hạn và đồng bộ

Binding 6 là `NodeAggregate` (`vec4(COM.xyz, mass)`); 7 là node topology
(`vec4` center/half size, hai `uvec4` children, một `uvec4` dữ liệu; 64 B);
8 là `uvec2` cho linked list và scratch mỗi object; 9 là `uvec4` metadata.
Chỉ số sentinel là `0xffffffff`. C++ kiểm tra kích thước/offset node tương ứng
với std430.

Với object capacity C, node capacity là `1 + 8*C`, đủ trường hợp mỗi object
phát sinh một node mới ở từng cấp. Bộ nhớ riêng cây là
`80*(1 + 8*C) + 8*C + 16 = 648*C + 96` byte, chưa tính driver. Tại C=256 là
165.984 byte, C=512 là 331.872 byte. Tám object buffer cũ thêm 148*C byte,
chưa tính grid. Cây được resize cùng object capacity, không allocate mỗi frame.
Trước allocate/resize, CPU kiểm tra số binding, 8 compute SSBO blocks, 32-bit
index, kích thước SSBO tối đa và dispatch limit; allocation thất bại được báo
lỗi. GPU kiểm tra capacity trước khi ghi và đặt status 1 nếu overflow, 2 nếu
dữ liệu/bounds không biểu diễn được. Không có direct-force fallback production.

Root bounds hiện được tính bởi một invocation, các bước còn lại song song theo
object/node. Tập trung nhiều vật thể vào một lá vẫn gây tải lớn cho một
invocation khi sort/tổng hợp và cần lực direct trong lá. Status read đồng bộ
mỗi bước cũng có chi phí; đây không phải readback toàn bộ vị trí. Trajectory
export sẵn có vẫn lấy mẫu ban đầu và mỗi 96 frame.

## Kiểm thử và tái lập

Build bằng CMake, sau đó chạy CTest trên máy có OpenGL 4.4:

```powershell
cmake --build build\msvc-vs2026 --config Debug --target gravity_mass_aggregation_test gravity_state_transition_test
ctest --test-dir build\msvc-vs2026 -C Debug -R "^gravity_(mass_aggregation|state_transition)$" --output-on-failure
```

`gravity_mass_aggregation_test` dùng oracle CPU direct độc lập và một shader
direct chỉ nằm trong `tests` để đo đối chứng, không được nối vào production.
Nó kiểm tra membership, child index, mass/COM của các node, force mở hết lá, collision,
self-interaction, initializing, một vật thể, tất cả initializing, 1.024 vị trí
trùng, mass âm/0, tổng mass vượt float, NaN/bounds không hợp lệ, overflow với
sentinel chống ghi tràn, GPU gate, tính lặp lại và các phân bố random. Test
transition dùng pipeline production, kiểm tra pause, dt, ping-pong, append,
resize và không swap khi cây lỗi.

Có thể so sánh trajectory với **bản sao** CSV baseline đã có bằng cách truyền
thêm đường dẫn CSV:

```powershell
build\msvc-vs2026\Debug\gravity_mass_aggregation_test.exe shaders C:\path\baseline_positions_copy.csv
```

Test tái tạo đầu vào mặc định 2+200 object, seed 123, `dt=1/96`; xác nhận
chương trình CPU direct tuần tự tái tạo lại CSV ở các checkpoint, rồi chạy
GPU 960 bước với theta 0 và 0.5. Bước GPU 96, 192, ... được ghép với dòng
baseline có nhãn step 95, 191, ...; bản CSV không bị sửa.

## Số liệu tham khảo

Đo ngày 01/11/2026 trên NVIDIA RTX 3050 Laptop, driver 610.62, OpenGL 4.4,
Windows, CPU i7-11800H, build MSVC Debug. Giới hạn được báo: 96 SSBO binding,
16 compute SSBO blocks, 2.147.483.647 B/block. Mỗi dòng là snapshot cố định,
seed 123, epsilon 10, theta 0.5, 3 warmup + 10 lần đo; không render và không
swap. Uniform trong [-1000,1000]^3; clustered gồm hai cụm quanh x=±500, bán
kính theo trục 20. Mass tuần hoàn từ 1e20 đến 7e20. Times GPU từ timestamp
query; wall gồm dispatch, đồng bộ và đọc status 16 byte. Test cấp capacity
theo N, không phải capacity lũy thừa 2 của production.

| Phân bố | N | Node dùng/capacity | Build/group ms | Aggregate ms | Traversal ms | Collision ms | Integration ms | Direct force test ms | Wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 202 | 1239/1617 | 0.277 | 0.090 | 0.402 | 0.071 | 0.006 | 0.338 | 1.007 |
| Clustered | 202 | 454/1617 | 0.277 | 0.086 | 0.626 | 0.071 | 0.006 | 0.338 | 1.231 |
| Uniform | 1024 | 5499/8193 | 1.129 | 0.101 | 0.961 | 0.344 | 0.006 | 1.837 | 2.790 |
| Clustered | 1024 | 1281/8193 | 0.895 | 0.086 | 2.086 | 0.314 | 0.006 | 1.612 | 3.574 |

Một lần resize production từ capacity 256 lên 512, gồm đồng bộ và materialize,
đo 0,714 ms; không coi đây là trung bình ổn định. Các số trên **không chứng
minh tăng tốc toàn mô phỏng**: ở 1024 uniform traversal nhanh hơn direct force,
nhưng build và collision thêm chi phí; các trường hợp còn lại traversal chậm
hơn direct. Rendering và benchmark CPU/GPU end-to-end không nằm trong phép đo.

Sai số acceleration so với CPU direct cùng snapshot; đơn vị tuyệt đối là
m/s², relative = độ lớn sai số / độ lớn lực tham chiếu:

| Phân bố | N | Absolute RMS | Absolute max | Relative RMS | Relative p50 | Relative p95 | Relative max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 202 | 0.02257 | 0.14915 | 0.00827 | 0.00496 | 0.01266 | 0.05745 |
| Clustered | 202 | 22.4736 | 82.6489 | 0.00633 | 0.00417 | 0.01233 | 0.02118 |
| Uniform | 1024 | 0.07656 | 0.33602 | 0.00712 | 0.00427 | 0.01219 | 0.05564 |
| Clustered | 1024 | 77.0933 | 182.094 | 0.00552 | 0.00377 | 0.00913 | 0.05279 |

Theta 0.5 được chọn tạm theo các mẫu trên (p95 dưới 5% trong test), không phải
giới hạn sai số cho mọi input. Theta 0 kiểm tra direct với tolerance float;
thứ tự tính khác nên không yêu cầu bitwise bằng oracle.

Đối chiếu bản sao CSV baseline có sẵn từ 10/10/2026, SHA-256
`DEE3EC7C4ADDE1C0F086CAEEB8AF1863FBEEEDC491AF6CB0D48CE5E527CC4FB5`.
Không chạy lại executable baseline vì đường dẫn CSV hard-code sẽ ghi đè file
người dùng. Nguồn gốc bản CSV cũ chưa được xác minh độc lập, nhưng phép tính
CPU tuần tự tái tạo đầu vào và vị trí của nó khớp chính xác tại bước 1 và
10 checkpoint mỗi 96 bước. GPU chạy không render, 202 object, 960 bước:

| Bước thực | Theta 0 RMS / max vị trí | Theta 0.5 RMS / max vị trí |
|---:|---:|---:|
| 96 | 0.390 / 3.872 | 0.404 / 3.791 |
| 480 | 10.571 / 134.625 | 11.016 / 134.121 |
| 960 | 109.773 / 733.620 | 111.714 / 751.429 |

Đây là sai lệch end-to-end, không quy toàn bộ cho Barnes–Hut: CPU baseline
không cộng epsilon², cập nhật object tuần tự và xen collision vào từng cặp;
GPU dùng snapshot, ping-pong và áp collision sau tổng lực. CSV cũ và môi trường
sinh nó không được coi là benchmark thời gian hay oracle force chính xác.

## Phân tích log CTest được cung cấp

### Kết quả tổng quát

- CTest chạy **một test** `gravity_mass_aggregation`: đạt, không có test thất
  bại; thời gian thực toàn bộ là 1,33 giây. Đây không phải thời gian trung bình
  của một frame. Dòng `Direct force and collision contract passed` xác nhận
  các kiểm tra trong executable đã hoàn thành.
- Môi trường được log ghi nhận: OpenGL 4.4.0, NVIDIA RTX 3050 Laptop GPU,
  driver 610.62; giới hạn SSBO là 96 binding, 16 compute block và
  2.147.483.647 byte/block.
- Lần chạy này không có đối số CSV baseline, nên không kiểm tra trajectory;
  log cũng không cho thấy test `gravity_state_transition` được chạy.

### Độ chính xác lực với theta = 0,5

Sai số được tính từ gia tốc của từng object so với CPU direct trên cùng
snapshot. Sai số tương đối bằng độ lớn vector sai số chia cho độ lớn gia tốc
tham chiếu, với mẫu số tối thiểu `1e-30`. Các giá trị phần trăm dưới đây được
chuyển từ tỷ lệ trong log; RMS là căn trung bình bình phương.

| Phân bố | N | Absolute RMS (m/s²) | Absolute max (m/s²) | Relative RMS | Relative p50 | Relative p95 | Relative max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 202 | 0,0225699 | 0,149147 | 0,827% | 0,496% | 1,266% | 5,745% |
| Clustered | 202 | 22,4736 | 82,6489 | 0,633% | 0,417% | 1,233% | 2,118% |
| Uniform | 1024 | 0,0765627 | 0,336017 | 0,712% | 0,427% | 1,219% | 5,564% |
| Clustered | 1024 | 77,0933 | 182,094 | 0,552% | 0,377% | 0,913% | 5,279% |

- Cả bốn trường hợp đạt điều kiện test: **relative p95 < 5%**. P95 cao nhất
  là 1,266%, nhưng điều này không có nghĩa mọi object đều dưới 5%: relative
  max vượt 5% ở ba trường hợp.
- Sai số tuyệt đối của clustered lớn hơn nhiều, trong khi sai số tương đối
  RMS vẫn dưới 1%. Hai thước đo khác nhau về cách chuẩn hóa; không thể chỉ
  dựa vào sai số tuyệt đối để kết luận clustered kém chính xác hơn.
- Các kết quả này chỉ mô tả những snapshot đã kiểm tra, không bảo đảm ngưỡng
  sai số cho mọi phân bố hoặc cho trajectory sau nhiều bước.

### Thời gian và quy mô cây

Đây là các trung bình của 10 lần đo sau 3 lần warmup trên snapshot cố định,
không render hoặc swap. Các cột theo giai đoạn là GPU timestamp; `wall` là
thời gian CPU từ trước khi dispatch đến sau khi đồng bộ pipeline, gồm đọc
status. Shader direct chỉ dùng để đối chứng và được đo riêng, không nằm trong
`wall`.

| Phân bố | N | Node dùng/capacity | Build/group (ms) | Aggregate (ms) | Traversal (ms) | Collision (ms) | Integration (ms) | Direct force (ms) | Wall (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 202 | 1239/1617 | 0,265421 | 0,088064 | 0,396493 | 0,0704512 | 0,0053248 | 0,338842 | 0,94542 |
| Clustered | 202 | 454/1617 | 0,265421 | 0,0832512 | 0,625254 | 0,0699392 | 0,0057344 | 0,337818 | 1,21299 |
| Uniform | 1024 | 5499/8193 | 0,904090 | 0,100864 | 0,957542 | 0,342938 | 0,0053248 | 1,739160 | 2,57106 |
| Clustered | 1024 | 1281/8193 | 0,865997 | 0,0905216 | 2,190030 | 0,342630 | 0,0056320 | 1,735880 | 3,69710 |

- Với 1024 object uniform, traversal nhanh hơn direct force khoảng 45% khi
  chỉ so sánh hai phép đo lực. Ba trường hợp còn lại traversal chậm hơn direct
  khoảng 17% đến 85%.
- Build/group cộng aggregate ở 1024 uniform mất khoảng 1,005 ms, nên lợi thế
  traversal riêng lẻ không đủ chứng minh toàn bộ pipeline nhanh hơn phương án
  direct. `wall` còn bao gồm collision, integration và đồng bộ, trong khi
  `direct_force` chỉ đo shader lực; không thể so sánh trực tiếp hai cột để
  suy ra tốc độ tăng hay giảm của toàn mô phỏng.
- Ở N=1024, clustered chỉ dùng 1281 node so với 5499 node của uniform nhưng
  traversal mất 2,190 ms so với 0,958 ms. Ít node không đồng nghĩa duyệt
  nhanh hơn: object tập trung có thể làm tăng công việc xử lý trực tiếp trong
  các lá và thay đổi số node cần mở.
- Collision tăng từ khoảng 0,070 ms ở N=202 lên 0,343 ms ở N=1024, phù hợp
  với việc phép kiểm tra cặp vẫn là O(N²); các phép đo riêng lẻ chưa đủ để
  xác lập quy luật hiệu năng tổng quát.

**Kết luận:** test đã chạy đạt và sai số p95 ở bốn mẫu thấp hơn ngưỡng 5%.
Không thể từ log này kết luận mô phỏng end-to-end được tăng tốc; cần benchmark
hai pipeline tương đương, nhiều kích thước và phân bố, cùng số liệu biến thiên
trước khi đưa ra kết luận hiệu năng tổng quát.
