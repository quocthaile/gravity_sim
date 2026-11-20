# KẾ HOẠCH BENCHMARK SAI SỐ GIỮA DIRECT N-BODY VÀ BARNES–HUT GPU

## 1. Mục tiêu

Xây dựng chương trình kiểm thử để đánh giá độ chính xác của thuật toán Barnes–Hut GPU so với thuật toán Direct N-body CPU đang được sử dụng làm baseline.

Việc kiểm thử gồm 3 giai đoạn theo thứ tự:

1. Kiểm tra tính đúng đắn của Mass Aggregation.
2. Đánh giá sai số gia tốc giữa Direct N-body và Barnes–Hut.
3. Đánh giá sai số quỹ đạo giữa hai chương trình sau quá trình mô phỏng.

**Phạm vi thực hiện:** Chỉ đánh giá sai số số học và sai lệch kết quả mô phỏng. Không thực hiện đo hiệu năng, FPS, thời gian thực thi, mức sử dụng CPU/GPU hoặc benchmark tốc độ.

## 2. Các chương trình và file liên quan

Sử dụng các thành phần hiện có trong repository:

- `src/gravity_sim_3Dgrid_baseline.cpp`: chương trình Direct N-body CPU, dùng làm kết quả tham chiếu.
- `src/gravity_sim_3Dgrid.cpp`: chương trình Barnes–Hut GPU cần đánh giá.
- `src/gravity_sim_3Dgrid_function.cpp`: các hàm hỗ trợ tạo vật thể và xử lý mô phỏng GPU cho `src/gravity_sim_3Dgrid.cpp`.
- `shaders/mass_aggregation.comp`: shader thực hiện Mass Aggregation.
- `shaders/barnes_hut.comp`: shader tính gia tốc bằng Barnes–Hut.
- `shaders/integrate.comp`: shader cập nhật trạng thái vật thể.
- `error_utils.cpp`: triển khai các hàm tính toán sai số.

Trước khi triển khai, hãy đọc lại các file trên và xác định luồng xử lý thực tế. Ưu tiên tái sử dụng các hàm, cấu trúc dữ liệu và shader hiện có. Không viết lại toàn bộ chương trình mô phỏng nếu không cần thiết.

## 3. Thiết lập chung cho các phép kiểm thử

### 3.1. Dữ liệu đầu vào

Hai chương trình phải sử dụng chính xác cùng một tập dữ liệu ban đầu.

Mỗi vật thể phải có các thông tin tương ứng:

- ID vật thể.
- Khối lượng.
- Vị trí ban đầu.
- Vận tốc ban đầu.
- Các thuộc tính cần thiết khác để tính lực hấp dẫn.

Không chỉ dựa vào việc hai chương trình sử dụng cùng random seed. Hãy tạo dữ liệu đầu vào một lần, sau đó cung cấp cùng dữ liệu đó cho cả Direct N-body và Barnes–Hut.

Khi so sánh kết quả, phải đối chiếu vật thể theo ID, không dựa vào vị trí của vật thể trong mảng.

### 3.2. Quy mô và phân bố vật thể

Sử dụng các cấu hình kiểm thử sau:

| Cấu hình | Tổng số vật thể |
|---|---:|
| Nhỏ | 32 |
| Trung bình | 128 |
| Hiện tại | 202 |

Lưu ý: Hàm `CreateObjects()` hiện tại có thêm 2 vật thể cố định. Nếu tiếp tục sử dụng hàm này, số vật thể ngẫu nhiên tương ứng phải là 30, 126 và 200 để đạt tổng số lượng nêu trên.

Kiểm thử với 3 kiểu phân bố:

- Uniform: vật thể phân bố tương đối đồng đều trong không gian 3D.
- Clustered: vật thể tập trung thành các cụm không gian.
- Disk: vật thể phân bố dạng đĩa, tương tự cách tạo các vật thể ngẫu nhiên hiện tại.

Mỗi kiểu phân bố phải sử dụng dữ liệu đầu vào xác định và có thể tái tạo lại. Không sử dụng một tập dữ liệu khác nhau giữa Baseline và Target.

### 3.3. Tham số mô phỏng

Sử dụng cùng thiết lập cho hai chương trình:

- Timestep: `dt = 1.0 / 96.0`.
- Tổng thời gian mô phỏng: `T = 10.0` giây.
- Tổng số bước mô phỏng: 960.
- Lấy mẫu quỹ đạo: mỗi 96 bước, tương ứng mỗi 1 giây.
- Chạy ít nhất 3 lần với cùng một cấu hình để kiểm tra tính nhất quán của kết quả.

Đối với Barnes–Hut, kiểm thử thêm các giá trị:

- `theta = 0.3`.
- `theta = 0.5`.
- `theta = 0.8`.

Theta là tham số quyết định mức độ xấp xỉ khi sử dụng node tổng hợp thay cho các vật thể riêng lẻ. Giá trị nhỏ thường yêu cầu duyệt cây nhiều hơn và cho kết quả gần Direct N-body hơn; giá trị lớn cho phép xấp xỉ nhiều hơn.

Không thay đổi `dt` giữa hai chương trình trong cùng một phép so sánh.
---

# GIAI ĐOẠN 1: KIỂM TRA TÍNH ĐÚNG ĐẮN CỦA MASS AGGREGATION

## 1.1. Mục tiêu

Xác minh quá trình Mass Aggregation trên GPU tính đúng tổng khối lượng và tâm khối của từng node trong Octree.

Giai đoạn này chỉ kiểm tra kết quả tổng hợp khối lượng và tâm khối. Không sử dụng kết quả Mass Aggregation để đánh giá gia tốc ở giai đoạn này.

## 1.2. Cách thực hiện

**Bước 1: Tạo dữ liệu đầu vào**

Tạo các tập vật thể theo những cấu hình đã xác định ở phần thiết lập chung.

Đảm bảo mỗi vật thể có ID, khối lượng và vị trí xác định.

**Bước 2: Thực hiện Mass Aggregation trên GPU**

Chạy luồng xử lý xây dựng Octree và Mass Aggregation hiện có.

Thu thập dữ liệu của từng node sau khi hoàn thành tổng hợp:

- ID node.
- Tổng khối lượng.
- Tâm khối.
- Quan hệ node cha và node con.
- Thông tin vật thể thuộc node lá nếu cần để xác minh.

**Bước 3: Tính kết quả tham chiếu trên CPU**

Viết hàm kiểm tra độc lập để tính tổng khối lượng và tâm khối từ các vật thể thuộc node.

Đối với node lá, tính trực tiếp từ các vật thể thuộc node đó.

Đối với node cha, tính từ kết quả tổng hợp của các node con.

Công thức tổng khối lượng:

\[
M=\sum_{i=1}^{n}m_i
\]

Công thức tâm khối:

\[
\vec{COM}=
\frac{\sum_{i=1}^{n}m_i\vec{p}_i}
{\sum_{i=1}^{n}m_i}
\]

Trong đó:

- \(m_i\): khối lượng vật thể hoặc node con.
- \(\vec{p}_i\): vị trí vật thể hoặc tâm khối node con.
- \(n\): số vật thể hoặc node con tham gia tổng hợp.

Ưu tiên sử dụng `double` trong phép tính tham chiếu trên CPU để hạn chế sai số tích lũy khi cộng nhiều giá trị.

**Bước 4: So sánh kết quả**

So sánh kết quả GPU với kết quả tham chiếu CPU theo từng node tương ứng.

### 1.3. Các chỉ số đánh giá

**A. Sai số tuyệt đối tổng khối lượng**

\[
E_M=|M_{GPU}-M_{ref}|
\]

**B. Sai số tuyệt đối tâm khối**

\[
E_{COM}=
\left\|
\vec{COM}_{GPU}-\vec{COM}_{ref}
\right\|_2
\]

Khai triển:

\[
E_{COM}=
\sqrt{
(COM_{GPU,x}-COM_{ref,x})^2+
(COM_{GPU,y}-COM_{ref,y})^2+
(COM_{GPU,z}-COM_{ref,z})^2
}
\]

Không sử dụng MAPE cho tâm khối vì các thành phần tọa độ có thể bằng hoặc gần 0.

Do phép tính trên GPU dùng `float`, so sánh khối lượng dùng sai số tuyệt đối `1e-6` và sai số tương đối `1e-5`. Với từng thành phần tâm khối, thang đo tương đối còn xét đường kính node để không đặt ngưỡng quá nhỏ khi tâm khối gần 0.

### 1.4. Các điều kiện cần xác minh

- Tổng khối lượng của node gốc phải bằng tổng khối lượng của toàn bộ vật thể.
- Tổng khối lượng của node cha phải bằng tổng khối lượng các node con.
- Tâm khối của node cha phải khớp với kết quả tính từ các node con.
- Không có vật thể bị bỏ sót hoặc được tính lặp trong quá trình phân chia node.
- Các node rỗng phải được xử lý đúng, không gây chia cho 0.
- Mỗi node phải được đối chiếu đúng với node tham chiếu tương ứng.

### 1.5. Kết quả cần lưu

Lưu dữ liệu kiểm thử theo từng node, bao gồm:

- Khối lượng tính trên GPU.
- Khối lượng tham chiếu trên CPU.
- Sai số tuyệt đối khối lượng.
- Tâm khối tính trên GPU.
- Tâm khối tham chiếu trên CPU.
- Sai số tuyệt đối tâm khối.

Chỉ chuyển sang giai đoạn 2 sau khi đã xác minh được tính đúng đắn của Mass Aggregation.

---

# GIAI ĐOẠN 2: ĐÁNH GIÁ SAI SỐ GIA TỐC GIỮA DIRECT N-BODY VÀ BARNES–HUT

## 2.1. Mục tiêu

Đánh giá mức độ chính xác của gia tốc do Barnes–Hut GPU tính toán so với gia tốc tham chiếu từ Direct N-body CPU.

Đây là phép kiểm thử trực tiếp nhất để đánh giá ảnh hưởng của thuật toán Barnes–Hut, vì chỉ so sánh kết quả gia tốc tại cùng một trạng thái vật lý, chưa để sai số tích lũy qua nhiều bước mô phỏng.

## 2.2. Cách thực hiện

**Bước 1: Tạo snapshot trạng thái vật thể**

Snapshot là bản ghi trạng thái của toàn bộ vật thể tại một thời điểm cụ thể.

Mỗi snapshot phải chứa:

- ID vật thể.
- Khối lượng.
- Vị trí.
- Vận tốc nếu thuật toán hiện tại cần sử dụng.

Sử dụng cùng một snapshot cho cả hai chương trình.

Ngoài trạng thái ban đầu, có thể lấy thêm snapshot từ một trạng thái mô phỏng đã xác định trước để kiểm tra ở các cấu hình không gian khác nhau. Khi đó, snapshot phải được tạo một lần và cung cấp giống nhau cho cả hai chương trình.

**Bước 2: Tính gia tốc tham chiếu**

Sử dụng Direct N-body CPU để tính gia tốc cho từng vật thể từ snapshot.

Kết quả phải được lưu riêng, không cập nhật vị trí hoặc vận tốc vật thể trong bước kiểm thử này.

**Bước 3: Tính gia tốc bằng Barnes–Hut GPU**

Sử dụng cùng snapshot để chạy:

1. Xây dựng Octree.
2. Thực hiện Mass Aggregation.
3. Duyệt cây Barnes–Hut.
4. Tính gia tốc cho từng vật thể.

Thực hiện lần lượt với `theta = 0.3`, `0.5` và `0.8`.

Không thực hiện bước tích phân cập nhật vị trí và vận tốc.

**Bước 4: So sánh gia tốc**

Đối chiếu gia tốc của từng vật thể theo ID giữa hai chương trình.

Đảm bảo cả hai chương trình sử dụng cùng hằng số hấp dẫn, quy tắc loại trừ tương tác tự thân, công thức lực hấp dẫn và các tham số xử lý khoảng cách đặc biệt.

## 2.3. Các chỉ số đánh giá

**A. Sai số vector gia tốc của từng vật thể**

\[
E_{a,i}=
\left\|
\vec a_{BH,i}-\vec a_{Direct,i}
\right\|_2
\]

Chỉ số này đánh giá độ lệch của vector gia tốc, bao gồm cả độ lớn và hướng.

**B. Gia tốc Vector RMSE**

\[
RMSE_a=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
\vec a_{BH,i}-\vec a_{Direct,i}
\right\|_2^2
}
\]

Sử dụng chỉ số này để đánh giá sai số gia tốc trung bình theo bình phương trên toàn bộ vật thể.

**C. Gia tốc Magnitude MAPE**

Trước tiên, tính độ lớn gia tốc:

\[
|\vec a_i|=\sqrt{a_{i,x}^2+a_{i,y}^2+a_{i,z}^2}
\]

Sau đó:

\[
MAPE_a=
\frac{100\%}{N}
\sum_{i=1}^{N}
\left|
\frac{
|\vec a_{BH,i}|-|\vec a_{Direct,i}|
}{
|\vec a_{Direct,i}|
}
\right|
\]

MAPE đánh giá sai lệch phần trăm về độ lớn gia tốc giữa hai chương trình.

Lưu ý: Không thực hiện phép chia khi độ lớn gia tốc tham chiếu bằng 0 hoặc quá nhỏ. Cần xác định một ngưỡng loại mẫu, lưu số lượng mẫu bị loại và ngưỡng đã sử dụng. Không được tự động thay sai số bằng 0.

## 2.4. Kết quả cần lưu

Đối với từng cấu hình số lượng vật thể, phân bố không gian và theta, lưu:

- ID vật thể.
- Gia tốc Direct N-body theo X, Y, Z.
- Gia tốc Barnes–Hut theo X, Y, Z.
- Độ lớn gia tốc của hai chương trình.
- Sai số vector gia tốc từng vật thể.
- Vector RMSE.
- Magnitude MAPE.
- Số lượng mẫu được sử dụng và bị loại khỏi phép tính MAPE.

In các kết quả tổng hợp ra terminal để có thể theo dõi trực tiếp từng cấu hình kiểm thử.

Chạy riêng giai đoạn này bằng `gravity_sim_3Dgrid --acceleration-benchmark`. Chế độ này tạo
snapshot kiểm thử xác định, chỉ tính gia tốc và không chạy bước tích phân.

---

# GIAI ĐOẠN 3: ĐÁNH GIÁ SAI SỐ QUỸ ĐẠO SAU QUÁ TRÌNH MÔ PHỎNG

## 3.1. Mục tiêu

Đánh giá mức độ sai lệch vị trí và vận tốc của từng vật thể giữa Direct N-body và Barnes–Hut sau nhiều bước mô phỏng.

Giai đoạn này nhằm xác định sai số gia tốc có ảnh hưởng như thế nào đến quỹ đạo theo thời gian.

Khác với giai đoạn 2, phép kiểm thử này phải thực hiện tích phân và cập nhật trạng thái vật thể qua 960 bước mô phỏng.

## 3.2. Cách thực hiện

**Bước 1: Khởi tạo hai chương trình**

Sử dụng cùng một tập dữ liệu ban đầu đã được xác định ở phần thiết lập chung.

Đảm bảo các vật thể có cùng ID, khối lượng, vị trí và vận tốc ban đầu.

**Bước 2: Chạy mô phỏng Direct N-body**

Chạy chương trình baseline với:

- `dt = 1.0 / 96.0`.
- `T = 10.0`.
- 960 bước mô phỏng.

Lưu trạng thái vị trí và vận tốc của từng vật thể theo ID tại các thời điểm:

- \(t=0\).
- \(t=1\).
- \(t=2\).
- ...
- \(t=10\) giây.

**Bước 3: Chạy mô phỏng Barnes–Hut GPU**

Sử dụng chính xác dữ liệu ban đầu của Baseline.

Chạy với cùng timestep, tổng thời gian mô phỏng và số bước mô phỏng.

Thực hiện riêng từng giá trị theta:

- `theta = 0.3`.
- `theta = 0.5`.
- `theta = 0.8`.

Lưu trạng thái vị trí và vận tốc theo cùng các thời điểm lấy mẫu.

**Bước 4: So sánh quỹ đạo**

Tại từng thời điểm lấy mẫu, đối chiếu trạng thái của từng vật thể giữa hai chương trình theo ID.

Không so sánh hai vật thể khác ID hoặc hai thời điểm khác nhau.

**Bước 5: Kiểm tra tính nhất quán của tích phân**

Kiểm tra cách cập nhật trạng thái của hai chương trình.

Đặc biệt, phải đảm bảo các vật thể trong cùng một bước mô phỏng đều sử dụng trạng thái đầu bước để tính gia tốc, sau đó mới cập nhật đồng bộ vị trí và vận tốc.

Nếu Baseline cập nhật trực tiếp từng vật thể theo kiểu tuần tự khiến vật thể phía sau sử dụng trạng thái đã được cập nhật của vật thể phía trước, cần xử lý sự khác biệt này trước khi so sánh quỹ đạo.

Có thể sử dụng hai mảng trạng thái CurrentState và NextState hoặc cơ chế tương đương để đảm bảo việc cập nhật đồng bộ.

Hai chương trình phải sử dụng cùng quy tắc tích phân và cùng quy ước thứ tự tính toán. Không được thay đổi thuật toán tích phân của một chương trình mà không kiểm tra ảnh hưởng đến tính công bằng của phép so sánh.

Hai chương trình cùng áp dụng quy tắc va chạm hiện có: với mỗi vật thể chồng lấn, nhân vận tốc sau khi cộng gia tốc với `-0.2`. Cả hai dùng Semi-implicit Euler: cập nhật vận tốc trước, sau đó dùng vận tốc mới để cập nhật vị trí. Baseline tính trạng thái bằng `double`; trạng thái trên GPU dùng `float`, vì vậy sai số số học do độ chính xác lưu trữ cũng có thể xuất hiện.

## 3.3. Các chỉ số đánh giá

**A. Sai số vector vị trí từng vật thể**

\[
E_{p,i}=
\left\|
\vec p_{BH,i}-\vec p_{Direct,i}
\right\|_2
\]

Đây là độ lệch vị trí thực tế của vật thể trong không gian 3D.

**B. Position RMSE**

\[
Position\ RMSE=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
\vec p_{BH,i}-\vec p_{Direct,i}
\right\|_2^2
}
\]

Sử dụng chỉ số này để tổng hợp sai số vị trí của toàn bộ vật thể tại mỗi thời điểm lấy mẫu.

**C. Position Magnitude MAPE**

Tính độ lớn vector vị trí của từng vật thể:

\[
|\vec p_i|=
\sqrt{p_{i,x}^2+p_{i,y}^2+p_{i,z}^2}
\]

Sau đó tính:

\[
MAPE_p=
\frac{100\%}{N}
\sum_{i=1}^{N}
\left|
\frac{
|\vec p_{BH,i}|-|\vec p_{Direct,i}|
}{
|\vec p_{Direct,i}|
}
\right|
\]

Chỉ số này đánh giá phần trăm sai lệch về độ lớn vị trí của vật thể giữa hai chương trình, tức khoảng cách từ vật thể đến gốc tọa độ.

Lưu ý: Position Magnitude MAPE không đo đầy đủ sai lệch vị trí trong không gian 3D. Hai vật thể có thể nằm ở hai hướng đối diện nhau nhưng có cùng độ lớn vị trí, khiến MAPE bằng 0. Vì vậy, phải sử dụng song song với Position RMSE.

Nếu độ lớn vị trí tham chiếu bằng 0 hoặc quá nhỏ, không được chia trực tiếp. Sử dụng ngưỡng loại mẫu được xác định rõ ràng và lưu số lượng mẫu bị loại.

**D. Velocity RMSE**

\[
Velocity\ RMSE=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
\vec v_{BH,i}-\vec v_{Direct,i}
\right\|_2^2
}
\]

Chỉ số này đánh giá sai lệch vận tốc của toàn bộ vật thể tại từng thời điểm lấy mẫu.

## 3.4. Kết quả cần lưu

Đối với mỗi cấu hình kiểm thử, lưu:

- ID vật thể.
- Thời điểm mô phỏng.
- Vị trí Baseline và Target.
- Vận tốc Baseline và Target.
- Sai số vector vị trí từng vật thể.
- Sai số độ lớn vị trí từng vật thể.
- Position RMSE.
- Position Magnitude MAPE.
- Velocity RMSE.

Các chỉ số tổng hợp phải được tính riêng cho từng thời điểm lấy mẫu, không gộp toàn bộ thời gian thành một giá trị duy nhất mà không lưu thông tin thời điểm.

Chương trình chạy 3 lần cho mỗi cấu hình theo thiết lập chung. Dữ liệu ban đầu được lưu một lần và dùng lại cho cả ba lần chạy.

---

# 4. TỔ CHỨC VÀ LƯU KẾT QUẢ

Lưu toàn bộ kết quả kiểm thử trong thư mục:

`numerical_error/`

Mỗi lần chạy tạo một thư mục riêng theo thời gian, ví dụ:

`numerical_error/run_YYYYMMDD_HHMMSS_mmm/`

Không ghi đè kết quả của những lần chạy trước.

Cấu trúc thư mục đề xuất:

```text
numerical_error/
└── run_YYYYMMDD_HHMMSS_mmm/
    ├── run_metadata.json
    ├── mass_aggregation/
    │   ├── mass_aggregation.csv
    │   └── mass_aggregation_summary.csv
    ├── acceleration/
    │   ├── acceleration_per_object.csv
    │   └── acceleration_summary.csv
    └── trajectory/
        ├── trajectory_per_object.csv
        └── trajectory_summary.csv
```

Trong đó:

- `run_metadata.json`: lưu cấu hình kiểm thử, số lượng vật thể, kiểu phân bố, random seed, timestep, tổng thời gian, theta và các thông tin cần thiết để tái tạo kết quả.
- `mass_aggregation.csv`: kết quả kiểm tra từng node.
- `mass_aggregation_summary.csv`: kết quả tổng hợp kiểm tra Mass Aggregation.
- `acceleration_per_object.csv`: gia tốc và sai số của từng vật thể.
- `acceleration_summary.csv`: RMSE, MAPE và các chỉ số tổng hợp gia tốc.
- `trajectory_per_object.csv`: vị trí, vận tốc và sai số của từng vật thể theo thời gian.
- `trajectory_summary.csv`: Position RMSE, Position Magnitude MAPE và Velocity RMSE theo từng thời điểm.

Tên cột phải rõ nghĩa, có đơn vị nếu đại lượng có đơn vị đo.

Ví dụ:

- `mass_gpu`, `mass_reference`, `mass_absolute_error`.
- `com_gpu_x`, `com_reference_x`, `com_absolute_error`.
- `acceleration_rmse`, `acceleration_magnitude_mape`.
- `position_rmse`, `position_magnitude_mape`, `velocity_rmse`.

## 5. Yêu cầu triển khai code

1. Sử dụng `error_utils.cpp` để triển khai các hàm tính sai số. Có thể tạo `error_utils.h` để khai báo hàm nếu cấu trúc hiện tại cần thiết và dùng Function Pointer 

2. Chia các hàm theo mục đích rõ ràng, chẳng hạn:
   - Tính sai số tuyệt đối của khối lượng.
   - Tính sai số vector.
   - Tính RMSE.
   - Tính MAPE cho độ lớn gia tốc.
   - Tính MAPE cho độ lớn vị trí.

3. Ưu tiên code đơn giản, dễ đọc và dễ hiểu ở trình độ sinh viên. Sử dụng tên biến và tên hàm thể hiện rõ mục đích.

4. Không sử dụng template phức tạp, kiến trúc trừu tượng không cần thiết hoặc tạo quá nhiều lớp chỉ để phục vụ việc tính sai số.

5. Các phép tính phải có kiểm tra dữ liệu không hợp lệ, đặc biệt là NaN, Inf, phép chia cho 0 và dữ liệu tham chiếu quá nhỏ.

6. Các hàm tính sai số phải độc lập với luồng hiển thị và render. Không đưa logic tính sai số trực tiếp vào shader render hoặc các hàm xử lý giao diện.

7. In kết quả tổng hợp của từng cấu hình kiểm thử ra terminal. Cần hiển thị rõ số lượng vật thể, kiểu phân bố, theta, thời điểm mô phỏng và các chỉ số sai số tương ứng.

8. Không sử dụng RMS thay cho RMSE. Chỉ triển khai những chỉ số đã nêu trong kế hoạch.

9. Không đưa phép đo hiệu năng vào kế hoạch này. Không thêm các chỉ số thời gian thực thi, FPS, CPU usage hoặc GPU usage.

10. Ưu tiên thay đổi ít nhất có thể đối với mã nguồn mô phỏng hiện tại. Những phần phục vụ benchmark nên được tổ chức riêng để không làm thay đổi luồng mô phỏng thông thường.

Chạy từng giai đoạn bằng một tùy chọn riêng:

- `gravity_sim_3Dgrid --mass-aggregation-benchmark`
- `gravity_sim_3Dgrid --acceleration-benchmark`
- `gravity_sim_3Dgrid --trajectory-benchmark`

Mỗi lần chạy tạo một thư mục kết quả mới trong `numerical_error`; không ghi đè kết quả trước đó.

## 6. Tiêu chuẩn hoàn thành

- Giai đoạn 1 xác minh được tổng khối lượng và tâm khối của các node so với kết quả tham chiếu CPU.
- Giai đoạn 2 tính được sai số gia tốc giữa Direct N-body và Barnes–Hut trên cùng snapshot.
- Giai đoạn 3 tính được sai số vị trí và vận tốc của từng vật thể theo thời gian.
- Tất cả phép so sánh đều sử dụng cùng dữ liệu đầu vào và đối chiếu vật thể theo ID.
- Các cấu hình kiểm thử có thể tái tạo bằng thông tin lưu trong `run_metadata.json`.
- Các chỉ số được lưu thành CSV và kết quả tổng hợp được in ra terminal.
- Không ghi đè kết quả của những lần chạy trước.
- Code dễ đọc, có chú thích cho các công thức quan trọng và không làm thay đổi chức năng mô phỏng hiện tại ngoài những điều chỉnh cần thiết để đảm bảo phép so sánh công bằng.

**Thứ tự ưu tiên đánh giá:** Mass Aggregation phải được xác minh trước; tiếp theo là sai số gia tốc trên snapshot cố định; cuối cùng là sai số quỹ đạo sau mô phỏng. Không sử dụng kết quả của một giai đoạn chưa được xác minh để kết luận về độ chính xác của giai đoạn tiếp theo.