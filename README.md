# Bài toán dự đoán nồng độ bụi mịn PM2.5 tại Thành Phố HCM.
## Sinh viên: Nguyễn Văn Tú - 19021381
Dự đoán nồng độ bụi mịn PM2.5 trên một trạm quan trắc tại Thành phố Hồ Chí Minh, tham khảo và triển khai theo bài báo [Prediction of PM2.5 Concentration Based on the LSTM-TSLightGBM Variable Weight Combination Model](https://www.mdpi.com/2073-4433/12/9/1211/htm) của tác giả [Xuchu Jiang](https://sciprofiles.com/profile/1563874) và các cộng sự xuất bản năm 2021.

# I. Tải dữ liệu
## Tải dữ liệu khí tượng
- Dữ liệu khí tượng được tải từ trang [Dữ liệu khí tượng](https://weather.uwyo.edu/surface/meteorogram/seasia.shtml)
- Sử dụng GET request để truy vấn dữ liệu từ web server và crawl về local.
- File dùng để craw dữ liệu từ trang web trên `src/download_meteorology.js`.
> Để chạy được file crawl dữ liệu này, cần cài node version >= 16.15.0

> Và cài đặt thêm các các thư viện yêu cầu:
> - npm i axios
> - npm i jsdom
> - npm i objects-to-csv

### Chạy crawl dữ liệu:
Giả sử bạn đang ở thư mục gốc sau khi clone repo này về, sau đó chạy các lệnh sau: 
```console
$ mkdir data data/raw_data data/raw_data/hcm_meteorological_data 
$ cd src
$ node src/download_meteorology_data.js
```
- Dữ liệu được tải về sẽ được lưu trong folder data: bao gồm nhiều file csv mỗi file tương ứng dữ liệu khí
tượng của một ngày.
- Mất 1 ngày dữ liệu 20190111.csv

## Tải dữ liệu PM2.5
Tải về [Dữ liệu PM2.5](https://www.airnow.gov/international/us-embassies-and-consulates/) của HCM trong 3 năm (2019, 2020, 2021).

# II. Tiền xử lý dữ liệu 
### Công việc thực hiện
+ Gộp các file csv dữ liệu thành cơ sở dữ liệu 3 năm (dữ liệu lưu theo 24 điểm tương ứng 24h trong ngày).
+ Xóa các cột dữ liệu bị thiếu - không mang lại thông tin hiệu quả.
+ Số hóa các cột dữ liệu đang ở dạng string
+ Format lại các biểu diễn dữ liệu (cột thời gian)
+ Áp dụng kỹ thuật Linear Interpolation để gen ra các dữ liệu thiếu (khoảng cách <3h). Nếu lớn hơn khoảng dữ liệu bị thiếu lớn hơn 3h ta sẽ loại bỏ.
#### Chi tiết quy trình, code tiền xử lý được mô tả rõ trong file data/preprocess_data.ipynb
#### Cuối cùng tạo ra file dữ liệu được làm sạch: clean_data.csv

# III. Huấn luyện mô hình
Module huấn luyện mô hình LSTM-TSLightGBM kết hợp, và tự động hóa cho tất cả các khung thời gian, ví dụ: 
+ Dùng dữ liệu 8 tiếng liên tục để dự đoán nồng độ bụi mịn 4h tiếp theo ( window_size = 8, stride_pred = 4 )

Thí nghiệm lấy tổ hợp nhiều giá trị window_size khác nhau để dự đoán các khoảng cách stride_pred khác nhau

    
    {window_sizes = [4, 8, 10, 12, 16, 18, 24, 32]} ...dự đoán... {stride_preds = [1, 2, 4, 8]}

Huấn luyện mô hình LSTM-TSLightGBM kết hợp, trước hết cần huấn luyện 2 mô hình đơn là LSTM

và LightGBM sau đó kết hợp chúng lại với trọng số là e1, e2. Quá trình này được mô tả rõ ràng gói gọn trong

module `main`.

### Đào tạo trên CPU local
Đã tự động lưu trữ các metrics đầu ra cho các mô hình.

Bạn đang ở thư mục gốc của project

Sau khi clone, sử dụng Pycharm tạo môi trường conda và cài đặt requirements.txt
```console
$ pip install -r requirements.txt
```
```console
$ python3 -m src.main
```

### Đào tạo nhanh với GPU trên colab
Tạo notebook colab rồi lần lượt chạy lệnh sau
```console
!git clone "https://github.com/tunnvv/predict-pm2.5-hcm.git"
```
```console
%cd /content/predict-pm2.5-hcm
```
```console
!python3 -m src.main
```
Zip file kết quả và download về
```console
!zip -r /content/predict-pm2.5-hcm/output.zip /content/predict-pm2.5-hcm/output
```
# IV. Test mô hình
Test hiệu quả dự đoán của mô hình LSTM-TSLightGBM trên tập test.

Mô hình đã train có tên 1h-4T, 1h-8T, ... 8h-32T tương ứng với khoảng cách dự đoán và window_size muốn test
```console
$ python3 src.api_test --model_name
```
