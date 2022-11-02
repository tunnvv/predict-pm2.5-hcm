# Bài toán dự đoán nồng độ bụi mịn PM2.5 tại Thành Phố HCM.
## Sinh viên: Nguyễn Văn Tú - 19021381

# Tải dữ liệu
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
- Nó bị lỗi, dừng lại vì mất 1 ngày dữ liệu 20190111.csv, nhưng 1 ngày ko đáng kể trong 3 năm dữ liệu thu về 

## Tải dữ liệu PM2.5
Tải về [Dữ liệu PM2.5](https://www.airnow.gov/international/us-embassies-and-consulates/) của HCM trong 3 năm (2019, 2020, 2021).

# Tiền xử lý dữ liệu 
### Công việc thực hiện
+ Gộp các file csv dữ liệu thành cơ sở dữ liệu 3 năm.
+ Xóa các cột dữ liệu bị thiếu - không mang lại thông tin hiệu quả.
+ Số hóa các cột dữ liệu đang ở dạng string
+ Format lại các biểu diễn dữ liệu (cột thời gian)
+ Áp dụng kỹ thuật Linear Interpolation để gen ra các cột dữ liệu thiếu (khoảng cách < 3h). Nếu lớn hơn ta sẽ loại bỏ.

#### Chi tiết quy trình, code tiền xử lý được mô tả rõ trong file data/preprocess_data.ipynb
#### Cuối cùng tạo ra file dữ liệu được làm sạch: clean_data.csv
