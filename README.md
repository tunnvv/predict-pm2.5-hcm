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

## Tải dữ liệu PM2.5
Tải về [Dữ liệu PM2.5](https://www.airnow.gov/international/us-embassies-and-consulates/) của HCM trong 3 năm (2019, 2020, 2021).
