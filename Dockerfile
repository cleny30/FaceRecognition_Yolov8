# Sử dụng image Python 3.12
FROM python:3.12-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các công cụ build cần thiết (CMake, build-essential, v.v.)
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev

# Copy các file cần thiết vào container
COPY . .

# Cài đặt các dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose cổng mà ứng dụng sẽ chạy
EXPOSE 5001

# Lệnh để chạy ứng dụng
CMD ["python", "app.py"]