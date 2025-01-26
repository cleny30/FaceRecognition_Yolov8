# Sử dụng image Python 3.9
FROM python:3.9-slim
# Thiết lập thư mục làm việc
WORKDIR /app
# Copy các file cần thiết vào container
COPY . .
# Cài đặt các dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose cổng mà ứng dụng sẽ chạy
EXPOSE 5001
# Lệnh để chạy ứng dụng
CMD ["python", "app.py"]