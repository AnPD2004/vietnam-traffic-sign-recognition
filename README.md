# Vietnamese Traffic Sign Recognition

## Chủ đề

Dự án này tập trung vào việc xây dựng hệ thống nhận diện và phân loại biển báo giao thông Việt Nam sử dụng Deep Learning. Hệ thống có khả năng phát hiện vị trí và xác định loại biển báo trong ảnh, hỗ trợ các ứng dụng an toàn giao thông và tự động hóa phương tiện.

## Nội dung

### Mục đích

- Phát triển mô hình AI có thể nhận diện chính xác các loại biển báo giao thông phổ biến tại Việt Nam.
- So sánh hiệu năng giữa hai phương pháp: phát hiện đối tượng (detection) và phân loại hình ảnh (classification).
- Đóng góp vào việc cải thiện an toàn giao thông thông qua công nghệ trí tuệ nhân tạo.

### Ứng dụng

- Hỗ trợ hệ thống lái xe tự động (autonomous vehicles) trong việc nhận diện biển báo.
- Công cụ hỗ trợ người lái xe, cảnh báo biển báo quan trọng.
- Nghiên cứu và phát triển trong lĩnh vực thị giác máy tính cho giao thông thông minh.
- Có thể tích hợp vào ứng dụng di động hoặc hệ thống giám sát giao thông.

## Công nghệ và Thuật toán

### Công nghệ chính

- **Ngôn ngữ lập trình**: Python
- **Thư viện Deep Learning**: PyTorch, Torchvision
- **Framework YOLO**: Ultralytics (YOLOv8)
- **Quản lý dự án**: uv (cho dependency management)

### Thuật toán sử dụng

- **YOLOv8n**: Mô hình phát hiện đối tượng (object detection) để xác định vị trí và phân loại biển báo trong ảnh. YOLOv8n là phiên bản nhẹ của YOLOv8, phù hợp cho việc triển khai trên thiết bị có tài nguyên hạn chế.
- **CNN (Convolutional Neural Network)**: Mô hình phân loại hình ảnh để so sánh với YOLO. Sử dụng mạng nơ-ron tích chập để phân loại biển báo đã được crop từ ảnh gốc.

### So sánh thuật toán

Dự án sẽ so sánh hai phương pháp:
- YOLOv8n: Phát hiện và phân loại trực tiếp trên ảnh toàn cảnh.
- CNN: Phân loại trên vùng biển báo đã được cắt từ ảnh (sử dụng bounding box từ YOLO hoặc nhãn có sẵn).

Tiêu chí so sánh bao gồm độ chính xác, tốc độ suy luận, kích thước mô hình và tính ứng dụng thực tế.