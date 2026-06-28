export const HERO_STATS = [
  { value: "52", label: "Lớp biển báo" },
  { value: "3", label: "Mô hình YOLO" },
  { value: "2", label: "Mô hình CNN" },
  { value: "3.2K+", label: "Ảnh huấn luyện" },
];

export const PLATFORM_FEATURES = [
  {
    id: "image",
    title: "Nhận dạng ảnh tĩnh",
    desc: "Tải ảnh đơn lẻ, chạy song song hai pipeline và đối chiếu bbox cùng nhãn phân loại trên cùng một khung hình.",
  },
  {
    id: "video",
    title: "Xử lý video theo frame",
    desc: "Phân tích chuỗi khung hình với tracking ổn định, phù hợp mô phỏng camera hành trình và camera giám sát.",
  },
  {
    id: "compare",
    title: "So sánh pipeline song song",
    desc: "Đặt cạnh nhau kết quả YOLO end-to-end và YOLO + CNN để đánh giá trade-off giữa tốc độ và độ chính xác.",
  },
  {
    id: "metrics",
    title: "Báo cáo metric chi tiết",
    desc: "Hiển thị thời gian suy luận, số lượng phát hiện, danh sách biển báo và ảnh kết quả có chú thích trực quan.",
  },
  {
    id: "dataset",
    title: "Tập dữ liệu chuẩn hóa",
    desc: "52 lớp biển báo Việt Nam với annotation YOLO và tập crop phục vụ huấn luyện CNN.",
  },
  {
    id: "api",
    title: "API & triển khai web",
    desc: "Backend Flask phục vụ inference; giao diện React cho demo, benchmark và mở rộng triển khai thực tế.",
  },
];

export const WORKFLOW_STEPS = [
  {
    step: "01",
    title: "Tải dữ liệu đầu vào",
    desc: "Người dùng cung cấp ảnh hoặc video qua giao diện web. Hệ thống kiểm tra định dạng và chuẩn bị buffer xử lý.",
  },
  {
    step: "02",
    title: "Phát hiện vùng biển báo",
    desc: "Mô hình YOLO xác định bounding box và điểm tin cậy cho từng biển báo trong khung hình.",
  },
  {
    step: "03",
    title: "Phân loại chi tiết",
    desc: "Pipeline 1 dùng đầu ra class của YOLO; Pipeline 2 crop vùng quan tâm và đưa qua mạng CNN chuyên biệt.",
  },
  {
    step: "04",
    title: "Trực quan hóa & so sánh",
    desc: "Kết quả được render lên ảnh/video đầu ra kèm bảng thống kê để đối chiếu hiệu năng giữa hai hướng tiếp cận.",
  },
];

export const PIPELINE_DETAILS = [
  {
    id: "flow1",
    tag: "Pipeline 1",
    name: "YOLO End-to-End",
    accent: "flow1",
    summary:
      "Một mô hình duy nhất đảm nhiệm detection và classification — tối ưu cho triển khai gọn, ít thành phần.",
    models: ["YOLOv5n", "YOLOv8n", "YOLOv11n"],
    strengths: [
      "Kiến trúc đơn giản, dễ triển khai edge",
      "Thời gian suy luận thấp trên GPU/CPU",
      "Phù hợp ứng dụng real-time",
    ],
    metrics: ["mAP@0.5", "mAP@0.5:0.95", "FPS", "Model size"],
  },
  {
    id: "flow2",
    tag: "Pipeline 2",
    name: "YOLO + CNN Hybrid",
    accent: "flow2",
    summary:
      "Tách bước định vị và phân loại — YOLO tìm vùng, CNN chuyên sâu phân loại trên ảnh crop.",
    models: ["YOLO (detector)", "ResNet50", "EfficientNet-B0"],
    strengths: [
      "Phân loại chuyên biệt trên vùng crop",
      "Linh hoạt thay thế backbone CNN",
      "Phù hợp khi ưu tiên độ chính xác nhãn",
    ],
    metrics: ["Accuracy", "F1-score", "Confusion matrix", "Inference time"],
  },
];

export const PIPELINE_COMPARE_ROWS = [
  { label: "Phát hiện vị trí", flow1: "YOLO", flow2: "YOLO" },
  { label: "Phân loại nhãn", flow1: "YOLO (đầu ra class)", flow2: "CNN trên ảnh crop" },
  { label: "Số mô hình suy luận", flow1: "1", flow2: "2" },
  { label: "Độ phức tạp triển khai", flow1: "Thấp", flow2: "Trung bình" },
  { label: "Ưu tiên tối ưu", flow1: "Tốc độ & gọn nhẹ", flow2: "Độ chính xác phân loại" },
  { label: "Đầu ra demo", flow1: "BBox + class + metrics", flow2: "BBox + class + metrics" },
];

export const BENCHMARK_STAGES = [
  {
    stage: "Giai đoạn 1",
    title: "Benchmark YOLO",
    desc: "So sánh YOLOv5n, YOLOv8n, YOLOv11n; nghiên cứu siêu tham số image size, learning rate, batch size và mosaic.",
    output: "Cấu hình YOLO tối ưu",
  },
  {
    stage: "Giai đoạn 2",
    title: "Benchmark CNN",
    desc: "Đánh giá ResNet50 và EfficientNet-B0 trên tập crop; thử learning rate và chiến lược transfer learning.",
    output: "Cấu hình CNN tối ưu",
  },
  {
    stage: "Giai đoạn 3",
    title: "So sánh pipeline",
    desc: "Đối chiếu trực tiếp pipeline end-to-end và hybrid trên cùng tập kiểm thử và metric thống nhất.",
    output: "Pipeline triển khai đề xuất",
  },
];

export const USE_CASES = [
  {
    title: "ADAS",
    desc: "Hỗ trợ cảnh báo tốc độ, cấm vượt và biển chỉ dẫn trên hệ thống hỗ trợ lái xe tiên tiến.",
  },
  {
    title: "Xe tự lái",
    desc: "Cung cấp nhận thức biển báo real-time làm đầu vào cho module lập kế hoạch và điều khiển.",
  },
  {
    title: "Giao thông thông minh",
    desc: "Phân tích luồng camera giao thông để giám sát tuân thủ và thu thập dữ liệu vận hành.",
  },
  {
    title: "Giám sát an toàn",
    desc: "Tích hợp vào hệ thống camera hành trình hoặc trung tâm điều hành giao thông.",
  },
];

export const TECH_STACK = [
  { name: "Python", category: "Ngôn ngữ" },
  { name: "PyTorch", category: "Deep Learning" },
  { name: "Ultralytics YOLO", category: "Detection" },
  { name: "Torchvision", category: "Classification" },
  { name: "Flask", category: "Backend" },
  { name: "React", category: "Frontend" },
  { name: "OpenCV", category: "Xử lý ảnh" },
  { name: "NumPy / Pandas", category: "Dữ liệu" },
];

export const FAQ_ITEMS = [
  {
    q: "Hệ thống hỗ trợ những định dạng file nào?",
    a: "Trang tính năng hỗ trợ ảnh tĩnh (JPG, PNG, …) và video phổ biến. File được xử lý qua API Flask và trả về ảnh/video có chú thích cùng bảng kết quả.",
  },
  {
    q: "Sự khác biệt chính giữa hai pipeline là gì?",
    a: "Pipeline 1 dùng một mô hình YOLO cho cả phát hiện và phân loại. Pipeline 2 dùng YOLO để định vị, sau đó CNN phân loại trên vùng crop — phù hợp khi cần tách bước và tinh chỉnh classifier.",
  },
  {
    q: "Dữ liệu huấn luyện có bao nhiêu lớp biển báo?",
    a: "Hệ thống được xây dựng trên tập 52 lớp biển báo giao thông Việt Nam, gồm khoảng 3.216 ảnh detection và tập crop phục vụ huấn luyện CNN.",
  },
  {
    q: "Tôi có cần đăng nhập để dùng thử không?",
    a: "Hiện tại bạn có thể truy cập trang Tính năng và chạy demo mà không cần đăng nhập. Tài khoản người dùng sẽ được kích hoạt ở giai đoạn tiếp theo để lưu lịch sử nhận dạng.",
  },
  {
    q: "Kết quả benchmark được lưu ở đâu?",
    a: "Artifact huấn luyện và báo cáo benchmark được lưu trong thư mục artifacts/pipeline1 và artifacts/pipeline2, bao gồm registry, runs và metadata best model.",
  },
];
