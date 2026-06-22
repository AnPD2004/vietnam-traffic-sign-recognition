export const PIPELINES = [
  {
    id: 1,
    title: "Pipeline 1",
    badge: "YOLO End-to-End",
    variant: "flow1",
    description:
      "Một mô hình YOLO thực hiện đồng thời phát hiện vị trí và phân loại biển báo.",
  },
  {
    id: 2,
    title: "Pipeline 2",
    badge: "YOLO + CNN",
    variant: "flow2",
    description:
      "YOLO định vị vùng biển báo; mạng CNN phân loại chi tiết trên ảnh crop.",
  },
];

export const TABS = [
  {
    id: "image",
    label: "Ảnh tĩnh",
    description:
      "Tải lên một ảnh và chạy song song cả hai pipeline để so sánh kết quả.",
  },
  {
    id: "video",
    label: "Video",
    description:
      "Tải lên video; hệ thống xử lý từng frame với tracking và ổn định bbox.",
  },
];
