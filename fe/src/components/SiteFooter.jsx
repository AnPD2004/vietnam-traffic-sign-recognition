export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer-inner">
        <p className="site-footer__copy">
          © {new Date().getFullYear()} Hệ thống Nhận dạng Biển báo Giao thông
          Việt Nam — Đồ án tốt nghiệp
        </p>
        <p className="site-footer__note">
          Nghiên cứu so sánh pipeline YOLO end-to-end và YOLO&nbsp;+&nbsp;CNN
          trên 52 lớp biển báo.
        </p>
      </div>
    </footer>
  );
}
