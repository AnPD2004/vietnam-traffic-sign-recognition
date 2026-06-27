import { Link } from "react-router-dom";

const FOOTER_LINKS = [
  {
    title: "Hệ thống",
    links: [
      { to: "/", label: "Trang chủ" },
      { to: "/tinh-nang", label: "Tính năng nhận dạng" },
      { to: "/#tinh-nang", label: "Tính năng nền tảng" },
      { to: "/#pipeline", label: "Kiến trúc pipeline" },
    ],
  },
  {
    title: "Nghiên cứu",
    links: [
      { to: "/#pipeline", label: "So sánh pipeline" },
      { to: "/#faq", label: "Câu hỏi thường gặp" },
      { to: "/tinh-nang", label: "Demo benchmark" },
    ],
  },
  {
    title: "Tài khoản",
    links: [
      { to: "/#dang-nhap", label: "Đăng nhập" },
      { to: "/#dang-ky", label: "Đăng ký" },
    ],
  },
];

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer-inner site-footer-inner--rich">
        <div className="site-footer__brand">
          <p className="site-footer__mark">VTSR</p>
          <p className="site-footer__name">
            HỆ THỐNG NHẬN DẠNG BIỂN BÁO GIAO THÔNG VIỆT NAM
          </p>
          <p className="site-footer__note">
            Nền tảng nghiên cứu và demo so sánh pipeline YOLO end-to-end và
            YOLO&nbsp;+&nbsp;CNN trên 52 lớp biển báo
            và triển khai thử nghiệm.
          </p>
        </div>

        <div className="site-footer__columns">
          {FOOTER_LINKS.map((group) => (
            <div key={group.title} className="site-footer__col">
              <h3 className="site-footer__col-title">{group.title}</h3>
              <ul className="site-footer__links">
                {group.links.map((link) => (
                  <li key={link.label}>
                    <Link to={link.to} className="site-footer__link">
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      <div className="site-footer-bottom">
        <div className="site-footer-bottom-inner">
          <p className="site-footer__copy">
            © {new Date().getFullYear()} Vietnamese Traffic Sign Recognition
          </p>
          <p className="site-footer__meta">
            PyTorch · Ultralytics YOLO · Flask · React
          </p>
        </div>
      </div>
    </footer>
  );
}
