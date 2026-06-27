import { NavLink, Link } from "react-router-dom";

const NAV_LINKS = [
  { to: "/", label: "Trang chủ", end: true },
  { to: "/tinh-nang", label: "Tính năng" },
];

export default function Navbar() {
  return (
    <nav className="navbar" aria-label="Điều hướng chính">
      <div className="navbar-inner">
        <Link to="/" className="navbar-brand">
          <span className="navbar-brand__mark">VTSR</span>
          <span className="navbar-brand__text">
            NHẬN DẠNG BIỂN BÁO GIAO THÔNG VIỆT NAM
          </span>
        </Link>

        <div className="navbar-links">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) =>
                `navbar-link${isActive ? " navbar-link--active" : ""}`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </div>

        <div className="navbar-actions">
          <Link to="/#dang-nhap" className="btn btn--ghost btn--sm">
            Đăng nhập
          </Link>
          <Link to="/#dang-ky" className="btn btn--primary btn--sm">
            Đăng ký
          </Link>
        </div>
      </div>
    </nav>
  );
}
