import { useState } from "react";
import { Link } from "react-router-dom";
import Navbar from "../components/Navbar.jsx";
import SiteFooter from "../components/SiteFooter.jsx";

const HIGHLIGHTS = [
  {
    value: "52",
    label: "Lớp biển báo",
    desc: "Phủ các nhóm biển báo giao thông phổ biến tại Việt Nam",
  },
  {
    value: "2",
    label: "Pipeline",
    desc: "YOLO end-to-end và mô hình lai YOLO + CNN",
  },
  {
    value: "3.2K+",
    label: "Ảnh huấn luyện",
    desc: "Tập dữ liệu detection và classification được chuẩn hóa",
  },
];

const PIPELINES = [
  {
    title: "Pipeline 1 — YOLO End-to-End",
    desc: "Một mô hình YOLO thực hiện đồng thời phát hiện vị trí và phân loại biển báo, tối ưu cho triển khai gọn nhẹ.",
  },
  {
    title: "Pipeline 2 — YOLO + CNN",
    desc: "YOLO định vị vùng biển báo, CNN chuyên biệt phân loại chi tiết — phù hợp khi cần độ chính xác phân loại cao hơn.",
  },
];

function AuthForm({ id, title, subtitle, fields, submitLabel, onSubmit }) {
  const [notice, setNotice] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    setNotice("Tính năng xác thực đang được phát triển. Vui lòng quay lại sau.");
    onSubmit?.();
  }

  return (
    <section id={id} className="card auth-card" aria-labelledby={`${id}-title`}>
      <div className="auth-card__header">
        <h2 id={`${id}-title`} className="section-title">
          {title}
        </h2>
        <p className="section-desc">{subtitle}</p>
      </div>
      <form className="auth-form" onSubmit={handleSubmit}>
        {fields.map((field) => (
          <label key={field.id} className="form-field">
            <span className="form-label">{field.label}</span>
            <input
              className="form-input"
              type={field.type}
              id={field.id}
              name={field.name}
              placeholder={field.placeholder}
              autoComplete={field.autoComplete}
              required={field.required !== false}
            />
          </label>
        ))}
        <button type="submit" className="btn btn--primary auth-form__submit">
          {submitLabel}
        </button>
        {notice && <p className="auth-notice">{notice}</p>}
      </form>
    </section>
  );
}

export default function HomePage() {
  return (
    <div className="app">
      <Navbar />
      <main className="home">
        <section className="hero">
          <div className="hero-inner">
            <h1 className="hero-title">
              HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG VIỆT NAM
            </h1>
            <p className="hero-lead">
              Nền tảng nghiên cứu và demo so sánh hai hướng tiếp cận nhận dạng
              biển báo: mô hình YOLO đơn khối và pipeline lai YOLO&nbsp;+&nbsp;CNN.
              Hỗ trợ nhận dạng trên ảnh và video trong môi trường thử nghiệm.
            </p>
            <div className="hero-actions">
              <Link to="/tinh-nang" className="btn btn--primary">
                Dùng thử tính năng
              </Link>
              <a href="#dang-nhap" className="btn btn--ghost">
                Đăng nhập
              </a>
            </div>
          </div>
        </section>

        <section className="home-section" aria-labelledby="highlights-title">
          <div className="home-section-inner">
            <h2 id="highlights-title" className="home-section__title">
              Tổng quan hệ thống
            </h2>
            <div className="highlight-grid">
              {HIGHLIGHTS.map((item) => (
                <article key={item.label} className="highlight-card card">
                  <p className="highlight-card__value">{item.value}</p>
                  <h3 className="highlight-card__label">{item.label}</h3>
                  <p className="highlight-card__desc">{item.desc}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="home-section" aria-labelledby="pipelines-title">
          <div className="home-section-inner">
            <h2 id="pipelines-title" className="home-section__title">
              Hai pipeline nghiên cứu
            </h2>
            <div className="pipeline-overview">
              {PIPELINES.map((item) => (
                <article key={item.title} className="pipeline-overview__card card">
                  <h3 className="pipeline-overview__title">{item.title}</h3>
                  <p className="pipeline-overview__desc">{item.desc}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="home-section" aria-labelledby="auth-title">
          <div className="home-section-inner">
            <h2 id="auth-title" className="home-section__title">
              Tài khoản người dùng
            </h2>
            <p className="home-section__lead">
              Đăng nhập hoặc đăng ký để lưu lịch sử nhận dạng và quản lý phiên
              làm việc. Giao diện sẵn sàng — chức năng backend sẽ được bổ sung
              trong giai đoạn tiếp theo.
            </p>
            <div className="auth-grid">
              <AuthForm
                id="dang-nhap"
                title="Đăng nhập"
                subtitle="Truy cập tài khoản đã đăng ký để sử dụng hệ thống."
                submitLabel="Đăng nhập"
                fields={[
                  {
                    id: "login-email",
                    name: "email",
                    label: "Email",
                    type: "email",
                    placeholder: "name@example.com",
                    autoComplete: "email",
                  },
                  {
                    id: "login-password",
                    name: "password",
                    label: "Mật khẩu",
                    type: "password",
                    placeholder: "••••••••",
                    autoComplete: "current-password",
                  },
                ]}
              />
              <AuthForm
                id="dang-ky"
                title="Đăng ký"
                subtitle="Tạo tài khoản mới để theo dõi kết quả nhận dạng."
                submitLabel="Đăng ký"
                fields={[
                  {
                    id: "register-name",
                    name: "name",
                    label: "Họ và tên",
                    type: "text",
                    placeholder: "Nguyễn Văn A",
                    autoComplete: "name",
                  },
                  {
                    id: "register-email",
                    name: "email",
                    label: "Email",
                    type: "email",
                    placeholder: "name@example.com",
                    autoComplete: "email",
                  },
                  {
                    id: "register-password",
                    name: "password",
                    label: "Mật khẩu",
                    type: "password",
                    placeholder: "Tối thiểu 8 ký tự",
                    autoComplete: "new-password",
                  },
                ]}
              />
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
