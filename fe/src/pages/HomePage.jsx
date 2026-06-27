import { Link } from "react-router-dom";
import Navbar from "../components/Navbar.jsx";
import SiteFooter from "../components/SiteFooter.jsx";
import AuthForm from "../components/home/AuthForm.jsx";
import FaqAccordion from "../components/home/FaqAccordion.jsx";
import { FEATURE_ICONS } from "../components/home/HomeIcons.jsx";
import {
  BENCHMARK_STAGES,
  FAQ_ITEMS,
  HERO_STATS,
  PIPELINE_COMPARE_ROWS,
  PIPELINE_DETAILS,
  PLATFORM_FEATURES,
  TECH_STACK,
  USE_CASES,
  WORKFLOW_STEPS,
} from "../data/homeContent.js";

function SectionHeader({ eyebrow, title, lead, id }) {
  return (
    <header className="section-header-block" id={id}>
      {eyebrow && <p className="section-eyebrow">{eyebrow}</p>}
      <h2 className="home-section__title home-section__title--flush">{title}</h2>
      {lead && <p className="home-section__lead home-section__lead--flush">{lead}</p>}
    </header>
  );
}

function FlowDiagram() {
  return (
    <div className="flow-diagram card" aria-hidden="true">
      <p className="flow-diagram__label">Luồng xử lý tổng quát</p>
      <div className="flow-diagram__track">
        <div className="flow-node">
          <span className="flow-node__dot" />
          <span>Ảnh / Video</span>
        </div>
        <div className="flow-arrow" />
        <div className="flow-node flow-node--accent">
          <span className="flow-node__dot" />
          <span>YOLO Detector</span>
        </div>
        <div className="flow-split">
          <div className="flow-branch">
            <div className="flow-arrow flow-arrow--short" />
            <div className="flow-node flow-node--flow1">
              <span className="flow-node__dot" />
              <span>YOLO Class</span>
            </div>
          </div>
          <div className="flow-branch">
            <div className="flow-arrow flow-arrow--short" />
            <div className="flow-node flow-node--flow2">
              <span className="flow-node__dot" />
              <span>CNN Classifier</span>
            </div>
          </div>
        </div>
        <div className="flow-arrow" />
        <div className="flow-node">
          <span className="flow-node__dot" />
          <span>Kết quả & So sánh</span>
        </div>
      </div>
      <div className="flow-diagram__legend">
        <span className="legend-item legend-item--flow1">Pipeline 1</span>
        <span className="legend-item legend-item--flow2">Pipeline 2</span>
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <div className="app">
      <Navbar />
      <main className="home">
        {/* Hero */}
        <section className="hero hero--rich" aria-labelledby="hero-title">
          <div className="hero-grid">
            <div className="hero-copy">
              <p className="hero-eyebrow">Deep Learning · Computer Vision</p>
              <h1 id="hero-title" className="hero-title">
                HỆ THỐNG NHẬN DẠNG BIỂN BÁO GIAO THÔNG VIỆT NAM
              </h1>
              <p className="hero-lead">
                Nền tảng nghiên cứu và demo chuyên nghiệp để so sánh hai hướng tiếp cận
                nhận dạng biển báo — YOLO end-to-end và pipeline lai YOLO&nbsp;+&nbsp;CNN —
                trên tập 52 lớp biển báo chuẩn hóa, hỗ trợ ảnh tĩnh và video.
              </p>
              <div className="hero-actions">
                <Link to="/tinh-nang" className="btn btn--primary">
                  Bắt đầu nhận dạng
                </Link>
                <a href="#tinh-nang" className="btn btn--ghost">
                  Khám phá nền tảng
                </a>
              </div>
              <ul className="hero-badges">
                <li className="hero-badge">Benchmark có hệ thống</li>
                <li className="hero-badge">So sánh song song 2 pipeline</li>
                <li className="hero-badge">Triển khai web-ready</li>
              </ul>
            </div>
            <FlowDiagram />
          </div>
        </section>

        {/* Stats band */}
        <section className="stats-band" aria-label="Chỉ số hệ thống">
          <div className="stats-band-inner">
            {HERO_STATS.map((stat) => (
              <div key={stat.label} className="stats-band__item">
                <p className="stats-band__value">{stat.value}</p>
                <p className="stats-band__label">{stat.label}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Platform features */}
        <section
          id="tinh-nang"
          className="home-section home-section--muted"
          aria-labelledby="features-title"
        >
          <div className="home-section-inner">
            <SectionHeader
              eyebrow="Nền tảng"
              title="Tính năng cốt lõi"
              lead="Bộ công cụ end-to-end từ huấn luyện, benchmark đến demo trực quan — phục vụ nghiên cứu và triển khai thử nghiệm."
            />
            <div className="feature-grid">
              {PLATFORM_FEATURES.map((feature) => {
                const Icon = FEATURE_ICONS[feature.id];
                return (
                  <article key={feature.id} className="feature-card card">
                    <div className="feature-card__icon">{Icon && <Icon />}</div>
                    <h3 className="feature-card__title">{feature.title}</h3>
                    <p className="feature-card__desc">{feature.desc}</p>
                  </article>
                );
              })}
            </div>
          </div>
        </section>

        {/* Workflow */}
        <section className="home-section" aria-labelledby="workflow-title">
          <div className="home-section-inner">
            <SectionHeader
              eyebrow="Quy trình"
              title="Cách hệ thống hoạt động"
              lead="Luồng xử lý thống nhất cho cả hai pipeline, từ đầu vào người dùng đến kết quả có thể đối chiếu."
            />
            <ol className="workflow-list">
              {WORKFLOW_STEPS.map((step) => (
                <li key={step.step} className="workflow-step card">
                  <span className="workflow-step__num">{step.step}</span>
                  <div className="workflow-step__body">
                    <h3 className="workflow-step__title">{step.title}</h3>
                    <p className="workflow-step__desc">{step.desc}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* Pipeline details */}
        <section
          id="pipeline"
          className="home-section home-section--muted"
          aria-labelledby="pipeline-title"
        >
          <div className="home-section-inner">
            <SectionHeader
              eyebrow="Kiến trúc"
              title="Hai pipeline nghiên cứu"
              lead="Thiết kế đối xứng để đo lường trade-off giữa độ chính xác, tốc độ suy luận và độ phức tạp triển khai."
            />
            <div className="pipeline-detail-grid">
              {PIPELINE_DETAILS.map((pipe) => (
                <article
                  key={pipe.id}
                  className={`pipeline-detail card pipeline-detail--${pipe.accent}`}
                >
                  <div className="pipeline-detail__head">
                    <span className="pipeline-detail__tag">{pipe.tag}</span>
                    <h3 className="pipeline-detail__name">{pipe.name}</h3>
                  </div>
                  <p className="pipeline-detail__summary">{pipe.summary}</p>
                  <div className="pipeline-detail__block">
                    <p className="pipeline-detail__label">Mô hình đánh giá</p>
                    <div className="chip-row">
                      {pipe.models.map((model) => (
                        <span key={model} className="chip">
                          {model}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="pipeline-detail__block">
                    <p className="pipeline-detail__label">Điểm mạnh</p>
                    <ul className="pipeline-detail__list">
                      {pipe.strengths.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="pipeline-detail__block">
                    <p className="pipeline-detail__label">Metric chính</p>
                    <div className="chip-row">
                      {pipe.metrics.map((metric) => (
                        <span key={metric} className="chip chip--muted">
                          {metric}
                        </span>
                      ))}
                    </div>
                  </div>
                </article>
              ))}
            </div>

            <div className="compare-table-wrap card">
              <h3 className="compare-table__title">Bảng so sánh nhanh</h3>
              <table className="compare-table">
                <thead>
                  <tr>
                    <th scope="col">Tiêu chí</th>
                    <th scope="col">Pipeline 1</th>
                    <th scope="col">Pipeline 2</th>
                  </tr>
                </thead>
                <tbody>
                  {PIPELINE_COMPARE_ROWS.map((row) => (
                    <tr key={row.label}>
                      <th scope="row">{row.label}</th>
                      <td>{row.flow1}</td>
                      <td>{row.flow2}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Benchmark */}
        <section className="home-section" aria-labelledby="benchmark-title">
          <div className="home-section-inner">
            <SectionHeader
              eyebrow="Nghiên cứu"
              title="Chiến lược benchmark"
              lead="Quy trình đánh giá ba giai đoạn đảm bảo kết luận dựa trên thí nghiệm có kiểm soát, không chỉ demo trực quan."
            />
            <div className="benchmark-grid">
              {BENCHMARK_STAGES.map((stage) => (
                <article key={stage.stage} className="benchmark-card card">
                  <p className="benchmark-card__stage">{stage.stage}</p>
                  <h3 className="benchmark-card__title">{stage.title}</h3>
                  <p className="benchmark-card__desc">{stage.desc}</p>
                  <p className="benchmark-card__output">
                    <span>Đầu ra:</span> {stage.output}
                  </p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* Use cases */}
        <section className="home-section home-section--muted" aria-labelledby="usecases-title">
          <div className="home-section-inner">
            <SectionHeader
              eyebrow="Ứng dụng"
              title="Lĩnh vực triển khai"
              lead="Hệ thống hướng tới các bài toán nhận thức biển báo trong môi trường giao thông thực tế."
            />
            <div className="usecase-grid">
              {USE_CASES.map((item) => (
                <article key={item.title} className="usecase-card card">
                  <h3 className="usecase-card__title">{item.title}</h3>
                  <p className="usecase-card__desc">{item.desc}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* Tech stack */}
        <section className="home-section" aria-labelledby="tech-title">
          <div className="home-section-inner">
            <SectionHeader
              eyebrow="Công nghệ"
              title="Stack kỹ thuật"
              lead="Xây dựng trên hệ sinh thái Python deep learning phổ biến, dễ mở rộng và tái lập thí nghiệm."
            />
            <div className="tech-grid">
              {TECH_STACK.map((item) => (
                <div key={item.name} className="tech-card card">
                  <p className="tech-card__name">{item.name}</p>
                  <p className="tech-card__category">{item.category}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="cta-banner" aria-labelledby="cta-title">
          <div className="cta-banner-inner card">
            <div className="cta-banner__copy">
              <h2 id="cta-title" className="cta-banner__title">
                Sẵn sàng thử nghiệm trên dữ liệu của bạn?
              </h2>
              <p className="cta-banner__desc">
                Truy cập trang Tính năng để tải ảnh hoặc video, chạy song song hai
                pipeline và xem kết quả trực quan cùng metric chi tiết.
              </p>
            </div>
            <Link to="/tinh-nang" className="btn btn--primary cta-banner__btn">
              Mở demo nhận dạng
            </Link>
          </div>
        </section>

        {/* FAQ */}
        <section
          id="faq"
          className="home-section home-section--muted"
          aria-labelledby="faq-title"
        >
          <div className="home-section-inner">
            <SectionHeader
              eyebrow="Hỗ trợ"
              title="Câu hỏi thường gặp"
              lead="Giải đáp nhanh về cách sử dụng, kiến trúc hệ thống và dữ liệu nghiên cứu."
            />
            <FaqAccordion items={FAQ_ITEMS} />
          </div>
        </section>

        {/* Auth */}
        <section className="home-section" aria-labelledby="auth-title">
          <div className="home-section-inner">
            <SectionHeader
              eyebrow="Tài khoản"
              title="Đăng nhập & Đăng ký"
              lead="Tạo tài khoản để lưu lịch sử nhận dạng, quản lý phiên làm việc và truy cập báo cáo cá nhân. Giao diện đã sẵn sàng — backend xác thực sẽ được tích hợp trong giai đoạn tiếp theo."
            />
            <div className="auth-grid">
              <AuthForm
                id="dang-nhap"
                title="Đăng nhập"
                subtitle="Truy cập tài khoản đã đăng ký để sử dụng đầy đủ tính năng."
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
                subtitle="Tạo tài khoản mới để theo dõi và lưu kết quả nhận dạng."
                submitLabel="Tạo tài khoản"
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
