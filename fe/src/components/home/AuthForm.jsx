import { useState } from "react";

export default function AuthForm({ id, title, subtitle, fields, submitLabel, onSubmit }) {
  const [notice, setNotice] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    setNotice("Tính năng xác thực đang được phát triển. Vui lòng quay lại sau.");
    onSubmit?.();
  }

  return (
    <section id={id} className="card auth-card" aria-labelledby={`${id}-title`}>
      <div className="auth-card__header">
        <h3 id={`${id}-title`} className="section-title">
          {title}
        </h3>
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
