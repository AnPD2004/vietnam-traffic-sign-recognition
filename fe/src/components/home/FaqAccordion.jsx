import { useState } from "react";

export default function FaqAccordion({ items }) {
  const [openId, setOpenId] = useState(items[0]?.q ?? null);

  return (
    <div className="faq-list">
      {items.map((item) => {
        const isOpen = openId === item.q;
        return (
          <div key={item.q} className={`faq-item card${isOpen ? " faq-item--open" : ""}`}>
            <button
              type="button"
              className="faq-item__trigger"
              aria-expanded={isOpen}
              onClick={() => setOpenId(isOpen ? null : item.q)}
            >
              <span className="faq-item__question">{item.q}</span>
              <span className="faq-item__icon" aria-hidden="true">
                {isOpen ? "−" : "+"}
              </span>
            </button>
            {isOpen && (
              <div className="faq-item__answer">
                <p>{item.a}</p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
