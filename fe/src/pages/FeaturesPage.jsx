import { useState } from "react";
import Navbar from "../components/Navbar.jsx";
import Footer from "../components/Footer.jsx";
import ImageTab from "../components/ImageTab.jsx";
import VideoTab from "../components/VideoTab.jsx";
import { TABS } from "../config/pipelines.js";

export default function FeaturesPage() {
  const [activeTab, setActiveTab] = useState("image");
  const currentTab = TABS.find((tab) => tab.id === activeTab);

  return (
    <div className="app">
      <Navbar />
      <header className="page-header">
        <div className="page-header-inner">
          <p className="header-eyebrow">Demo nhận dạng</p>
          <h1>Tính năng nhận dạng biển báo</h1>
          <p className="header-subtitle">
            Tải ảnh hoặc video để so sánh kết quả giữa pipeline YOLO end-to-end
            và pipeline lai YOLO&nbsp;+&nbsp;CNN trên cùng một đầu vào.
          </p>
        </div>
      </header>
      <div className="page">
        <nav className="tabs" aria-label="Chế độ nhận dạng">
          <div className="tabs-row">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={`tab${activeTab === tab.id ? " tab--active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
                aria-current={activeTab === tab.id ? "page" : undefined}
              >
                {tab.label}
              </button>
            ))}
          </div>
          {currentTab && (
            <p className="tab-description">{currentTab.description}</p>
          )}
        </nav>
        {activeTab === "image" ? <ImageTab /> : <VideoTab />}
      </div>
      <Footer />
    </div>
  );
}
