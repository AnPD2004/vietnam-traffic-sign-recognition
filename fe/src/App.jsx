import { useState } from "react";
import Header from "./components/Header.jsx";
import Footer from "./components/Footer.jsx";
import ImageTab from "./components/ImageTab.jsx";
import VideoTab from "./components/VideoTab.jsx";
import { TABS } from "./config/pipelines.js";

export default function App() {
  const [activeTab, setActiveTab] = useState("image");
  const currentTab = TABS.find((tab) => tab.id === activeTab);

  return (
    <div className="app">
      <Header />
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
