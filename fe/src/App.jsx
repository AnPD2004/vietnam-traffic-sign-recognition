import { useState } from "react";
import Header from "./components/Header.jsx";
import Footer from "./components/Footer.jsx";
import ImageTab from "./components/ImageTab.jsx";
import VideoTab from "./components/VideoTab.jsx";

const TABS = [
  { id: "image", label: "Ảnh" },
  { id: "video", label: "Video" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("image");

  return (
    <>
      <Header />
      <div className="page">
        <nav className="tabs" aria-label="Chế độ nhận dạng">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`tab${activeTab === tab.id ? " tab--active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        {activeTab === "image" ? <ImageTab /> : <VideoTab />}
      </div>
      <Footer />
    </>
  );
}
