const iconProps = {
  width: 22,
  height: 22,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

export function IconImage() {
  return (
    <svg {...iconProps}>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <circle cx="8.5" cy="10" r="1.5" />
      <path d="M21 16l-5.5-5.5L6 19" />
    </svg>
  );
}

export function IconVideo() {
  return (
    <svg {...iconProps}>
      <rect x="2" y="6" width="14" height="12" rx="2" />
      <path d="M16 10l6-3v10l-6-3z" />
    </svg>
  );
}

export function IconCompare() {
  return (
    <svg {...iconProps}>
      <path d="M8 4H6a2 2 0 00-2 2v12a2 2 0 002 2h2" />
      <path d="M16 4h2a2 2 0 012 2v12a2 2 0 01-2 2h-2" />
      <path d="M12 8v8M9 11l3-3 3 3M9 13l3 3 3-3" />
    </svg>
  );
}

export function IconChart() {
  return (
    <svg {...iconProps}>
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="M8 15v-4M12 15V9M16 15V7" />
    </svg>
  );
}

export function IconDataset() {
  return (
    <svg {...iconProps}>
      <ellipse cx="12" cy="6" rx="8" ry="3" />
      <path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6" />
      <path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
    </svg>
  );
}

export function IconApi() {
  return (
    <svg {...iconProps}>
      <path d="M4 7h6M4 12h10M4 17h14" />
      <circle cx="19" cy="7" r="2" />
      <circle cx="17" cy="17" r="2" />
    </svg>
  );
}

export const FEATURE_ICONS = {
  image: IconImage,
  video: IconVideo,
  compare: IconCompare,
  metrics: IconChart,
  dataset: IconDataset,
  api: IconApi,
};
