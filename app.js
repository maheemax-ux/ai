const paletteLibrary = {
  bold: [
    {
      bg1: "#f4efe8",
      bg2: "#dce8f2",
      panel: "#fffaf3",
      text: "#18222d",
      accent: "#db5f36",
      accentDark: "#9e3d1d",
      font: '"Avenir Next", "Segoe UI", sans-serif',
    },
    {
      bg1: "#eef6ff",
      bg2: "#d8f0e4",
      panel: "#ffffff",
      text: "#10223b",
      accent: "#0f9d8a",
      accentDark: "#0a675c",
      font: '"Trebuchet MS", sans-serif',
    },
  ],
  premium: [
    {
      bg1: "#f7f1ea",
      bg2: "#e7d8c8",
      panel: "#fffdf8",
      text: "#2b2118",
      accent: "#b77938",
      accentDark: "#7a4d1f",
      font: 'Georgia, serif',
    },
    {
      bg1: "#f5f3ff",
      bg2: "#ddd6fe",
      panel: "#ffffff",
      text: "#231942",
      accent: "#be95c4",
      accentDark: "#7b2c82",
      font: '"Palatino Linotype", serif',
    },
  ],
  playful: [
    {
      bg1: "#fff5d6",
      bg2: "#ffd7ba",
      panel: "#fffdf7",
      text: "#253047",
      accent: "#ff7b54",
      accentDark: "#bc4b1d",
      font: '"Trebuchet MS", sans-serif',
    },
    {
      bg1: "#f3f0ff",
      bg2: "#cde7ff",
      panel: "#ffffff",
      text: "#1d3557",
      accent: "#ff4d6d",
      accentDark: "#b5173f",
      font: '"Gill Sans", sans-serif',
    },
  ],
  calm: [
    {
      bg1: "#edf6f9",
      bg2: "#d9eddf",
      panel: "#ffffff",
      text: "#1f3b4d",
      accent: "#4ea8de",
      accentDark: "#33658a",
      font: '"Avenir Next", sans-serif',
    },
    {
      bg1: "#f1f5f9",
      bg2: "#e2e8f0",
      panel: "#ffffff",
      text: "#243b53",
      accent: "#5c7cfa",
      accentDark: "#364fc7",
      font: '"Segoe UI", sans-serif',
    },
  ],
};

const headlineTemplates = [
  "Turn {theme} into the page people remember.",
  "Launch {theme} with a website that feels alive.",
  "Make your {theme} look sharp from the first scroll.",
  "Give {audience} a reason to trust your {theme} fast.",
];

const bodyTemplates = [
  "This concept blends strong hierarchy, expressive visuals, and a clear path to action for {audience}.",
  "Use this landing page to frame {theme} with confidence, warmth, and a sharper first impression.",
  "This version is built to feel intentional, modern, and easy for {audience} to understand in seconds.",
];

const featurePool = [
  ["Fast launch", "Go from idea to branded page quickly."],
  ["Clear story", "Give the visitor one memorable promise."],
  ["Responsive layout", "Looks good on both mobile and desktop."],
  ["Bold visuals", "Use color and spacing to create energy."],
  ["Conversion focus", "Lead visitors toward a strong next step."],
  ["Flexible sections", "Adapt the page for products, services, or campaigns."],
];

const testimonialPool = [
  "\"The page finally felt like our brand, not a template.\"",
  "\"We shipped a sharper landing page in one weekend.\"",
  "\"The new design made our message easier to trust immediately.\"",
];

const faqPool = [
  ["Can I edit this layout?", "Yes. The generated HTML is plain and easy to customize."],
  ["Will it work on GitHub Pages?", "Yes. This app creates static HTML, CSS, and JavaScript."],
  ["Can I use my own branding?", "Yes. Swap the colors, copy, and sections however you want."],
];

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function titleCase(value) {
  return value
    .trim()
    .split(/\s+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function pickSections(form) {
  return [...form.querySelectorAll('input[name="sections"]:checked')].map((node) => node.value);
}

function pickPalette(tone) {
  return randomItem(paletteLibrary[tone] || paletteLibrary.bold);
}

function pickLayout(layout) {
  if (layout !== "auto") {
    return layout;
  }
  return randomItem(["split", "glow", "editorial"]);
}

function fillTemplate(template, values) {
  return template
    .replaceAll("{theme}", values.theme)
    .replaceAll("{audience}", values.audience);
}

function buildFeatureCards() {
  const shuffled = [...featurePool].sort(() => Math.random() - 0.5).slice(0, 3);
  return shuffled
    .map(
      ([title, text]) =>
        `<article class="feature-card"><h3>${title}</h3><p>${text}</p></article>`
    )
    .join("");
}

function buildStats() {
  const stats = [
    ["42%", "Average lift after clearer hero messaging."],
    ["7 days", "Typical sprint to launch a focused page."],
    ["100%", "Custom styling without boilerplate vibes."],
  ];
  return stats
    .map(
      ([value, text]) =>
        `<article class="stat-card"><strong>${value}</strong><p>${text}</p></article>`
    )
    .join("");
}

function buildFaq() {
  return faqPool
    .map(
      ([question, answer]) =>
        `<article class="faq-item"><h3>${question}</h3><p>${answer}</p></article>`
    )
    .join("");
}

function buildLandingPage(options) {
  const palette = pickPalette(options.tone);
  const layout = pickLayout(options.layout);
  const sections = options.sections;
  const title = `${titleCase(options.theme)} Landing Page`;
  const headline = fillTemplate(randomItem(headlineTemplates), options);
  const body = fillTemplate(randomItem(bodyTemplates), options);
  const primaryCta = randomItem(["Start now", "Book a call", "Launch project", "See the concept"]);
  const secondaryCta = randomItem(["View work", "Read story", "See demo", "Explore details"]);
  const testimonial = randomItem(testimonialPool);

  const layoutClass =
    layout === "split" ? "hero hero-split" : layout === "glow" ? "hero hero-glow" : "hero hero-editorial";

  const statsSection = sections.includes("stats")
    ? `<section class="stats-grid">${buildStats()}</section>`
    : "";
  const testimonialSection = sections.includes("testimonial")
    ? `<section class="quote-block"><p>${testimonial}</p><span>${titleCase(options.audience)}</span></section>`
    : "";
  const faqSection = sections.includes("faq")
    ? `<section class="faq-grid"><h2>Quick answers</h2>${buildFaq()}</section>`
    : "";
  const featureSection = sections.includes("features")
    ? `<section class="feature-grid">${buildFeatureCards()}</section>`
    : "";

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${title}</title>
    <style>
      :root {
        --bg1: ${palette.bg1};
        --bg2: ${palette.bg2};
        --panel: ${palette.panel};
        --text: ${palette.text};
        --accent: ${palette.accent};
        --accent-dark: ${palette.accentDark};
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: ${palette.font};
        color: var(--text);
        background:
          radial-gradient(circle at top left, color-mix(in srgb, var(--accent) 16%, transparent), transparent 30%),
          linear-gradient(135deg, var(--bg1), var(--bg2));
      }

      .page {
        width: min(1120px, calc(100% - 32px));
        margin: 0 auto;
        padding: 28px 0 80px;
      }

      .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        padding-bottom: 30px;
      }

      .topbar strong {
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.9rem;
      }

      .topbar nav {
        display: flex;
        gap: 18px;
        color: color-mix(in srgb, var(--text) 72%, white);
        flex-wrap: wrap;
      }

      .hero {
        display: grid;
        gap: 24px;
        align-items: stretch;
      }

      .hero-split {
        grid-template-columns: 1.2fr 0.85fr;
      }

      .hero-glow,
      .hero-editorial {
        grid-template-columns: 1fr;
      }

      .hero-card,
      .side-card,
      .stat-card,
      .feature-card,
      .quote-block,
      .faq-item {
        background: color-mix(in srgb, var(--panel) 88%, white);
        border-radius: 28px;
        box-shadow: 0 24px 60px rgba(24, 34, 45, 0.12);
      }

      .hero-card {
        position: relative;
        padding: 38px;
        overflow: hidden;
      }

      .hero-glow .hero-card::after {
        content: "";
        position: absolute;
        right: -70px;
        bottom: -70px;
        width: 240px;
        height: 240px;
        border-radius: 999px;
        background: color-mix(in srgb, var(--accent) 28%, transparent);
        filter: blur(16px);
      }

      .hero-card p,
      .feature-card p,
      .quote-block p,
      .faq-item p,
      .stat-card p {
        line-height: 1.7;
        color: color-mix(in srgb, var(--text) 72%, white);
      }

      .eyebrow {
        margin: 0 0 12px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-size: 0.78rem;
        color: var(--accent-dark);
      }

      h1 {
        margin: 0 0 18px;
        font-size: clamp(3rem, 7vw, 5.8rem);
        line-height: 0.92;
      }

      h2,
      h3 {
        margin-top: 0;
      }

      .action-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 28px;
      }

      .button {
        border: none;
        border-radius: 999px;
        padding: 14px 20px;
        font: inherit;
        cursor: pointer;
      }

      .button-primary {
        background: var(--accent);
        color: white;
      }

      .button-secondary {
        background: transparent;
        color: var(--accent-dark);
        border: 2px solid color-mix(in srgb, var(--accent) 50%, white);
      }

      .side-card {
        padding: 28px;
        display: grid;
        gap: 18px;
      }

      .feature-grid,
      .stats-grid,
      .faq-grid {
        display: grid;
        gap: 18px;
        margin-top: 24px;
      }

      .feature-grid {
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }

      .stats-grid {
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      }

      .feature-card,
      .stat-card,
      .faq-item {
        padding: 22px;
      }

      .stat-card strong {
        font-size: 2rem;
      }

      .quote-block {
        margin-top: 24px;
        padding: 24px 28px;
      }

      .quote-block span {
        display: inline-block;
        margin-top: 10px;
        color: var(--accent-dark);
        font-weight: 700;
      }

      @media (max-width: 860px) {
        .hero-split {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 640px) {
        .page {
          width: min(100% - 20px, 1120px);
        }

        .hero-card,
        .side-card {
          padding: 24px;
        }

        .action-row {
          flex-direction: column;
        }

        .button {
          width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <main class="page">
      <header class="topbar">
        <strong>${titleCase(options.theme)}</strong>
        <nav>
          <span>Work</span>
          <span>About</span>
          <span>Contact</span>
        </nav>
      </header>

      <section class="${layoutClass}">
        <article class="hero-card">
          <p class="eyebrow">${titleCase(options.audience)}</p>
          <h1>${headline}</h1>
          <p>${body}</p>
          <div class="action-row">
            <button class="button button-primary">${primaryCta}</button>
            <button class="button button-secondary">${secondaryCta}</button>
          </div>
          ${featureSection}
          ${testimonialSection}
        </article>

        <aside class="side-card">
          <p class="eyebrow">Why it works</p>
          <h2>Built for ${options.audience}</h2>
          <p>
            This concept creates a stronger first impression for your ${options.theme}
            offer with clearer hierarchy and a more intentional visual rhythm.
          </p>
          ${statsSection}
          ${faqSection}
        </aside>
      </section>
    </main>
  </body>
</html>`;
}

const form = document.getElementById("generator-form");
const previewFrame = document.getElementById("preview-frame");
const codeOutput = document.getElementById("code-output");
const statusText = document.getElementById("status-text");
const shuffleButton = document.getElementById("shuffle-button");
const copyButton = document.getElementById("copy-button");
const downloadButton = document.getElementById("download-button");

let currentHtml = "";

function updateStatus(message) {
  statusText.textContent = message;
}

function renderLandingPage() {
  const formData = new FormData(form);
  const options = {
    theme: (formData.get("theme") || "design agency").toString().trim() || "design agency",
    audience: (formData.get("audience") || "creative founders").toString().trim() || "creative founders",
    tone: (formData.get("tone") || "bold").toString(),
    layout: (formData.get("layout") || "auto").toString(),
    sections: pickSections(form),
  };

  currentHtml = buildLandingPage(options);
  previewFrame.srcdoc = currentHtml;
  codeOutput.value = currentHtml;
  updateStatus(`Generated ${titleCase(options.theme)}`);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  renderLandingPage();
});

shuffleButton.addEventListener("click", () => {
  renderLandingPage();
});

copyButton.addEventListener("click", async () => {
  if (!currentHtml) {
    renderLandingPage();
  }
  await navigator.clipboard.writeText(currentHtml);
  updateStatus("HTML copied");
});

downloadButton.addEventListener("click", () => {
  if (!currentHtml) {
    renderLandingPage();
  }

  const theme = document.getElementById("theme-input").value.trim() || "landing-page";
  const safeName = theme.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  const blob = new Blob([currentHtml], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${safeName}.html`;
  link.click();
  URL.revokeObjectURL(url);
  updateStatus("HTML downloaded");
});

renderLandingPage();
