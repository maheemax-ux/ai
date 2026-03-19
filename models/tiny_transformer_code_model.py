import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is not installed. Run `pip3 install -r requirements.txt` and try again."
    ) from exc


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CODE_TRAINING_PATH = DATA_DIR / "code_training.txt"
CODE_CHECKPOINT_PATH = BASE_DIR / "tiny_transformer_code_model.pt"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_code_corpus() -> str:
    corpus = read_text(CODE_TRAINING_PATH)
    if not corpus:
        raise SystemExit(
            "No code training data found. Add examples to data/code_training.txt first."
        )
    return corpus


def build_landing_page_prompt(theme: str) -> str:
    safe_theme = theme.strip() or "creative studio"
    art_direction = random.choice(
        [
            "editorial layout with oversized typography",
            "playful startup page with bright accents",
            "premium studio page with soft gradients",
            "clean product launch page with card sections",
        ]
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "  <head>\n"
        '    <meta charset="UTF-8" />\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        f"    <title>{safe_theme.title()} Landing Page</title>\n"
        "    <style>\n"
        "      :root {\n"
        "        --bg: #f4efe8;\n"
        "        --text: #1f2933;\n"
        "        --accent: #e76f51;\n"
        "      }\n"
        "\n"
        "      body {\n"
        "        margin: 0;\n"
        '        font-family: "Trebuchet MS", sans-serif;\n'
        "        background: var(--bg);\n"
        "        color: var(--text);\n"
        "      }\n"
        "    </style>\n"
        "  </head>\n"
        "  <body>\n"
        '    <section class="hero">\n'
        '      <div class="hero-card">\n'
        f"        <p>{safe_theme.title()}</p>\n"
        f"        <h1>{art_direction.title()}</h1>\n"
    )


def normalize_theme_label(theme: str) -> str:
    return " ".join(theme.strip().split()) or "creative studio"


def theme_headline_variants(theme: str) -> list[str]:
    safe_theme = normalize_theme_label(theme)
    lower_theme = safe_theme.lower()
    variants = [
        f"Build a bold {safe_theme} landing page.",
        f"Turn {safe_theme} into a homepage people remember.",
        f"Launch {safe_theme} with a page that feels alive.",
    ]
    if "brand" in lower_theme:
        variants.append(f"Give your {safe_theme} a striking first impression.")
    else:
        variants.append(f"Give your {safe_theme} brand a striking first impression.")
    return variants


def extract_html_document(text: str) -> str:
    start = text.find("<!doctype html>")
    if start == -1:
        start = text.find("<html")
    if start == -1:
        return text.strip()

    end = text.rfind("</html>")
    if end != -1:
        return text[start:end + len("</html>")].strip()
    return text[start:].strip()


def ensure_landing_page_shape(text: str, theme: str) -> str:
    document = extract_html_document(text)
    if "</html>" in document and "</body>" in document and "</style>" in document:
        return document

    safe_theme = normalize_theme_label(theme)
    palette = random.choice(
        [
            {
                "bg1": "#f4efe8",
                "bg2": "#dce7f3",
                "panel": "#fffaf3",
                "text": "#1f2933",
                "accent": "#e76f51",
                "accent_dark": "#b84a2d",
                "font": '"Trebuchet MS", sans-serif',
            },
            {
                "bg1": "#fdf6ec",
                "bg2": "#f7d9c4",
                "panel": "#fffdf8",
                "text": "#283044",
                "accent": "#f3722c",
                "accent_dark": "#bc4b1d",
                "font": 'Georgia, serif',
            },
            {
                "bg1": "#eef6ff",
                "bg2": "#d6eadf",
                "panel": "#ffffff",
                "text": "#14213d",
                "accent": "#2a9d8f",
                "accent_dark": "#1f6f66",
                "font": '"Avenir Next", sans-serif',
            },
        ]
    )
    headline_options = theme_headline_variants(safe_theme)
    body_options = [
        "This generated page leans into strong hierarchy, warm color, and a clear path to action.",
        "This version pushes a more expressive layout so the page feels intentional instead of generic.",
        "This page aims for a clear message, visual energy, and a fast path from headline to CTA.",
    ]
    feature_sets = [
        [
            ("Clear message", "Lead with one strong promise."),
            ("Responsive layout", "Works on desktop and mobile."),
            ("Warm visual style", "Uses expressive color and spacing."),
        ],
        [
            ("Fast launch", "Go from idea to branded page quickly."),
            ("Bold typography", "Large headlines create instant focus."),
            ("Flexible sections", "Easy to adapt for campaigns or products."),
        ],
        [
            ("Story first", "Use copy and visuals to shape a memorable pitch."),
            ("Designed to convert", "Buttons and sections guide the next step."),
            ("Distinct look", "Avoids the flat, generic template feel."),
        ],
    ]
    cta_pairs = [
        ("Start now", "See demo"),
        ("Book a call", "Explore work"),
        ("Launch project", "View gallery"),
        ("Try concept", "Read story"),
    ]
    headline = random.choice(headline_options)
    body_copy = random.choice(body_options)
    features = random.choice(feature_sets)
    primary_cta, secondary_cta = random.choice(cta_pairs)
    section_class = random.choice(["hero", "hero hero-split", "hero hero-glow"])
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "  <head>\n"
        '    <meta charset="UTF-8" />\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        f"    <title>{safe_theme.title()} Landing Page</title>\n"
        "    <style>\n"
        "      :root {\n"
        f"        --bg: {palette['bg1']};\n"
        f"        --panel: {palette['panel']};\n"
        f"        --text: {palette['text']};\n"
        f"        --accent: {palette['accent']};\n"
        f"        --accent-dark: {palette['accent_dark']};\n"
        "      }\n"
        "      * {\n"
        "        box-sizing: border-box;\n"
        "      }\n"
        "      body {\n"
        "        margin: 0;\n"
        f"        font-family: {palette['font']};\n"
        "        color: var(--text);\n"
        f"        background: linear-gradient(135deg, {palette['bg1']}, {palette['bg2']});\n"
      "      }\n"
        "      .hero {\n"
        "        min-height: 100vh;\n"
        "        display: grid;\n"
        "        place-items: center;\n"
        "        padding: 48px 20px;\n"
        "      }\n"
        "      .hero-card {\n"
        "        width: min(820px, 100%);\n"
        "        background: var(--panel);\n"
        "        border-radius: 28px;\n"
        "        padding: 40px;\n"
        "        box-shadow: 0 24px 60px rgba(31, 41, 51, 0.14);\n"
        "      }\n"
        "      .eyebrow {\n"
        "        letter-spacing: 0.12em;\n"
        "        text-transform: uppercase;\n"
        "        color: var(--accent-dark);\n"
        "        font-size: 0.8rem;\n"
        "      }\n"
        "      h1 {\n"
        "        margin: 12px 0;\n"
        "        font-size: clamp(2.8rem, 7vw, 5rem);\n"
        "        line-height: 0.95;\n"
        "      }\n"
        "      p {\n"
        "        line-height: 1.7;\n"
        "      }\n"
        "      .actions {\n"
        "        display: flex;\n"
        "        gap: 12px;\n"
        "        flex-wrap: wrap;\n"
        "        margin-top: 24px;\n"
        "      }\n"
        "      .button {\n"
        "        border: none;\n"
        "        border-radius: 999px;\n"
        "        padding: 14px 20px;\n"
        "        cursor: pointer;\n"
        "      }\n"
        "      .button-primary {\n"
        "        background: var(--accent);\n"
        "        color: white;\n"
        "      }\n"
        "      .button-secondary {\n"
        "        background: transparent;\n"
        "        border: 2px solid var(--accent);\n"
        "        color: var(--accent-dark);\n"
        "      }\n"
        "      .features {\n"
        "        display: grid;\n"
        "        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));\n"
        "        gap: 16px;\n"
        "        margin-top: 32px;\n"
        "      }\n"
        "      .feature {\n"
        "        padding: 18px;\n"
        "        border-radius: 18px;\n"
        "        background: white;\n"
        "      }\n"
        "      .hero-split .hero-card {\n"
        "        display: grid;\n"
        "        gap: 18px;\n"
        "      }\n"
        "      .hero-glow .hero-card {\n"
        "        position: relative;\n"
        "        overflow: hidden;\n"
        "      }\n"
        "      .hero-glow .hero-card::after {\n"
        "        content: '';\n"
        "        position: absolute;\n"
        "        inset: auto -80px -80px auto;\n"
        "        width: 220px;\n"
        "        height: 220px;\n"
        "        border-radius: 999px;\n"
        "        background: color-mix(in srgb, var(--accent) 24%, transparent);\n"
        "        filter: blur(12px);\n"
        "      }\n"
        "      @media (max-width: 640px) {\n"
        "        .hero-card {\n"
        "          padding: 24px;\n"
        "        }\n"
        "        .actions {\n"
        "          flex-direction: column;\n"
        "        }\n"
        "      }\n"
        "    </style>\n"
        "  </head>\n"
        "  <body>\n"
        f'    <section class="{section_class}">\n'
        '      <div class="hero-card">\n'
        f'        <p class="eyebrow">{safe_theme.title()}</p>\n'
        f"        <h1>{headline}</h1>\n"
        f"        <p>{body_copy}</p>\n"
        '        <div class="actions">\n'
        f'          <button class="button button-primary">{primary_cta}</button>\n'
        f'          <button class="button button-secondary">{secondary_cta}</button>\n'
        "        </div>\n"
        '        <div class="features">\n'
        f'          <article class="feature"><h2>{features[0][0]}</h2><p>{features[0][1]}</p></article>\n'
        f'          <article class="feature"><h2>{features[1][0]}</h2><p>{features[1][1]}</p></article>\n'
        f'          <article class="feature"><h2>{features[2][0]}</h2><p>{features[2][1]}</p></article>\n'
        "        </div>\n"
        "      </div>\n"
        "    </section>\n"
        "  </body>\n"
        "</html>\n"
    )


@dataclass
class CharDataset:
    text: str

    def __post_init__(self) -> None:
        chars = sorted(set(self.text))
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}
        self.encoded = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode_string(self, value: str) -> list[int]:
        return [self.stoi[ch] for ch in value if ch in self.stoi]

    def decode_tokens(self, tokens: list[int]) -> str:
        return "".join(self.itos[idx] for idx in tokens)

    def sample_batch(self, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.encoded) <= block_size:
            raise SystemExit("Code dataset is too small for the chosen block size.")
        starts = torch.randint(0, len(self.encoded) - block_size - 1, (batch_size,))
        x = torch.stack([self.encoded[start:start + block_size] for start in starts]).to(device)
        y = torch.stack([self.encoded[start + 1:start + block_size + 1] for start in starts]).to(device)
        return x, y


class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [SelfAttentionHead(embed_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, embed_dim, block_size, dropout)
        self.ffwd = FeedForward(embed_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyTransformerCodeModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, time_steps = idx.shape
        positions = torch.arange(time_steps, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(positions)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch_size * time_steps, -1), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        seed_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        dataset: CharDataset,
        device: torch.device,
    ) -> str:
        self.eval()
        if not seed_tokens:
            seed_tokens = [random.randrange(dataset.vocab_size)]

        idx = torch.tensor([seed_tokens], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-3)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return dataset.decode_tokens(idx[0].tolist())


def save_checkpoint(model: TinyTransformerCodeModel, dataset: CharDataset, config: dict[str, int | float]) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "chars": sorted(dataset.stoi.keys()),
        "config": config,
    }
    torch.save(payload, CODE_CHECKPOINT_PATH)


def load_checkpoint(device: torch.device) -> tuple[TinyTransformerCodeModel, CharDataset, dict[str, int | float]]:
    corpus = build_code_corpus()
    dataset = CharDataset(corpus)
    payload = torch.load(CODE_CHECKPOINT_PATH, map_location=device)
    checkpoint_chars = payload["chars"]
    current_chars = sorted(dataset.stoi.keys())
    if checkpoint_chars != current_chars:
        raise SystemExit(
            "Code training characters changed since the checkpoint was created. Retrain the transformer first."
        )
    config = payload["config"]
    model = TinyTransformerCodeModel(
        vocab_size=dataset.vocab_size,
        block_size=int(config["block_size"]),
        embed_dim=int(config["embed_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, dataset, config


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = CharDataset(build_code_corpus())
    model = TinyTransformerCodeModel(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Training transformer on {device} with vocab size {dataset.vocab_size}")
    for step in range(1, args.steps + 1):
        xb, yb = dataset.sample_batch(args.batch_size, args.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            perplexity = math.exp(loss.item())
            print(f"step {step:4d} | loss {loss.item():.4f} | ppl {perplexity:.2f}")

    save_checkpoint(
        model,
        dataset,
        {
            "block_size": args.block_size,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
    )
    print(f"Saved transformer checkpoint to {CODE_CHECKPOINT_PATH}")


def generate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if not CODE_CHECKPOINT_PATH.exists():
        raise SystemExit("No transformer checkpoint found. Run `python3 tiny_transformer_code_model.py train` first.")

    model, dataset, _ = load_checkpoint(device)
    prompt = args.prompt
    seed_tokens = dataset.encode_string(prompt)
    output = model.generate(
        seed_tokens=seed_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dataset=dataset,
        device=device,
    )
    print(output)


def landing_page(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if not CODE_CHECKPOINT_PATH.exists():
        raise SystemExit("No transformer checkpoint found. Run `python3 tiny_transformer_code_model.py train` first.")

    model, dataset, _ = load_checkpoint(device)
    prompt = build_landing_page_prompt(args.theme)
    seed_tokens = dataset.encode_string(prompt)
    output = model.generate(
        seed_tokens=seed_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dataset=dataset,
        device=device,
    )
    print(ensure_landing_page_shape(output, args.theme))


def complete(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if not CODE_CHECKPOINT_PATH.exists():
        raise SystemExit("No transformer checkpoint found. Run `python3 tiny_transformer_code_model.py train` first.")

    model, dataset, _ = load_checkpoint(device)

    print("Tiny Transformer Code Model")
    print("Type 'exit' to quit.\n")
    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == "exit":
            print("Model: See you soon.")
            break
        seed_tokens = dataset.encode_string(prompt)
        output = model.generate(
            seed_tokens=seed_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            dataset=dataset,
            device=device,
        )
        print("Model:")
        print(output)
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and sample a tiny transformer-based code model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the transformer code model.")
    train_parser.add_argument("--steps", type=int, default=600)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--block-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--embed-dim", type=int, default=128)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    generate_parser = subparsers.add_parser("generate", help="Generate code from a prompt.")
    generate_parser.add_argument("--prompt", type=str, default="def ")
    generate_parser.add_argument("--max-new-tokens", type=int, default=220)
    generate_parser.add_argument("--temperature", type=float, default=0.5)
    generate_parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    landing_parser = subparsers.add_parser("landing-page", help="Generate a complete landing page document.")
    landing_parser.add_argument("--theme", type=str, default="creative studio")
    landing_parser.add_argument("--max-new-tokens", type=int, default=900)
    landing_parser.add_argument("--temperature", type=float, default=0.35)
    landing_parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    complete_parser = subparsers.add_parser("complete", help="Interactive code completion mode.")
    complete_parser.add_argument("--max-new-tokens", type=int, default=220)
    complete_parser.add_argument("--temperature", type=float, default=0.5)
    complete_parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "landing-page":
        landing_page(args)
    elif args.command == "complete":
        complete(args)


if __name__ == "__main__":
    main()
