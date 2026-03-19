import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - only hit when torch is missing
    raise SystemExit(
        "PyTorch is not installed. Run `pip3 install torch` and try again."
    ) from exc


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROFILE_PATH = DATA_DIR / "personality_profile.txt"
TRAINING_PATH = DATA_DIR / "training_text.txt"
CHECKPOINT_PATH = BASE_DIR / "tiny_personality_model.pt"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_corpus() -> str:
    profile_text = read_text(PROFILE_PATH)
    training_text = read_text(TRAINING_PATH)
    combined = (profile_text + "\n\n" + training_text).strip()
    if not combined:
        raise SystemExit(
            "No training data found. Fill data/personality_profile.txt and data/training_text.txt first."
        )
    return combined


@dataclass
class TextDataset:
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
            raise SystemExit("Training text is too small. Add more lines to data/training_text.txt.")
        starts = torch.randint(0, len(self.encoded) - block_size - 1, (batch_size,))
        x = torch.stack([self.encoded[start:start + block_size] for start in starts]).to(device)
        y = torch.stack([self.encoded[start + 1:start + block_size + 1] for start in starts]).to(device)
        return x, y


class TinyCharModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 48, hidden_dim: int = 96, num_layers: int = 1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        embedded = self.embedding(idx)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.head(output)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, hidden

    @torch.no_grad()
    def generate(
        self,
        seed_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        dataset: TextDataset,
        device: torch.device,
    ) -> str:
        self.eval()
        if not seed_tokens:
            seed_tokens = [random.randrange(dataset.vocab_size)]

        idx = torch.tensor([seed_tokens], dtype=torch.long, device=device)
        hidden = None

        # Prime the recurrent state with the full prompt so generation continues from it.
        if idx.size(1) > 1:
            _, _, hidden = self(idx[:, :-1], hidden=hidden)

        for _ in range(max_new_tokens):
            logits, _, hidden = self(idx[:, -1:], hidden=hidden)
            next_logits = logits[:, -1, :] / max(temperature, 1e-3)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return dataset.decode_tokens(idx[0].tolist())


def save_checkpoint(model: TinyCharModel, dataset: TextDataset) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "chars": sorted(dataset.stoi.keys()),
    }
    torch.save(payload, CHECKPOINT_PATH)


def load_checkpoint(device: torch.device) -> tuple[TinyCharModel, TextDataset]:
    corpus = build_corpus()
    dataset = TextDataset(corpus)
    payload = torch.load(CHECKPOINT_PATH, map_location=device)
    checkpoint_chars = payload["chars"]
    current_chars = sorted(dataset.stoi.keys())
    if checkpoint_chars != current_chars:
        raise SystemExit(
            "Training data characters changed since the checkpoint was created. Retrain the model first."
        )
    model = TinyCharModel(vocab_size=dataset.vocab_size)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, dataset


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = TextDataset(build_corpus())
    model = TinyCharModel(vocab_size=dataset.vocab_size)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Training on {device} with vocab size {dataset.vocab_size}")
    for step in range(1, args.steps + 1):
        xb, yb = dataset.sample_batch(args.batch_size, args.block_size, device)
        _, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            perplexity = math.exp(loss.item())
            print(f"step {step:4d} | loss {loss.item():.4f} | ppl {perplexity:.2f}")

    save_checkpoint(model, dataset)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


def chat(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if not CHECKPOINT_PATH.exists():
        raise SystemExit("No checkpoint found. Run `python3 tiny_pytorch_ai.py train` first.")

    model, dataset = load_checkpoint(device)

    print("Tiny PyTorch Personality AI")
    print("Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() == "exit":
            print("AI: See you soon.")
            break

        prompt = f"User: {user_message}\nAI:"
        seed_tokens = dataset.encode_string(prompt)
        output = model.generate(
            seed_tokens=seed_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            dataset=dataset,
            device=device,
        )
        reply = output[len(prompt):].strip()
        if not reply:
            reply = "I am still learning your style. Add more training lines and retrain me."
        first_line = reply.splitlines()[0].strip()
        if first_line and first_line[-1].isalnum():
            first_line += "."
        print("AI:", first_line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and chat with a tiny personality model in PyTorch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the tiny character model.")
    train_parser.add_argument("--steps", type=int, default=400)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--block-size", type=int, default=48)
    train_parser.add_argument("--learning-rate", type=float, default=3e-3)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    chat_parser = subparsers.add_parser("chat", help="Chat with the trained model.")
    chat_parser.add_argument("--max-new-tokens", type=int, default=120)
    chat_parser.add_argument("--temperature", type=float, default=0.5)
    chat_parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "chat":
        chat(args)


if __name__ == "__main__":
    main()
