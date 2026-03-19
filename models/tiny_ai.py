import random
import re
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROFILE_PATH = DATA_DIR / "personality_profile.txt"
TRAINING_PATH = DATA_DIR / "training_text.txt"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def split_sentences(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for part in raw_parts:
        cleaned = " ".join(part.strip().split())
        if cleaned:
            sentences.append(cleaned)
    return sentences


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


class TinyPersonalityAI:
    def __init__(self, profile_text: str, training_text: str) -> None:
        self.profile_text = profile_text
        self.training_text = training_text
        self.sentences = split_sentences(training_text)
        self.bigram_map = self._build_bigram_map(training_text)
        self.style_words = self._extract_style_words(profile_text, training_text)
        self.fallbacks = [
            "I want to answer that in a way that feels honest and calm.",
            "That matters to me, and I would probably respond with curiosity first.",
            "My first instinct is to keep it simple, thoughtful, and real.",
            "I would lean toward a warm answer instead of a cold one.",
        ]

    def _build_bigram_map(self, text: str) -> dict[str, list[str]]:
        words = tokenize(text)
        mapping: dict[str, list[str]] = {}
        for first, second in zip(words, words[1:]):
            mapping.setdefault(first, []).append(second)
        return mapping

    def _extract_style_words(self, profile_text: str, training_text: str) -> list[str]:
        profile_traits = []
        for line in profile_text.splitlines():
            raw = line.strip()
            if not raw.startswith("-"):
                continue
            cleaned = raw.lstrip("-").strip().lower()
            if cleaned and len(cleaned.split()) <= 3:
                profile_traits.append(cleaned)
        if profile_traits:
            return profile_traits[:12]

        stop_words = {
            "the", "and", "that", "with", "from", "this", "have", "your", "just",
            "into", "they", "them", "about", "would", "could", "there", "their",
            "what", "when", "where", "which", "while", "because", "been", "being",
            "like", "very", "really", "always", "maybe", "some", "more", "than",
            "feel", "want", "make", "keep", "calm", "kind", "honest", "people",
            "someone", "something", "things", "advice", "words", "helping", "thing",
            "step", "steps", "technology", "progress", "style", "sound",
        }
        counts: dict[str, int] = {}
        for word in tokenize(training_text):
            if len(word) < 4 or word in stop_words:
                continue
            counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [word for word, _ in ranked[:12]]

    def _pick_seed(self, user_message: str) -> str:
        user_words = [w for w in tokenize(user_message) if w in self.bigram_map]
        if user_words:
            return random.choice(user_words)
        if self.style_words:
            style_candidates = [w for w in self.style_words if w in self.bigram_map]
            if style_candidates:
                return random.choice(style_candidates)
        if self.bigram_map:
            return random.choice(list(self.bigram_map.keys()))
        return "hello"

    def _generate_phrase(self, seed: str, max_words: int = 18) -> str:
        words = [seed]
        current = seed
        for _ in range(max_words - 1):
            next_words = self.bigram_map.get(current)
            if not next_words:
                break
            current = random.choice(next_words)
            words.append(current)
        phrase = " ".join(words).strip()
        if not phrase:
            return ""
        if len(words) < 5:
            return ""
        return phrase[0].upper() + phrase[1:] + "."

    def _match_example_sentence(self, user_message: str) -> str | None:
        user_words = set(tokenize(user_message))
        best_sentence = None
        best_score = 0
        for sentence in self.sentences:
            score = len(user_words.intersection(tokenize(sentence)))
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence if best_score > 0 else None

    def reply(self, user_message: str) -> str:
        seed = self._pick_seed(user_message)
        generated = self._generate_phrase(seed)
        matched = self._match_example_sentence(user_message)

        pieces = []
        if matched:
            pieces.append(matched)
        if not matched and generated and generated.lower() not in {part.lower() for part in pieces}:
            pieces.append(f"I would put it like this: {generated}")
        if not pieces:
            pieces.append(random.choice(self.fallbacks))

        if self.style_words:
            style_hint = ", ".join(self.style_words[:3])
            pieces.append(f"I usually come across as {style_hint}.")

        return " ".join(pieces)


def main() -> None:
    profile_text = read_text(PROFILE_PATH)
    training_text = read_text(TRAINING_PATH)

    if not training_text:
        raise SystemExit(
            "No training text found. Add some writing samples to data/training_text.txt and try again."
        )

    bot = TinyPersonalityAI(profile_text, training_text)

    print("Tiny Personality AI")
    print("Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() == "exit":
            print("AI: See you soon.")
            break
        print("AI:", bot.reply(user_message))


if __name__ == "__main__":
    main()
