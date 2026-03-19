import argparse
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from html.parser import HTMLParser

try:
    import torch
    from tiny_pytorch_ai import CHECKPOINT_PATH, load_checkpoint
except Exception:  # pragma: no cover - optional fallback when torch/model is unavailable
    torch = None
    CHECKPOINT_PATH = None
    load_checkpoint = None


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROFILE_PATH = DATA_DIR / "personality_profile.txt"
TRAINING_PATH = DATA_DIR / "training_text.txt"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def extract_profile_traits(profile_text: str) -> list[str]:
    traits = []
    for line in profile_text.splitlines():
        raw = line.strip()
        if not raw.startswith("-"):
            continue
        cleaned = raw.lstrip("-").strip()
        if cleaned:
            traits.append(cleaned)
    return traits


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [" ".join(part.split()) for part in parts if part.strip()]


def pick_style_line(training_text: str, query: str) -> str | None:
    query_words = set(re.findall(r"[a-zA-Z']+", query.lower()))
    best_line = None
    best_score = 0
    for sentence in split_sentences(training_text):
        words = set(re.findall(r"[a-zA-Z']+", sentence.lower()))
        score = len(query_words.intersection(words))
        if score > best_score:
            best_score = score
            best_line = sentence
    return best_line


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TinyPersonalityAI/1.0 (educational project)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TinyPersonalityAI/1.0 (educational project)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode("utf-8", errors="replace")


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_result = False
        self._capture_title = False
        self._capture_snippet = False
        self._current_title = []
        self._current_snippet = []
        self._current_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "") or ""

        if tag == "a" and "result__a" in class_name:
            self._in_result = True
            self._capture_title = True
            self._current_title = []
            raw_url = attrs_dict.get("href", "") or ""
            self._current_url = decode_duckduckgo_redirect(raw_url)
        elif tag == "a" and "result-link" in class_name:
            self._in_result = True
            self._capture_title = True
            self._current_title = []
            raw_url = attrs_dict.get("href", "") or ""
            self._current_url = decode_duckduckgo_redirect(raw_url)
        elif self._in_result and tag == "a" and not self._current_url:
            raw_url = attrs_dict.get("href", "") or ""
            if raw_url:
                self._current_url = decode_duckduckgo_redirect(raw_url)

        if "result__snippet" in class_name or "result-snippet" in class_name:
            self._capture_snippet = True
            self._current_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
        if tag == "div" and self._capture_snippet:
            self._capture_snippet = False
            self._flush_result()

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title.append(data)
        if self._capture_snippet:
            self._current_snippet.append(data)

    def close(self) -> None:
        super().close()
        self._flush_result()

    def _flush_result(self) -> None:
        title = " ".join("".join(self._current_title).split()).strip()
        snippet = " ".join("".join(self._current_snippet).split()).strip()
        if self._in_result and title and self._current_url:
            result = {
                "title": html.unescape(title),
                "snippet": html.unescape(snippet),
                "url": self._current_url,
                "source": infer_source_name(self._current_url),
            }
            if result not in self.results:
                self.results.append(result)
        self._in_result = False
        self._capture_title = False
        self._capture_snippet = False
        self._current_title = []
        self._current_snippet = []
        self._current_url = ""


def decode_duckduckgo_redirect(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    parsed = urllib.parse.urlparse(url)
    if url.startswith("/l/?") or (parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/"):
        query_params = urllib.parse.parse_qs(parsed.query)
        target = query_params.get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)
    return url


def infer_source_name(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "web"


def clean_generated_text(text: str) -> str:
    cleaned = " ".join(text.replace("\n", " ").split()).strip(" :;-")
    cleaned = re.sub(r"\b(ai|user|facts|question|sources|summary)\b\s*:?", "", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split()).strip(" ,;:-")
    if not cleaned:
        return ""
    match = re.search(r"(.{12,160}?[.!?])(?:\s|$)", cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = cleaned[:160].strip()
        if cleaned and cleaned[-1].isalnum():
            cleaned += "."
    return cleaned


class NeuralPersonalityVoice:
    def __init__(self, temperature: float = 0.45, max_new_tokens: int = 90, force_cpu: bool = False) -> None:
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.force_cpu = force_cpu
        self.available = False
        self.error = ""
        self.model = None
        self.dataset = None
        self.device = None

        if torch is None or load_checkpoint is None or CHECKPOINT_PATH is None:
            self.error = "PyTorch or the tiny model loader is unavailable."
            return
        if not CHECKPOINT_PATH.exists():
            self.error = "No neural checkpoint found yet."
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
            self.model, self.dataset = load_checkpoint(self.device)
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def generate_take(self, query: str, results: list[dict[str, str]]) -> str:
        if not self.available or self.model is None or self.dataset is None or self.device is None:
            return ""

        fact_lines = []
        for result in results[:2]:
            snippet = result.get("snippet", "").replace("\n", " ").strip()
            if snippet:
                fact_lines.append(f"{result['title']}: {snippet}")
            else:
                fact_lines.append(result["title"])

        prompt = (
            f"question: {query}\n"
            f"facts: {' '.join(fact_lines)}\n"
            "my take: "
        )
        seed_tokens = self.dataset.encode_string(prompt)
        encoded_prompt = self.dataset.decode_tokens(seed_tokens)
        output = self.model.generate(
            seed_tokens=seed_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            dataset=self.dataset,
            device=self.device,
        )

        continuation = output[len(encoded_prompt):] if output.startswith(encoded_prompt) else output
        return clean_generated_text(continuation)


def wikipedia_search(query: str, limit: int = 3) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": limit,
        }
    )
    search_url = f"https://en.wikipedia.org/w/api.php?{params}"
    data = fetch_json(search_url)

    results = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = re.sub(r"<.*?>", "", item.get("snippet", ""))
        snippet = html.unescape(snippet).strip()
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        results.append(
            {
                "title": title,
                "snippet": snippet,
                "url": page_url,
                "source": "wikipedia.org",
            }
        )
    return results


def duckduckgo_search(query: str, limit: int = 5) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"q": query})
    search_url = f"https://html.duckduckgo.com/html/?{params}"
    page = fetch_text(search_url)
    parser = DuckDuckGoHTMLParser()
    parser.feed(page)
    parser.close()
    return parser.results[:limit]


def search_web(query: str, limit: int) -> list[dict[str, str]]:
    try:
        results = duckduckgo_search(query, limit=limit)
        if results:
            return results
    except Exception:
        pass
    return wikipedia_search(query, limit=limit)


def build_grounded_reply(
    query: str,
    results: list[dict[str, str]],
    traits: list[str],
    training_text: str,
    neural_voice: NeuralPersonalityVoice | None = None,
) -> str:
    if not results:
        style_line = pick_style_line(training_text, query)
        pieces = ["I could not find a good live result for that search."]
        if style_line:
            pieces.append(style_line)
        return " ".join(pieces)

    normalized_results = []
    for result in results:
        clean_url = decode_duckduckgo_redirect(html.unescape(result["url"]))
        normalized_results.append(
            {
                **result,
                "url": clean_url,
                "source": infer_source_name(clean_url),
            }
        )

    tone_traits = ", ".join(traits[:3]) if traits else "clear, practical, calm"
    intro = f"I looked it up online. Here is the clearest summary I found, in a {tone_traits} tone:"

    summary_parts = []
    for result in normalized_results[:3]:
        if result["snippet"]:
            summary_parts.append(f"{result['title']} ({result['source']}): {result['snippet']}.")
        else:
            summary_parts.append(f"{result['title']} ({result['source']}) looks relevant.")

    source_line = "Sources: " + ", ".join(result["url"] for result in normalized_results[:3])

    style_line = None
    if neural_voice is not None:
        neural_take = neural_voice.generate_take(query, normalized_results)
        if neural_take:
            style_line = f"My neural take: {neural_take}"

    if not style_line:
        matched_style_line = pick_style_line(training_text, query)
        if matched_style_line:
            style_line = f"My personality take: {matched_style_line}"

    pieces = [intro, " ".join(summary_parts)]
    if style_line:
        pieces.append(style_line)
    pieces.append(source_line)
    return " ".join(pieces)


def chat(args: argparse.Namespace) -> None:
    profile_text = read_text(PROFILE_PATH)
    training_text = read_text(TRAINING_PATH)
    traits = extract_profile_traits(profile_text)
    neural_voice = NeuralPersonalityVoice(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        force_cpu=args.cpu,
    )

    print("Internet Search Chatbot")
    if neural_voice.available:
        print("Neural personality mode is on.")
    else:
        print(f"Neural personality mode is off. {neural_voice.error}")
    print("Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() == "exit":
            print("AI: See you soon.")
            break

        try:
            results = search_web(user_message, limit=args.limit)
            reply = build_grounded_reply(
                user_message,
                results,
                traits,
                training_text,
                neural_voice=neural_voice,
            )
        except Exception as exc:
            reply = (
                "I tried to search the internet, but the request failed. "
                f"Error: {exc}"
            )

        print("AI:", reply)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chatbot that searches the internet before replying.")
    parser.add_argument("--limit", type=int, default=3, help="How many search results to fetch.")
    parser.add_argument("--max-new-tokens", type=int, default=90, help="Neural style tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.45, help="Neural sampling temperature.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for the PyTorch model.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    chat(args)


if __name__ == "__main__":
    main()
