"""BM25 phrase-token helpers for ingest indexing."""

import re

from config import ANCHOR_HINT_WORDS, ANCHOR_PHRASE_RE, ANCHOR_STOPWORDS


def extract_anchor_phrases(*texts: str) -> list[str]:
    """Pull 'Label + number' locators from chunk text."""
    found, seen = [], set()
    for text in texts:
        if not text:
            continue
        for m in ANCHOR_PHRASE_RE.finditer(text):
            label, num = m.group(1), m.group(2)
            phrase = normalize_anchor_phrase(f"{label} {num}")
            if not is_useful_anchor_phrase(phrase):
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(phrase)
    return found


def is_useful_anchor_phrase(phrase: str) -> bool:
    """Keep real locators; drop stopword noise like 'the 9'."""
    m = re.match(r"(?i)^([a-z]+)\s+(\d+)$", phrase.strip())
    if not m:
        return False
    word = m.group(1).lower()
    if word in ANCHOR_HINT_WORDS:
        return True
    return word not in ANCHOR_STOPWORDS


def normalize_anchor_phrase(phrase: str) -> str:
    """Collapse spaces; Q9 → Question 9; capitalize label lightly."""
    phrase = re.sub(r"\s+", " ", phrase).strip()
    m = re.fullmatch(r"(?i)q\s*#?\s*(\d+)", phrase)
    if m:
        return f"Question {m.group(1)}"
    m = re.match(r"(?i)^([a-z]+)\s*#?\s*(\d+)\s*$", phrase)
    if not m:
        return phrase
    word, num = m.group(1).lower(), m.group(2)
    aliases = {"hw": "Homework", "q": "Question"}
    return f"{aliases.get(word, word.capitalize())} {num}"


def phrase_token(phrase: str) -> str:
    """Any Label+number locator → one BM25 term."""
    norm = normalize_anchor_phrase(phrase).lower()
    return "ph" + re.sub(r"[^a-z0-9]+", "", norm)


def phrases_in_text(text: str) -> list[str]:
    """Locator phrases appearing inside a chunk (for BM25 index enrichment)."""
    return extract_anchor_phrases(text)


def bm25_index_text(text: str) -> str:
    """Original text + dynamically derived whole-phrase tokens for sparse BM25."""
    tokens = [phrase_token(p) for p in phrases_in_text(text)]
    if not tokens:
        return text
    seen, uniq = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return text + "\n" + " ".join(uniq)
