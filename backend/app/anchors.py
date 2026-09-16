"""Label+number locator phrases and BM25 phrase-token helpers."""

import re

from config import ANCHOR_HINT_WORDS, ANCHOR_PHRASE_RE, ANCHOR_STOPWORDS


def extract_anchor_phrases(*texts: str) -> list[str]:
    """Pull 'Label + number' locators from user/rewrite/chunk text (course-agnostic)."""
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
    """
    Any Label+number locator → one BM25 term (computed, never hardcoded).
    formula: "ph" + alnum(normalize(phrase))
    """
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


def bm25_phrase_query(*texts: str) -> str:
    """Query-side expansion: emit glued tokens for whatever locators appear."""
    tokens = [phrase_token(p) for p in extract_anchor_phrases(*texts)]
    seen, uniq = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return " ".join(uniq)


def text_has_phrase(text: str, phrase: str) -> bool:
    """Check payload literally contains the locator (generic; not quiz-specific)."""
    if not text:
        return False
    low = text.lower()
    canon = normalize_anchor_phrase(phrase)
    variants = [canon, phrase]
    # compact form: "Question 9" ↔ "Q9" only when first word starts with Q-word single letter alias
    m = re.match(r"(?i)^([a-z]+)\s+(\d+)$", canon)
    if m:
        word, num = m.group(1), m.group(2)
        variants.append(f"{word} {num}")
        variants.append(f"{word}{num}")
        if word.lower() == "question":
            variants.extend([f"Q{num}", f"Q {num}"])
        if word.lower() == "homework":
            variants.extend([f"HW{num}", f"HW {num}"])
    seen = set()
    for v in variants:
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        parts = [re.escape(p) for p in re.split(r"\s+", k) if p]
        if not parts:
            continue
        # also allow zero-space compact match already in variants
        pat = r"\s*".join(parts)
        if re.search(rf"(?<![a-z0-9]){pat}(?![a-z0-9])", low):
            return True
    return False


def phrase_match_score(text: str, phrase: str) -> int:
    return 1 if text_has_phrase(text, phrase) else 0
