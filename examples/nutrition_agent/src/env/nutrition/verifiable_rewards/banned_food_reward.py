import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

_NEGATION_WINDOW_TOKENS = 3  # how many tokens around a match to scan for negation cues

# Phrases like "no", "without", "free", "free from", "zero", etc.
_NEGATION_CUES = {
    "no", "without", "free", "free-from", "freefrom", "zero", "avoid", "exclude", "excluding",
    "omitting", "omit", "minus", "sans"
}

# Common hyphenated negations, we’ll also catch via token window.
_NEGATION_PATTERNS = [
    re.compile(r"\bno\b", re.I),
    re.compile(r"\bwithout\b", re.I),
    re.compile(r"\bfree(?:\s+from)?\b", re.I),
    re.compile(r"\bzero\b", re.I),
    re.compile(r"\bavoid(?:ing)?\b", re.I),
    re.compile(r"\bexclude(?:d|ing)?\b", re.I),
    re.compile(r"\bomit(?:ted|ting)?\b", re.I),
    re.compile(r"\bsans\b", re.I),
]

def _normalize(text: str) -> str:
    # Lowercase + Unicode normalize (so “é” vs “é” treated same)
    return unicodedata.normalize("NFKC", text).lower()

def _tokenize(text: str) -> List[str]:
    # tokens include hyphen‐joined words as single tokens (e.g., egg-free)
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)

def _plural_variants(word: str) -> Sequence[str]:
    """Very light plural handling without extra deps."""
    base = word
    variants = {base}
    if len(base) == 0:
        return list(variants)
    # standard s
    variants.add(base + "s")
    # es (tomato -> tomatoes handled poorly; we keep simple)
    if base.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(base + "es")
    # y -> ies
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        variants.add(base[:-1] + "ies")
    # f/fe -> ves (knife -> knives); heuristic
    if base.endswith("f"):
        variants.add(base[:-1] + "ves")
    if base.endswith("fe"):
        variants.add(base[:-2] + "ves")
    return list(variants)

def _build_keyword_regex(kw: str) -> re.Pattern:
    """
    Build a regex that matches the keyword with simple inflectional variants:
    - single word: egg -> \b(?:egg|eggs|...)\b
    - multi word: peanut butter -> \bpeanut[-\s]+butter(s|es|...)?\b
      (pluralization is applied to the last token)
    """
    kw_norm = _normalize(kw).strip()
    parts = [p for p in re.findall(r"[a-z0-9]+", kw_norm) if p]
    if not parts:
        # fallback: literal word boundary around normalized kw
        return re.compile(rf"\b{re.escape(kw_norm)}\b", re.I)

    if len(parts) == 1:
        # Single word → union of simple plural variants
        alts = "|".join(re.escape(v) for v in _plural_variants(parts[0]))
        return re.compile(rf"\b(?:{alts})\b", re.I)
    else:
        # Multiword: pluralize only the last token; allow space or hyphen between words
        head = r"[-\s]+".join(re.escape(p) for p in parts[:-1])
        last_alts = "|".join(re.escape(v) for v in _plural_variants(parts[-1]))
        return re.compile(rf"\b{head}[-\s]+(?:{last_alts})\b", re.I)

def _has_negation_context(text: str, match_span: Tuple[int, int]) -> bool:
    """
    Detect if match is in a negated/safe context like "egg-free", "no eggs",
    "without egg(s)", "free from eggs)" within a small window.
    """
    start, end = match_span
    # Quick direct checks around the span (hyphenated forms)
    window_lo = max(0, start - 30)
    window_hi = min(len(text), end + 30)
    window = text[window_lo:window_hi]

    # Fast pattern checks
    for pat in _NEGATION_PATTERNS:
        if pat.search(window):
            # ensure the cue is not clearly far away (we also token-scan below)
            break
    else:
        # not found with fast patterns—still run token window check
        pass

    # Token-level proximity check
    tokens = _tokenize(text)
    # Map char index to token index by rebuilding with spans
    idx = 0
    spans = []
    for t in tokens:
        m = re.search(re.escape(t), text[idx:])
        if not m:
            continue
        s = idx + m.start()
        e = idx + m.end()
        spans.append((t, s, e))
        idx = e

    # find tokens that overlap the match; then scan neighbors
    hit_positions = [i for i, (_, s, e) in enumerate(spans) if not (e <= start or s >= end)]
    if not hit_positions:
        return False
    left = max(0, hit_positions[0] - _NEGATION_WINDOW_TOKENS)
    right = min(len(spans), hit_positions[-1] + _NEGATION_WINDOW_TOKENS + 1)
    neighbor_tokens = [t for (t, _, _) in spans[left:right]]

    # Direct cues among nearby tokens
    if any(t in _NEGATION_CUES for t in neighbor_tokens):
        return True

    # Hyphenated “egg-free” style
    for t in neighbor_tokens:
        if t.endswith("-free") or t.endswith("-free*"):  # wildcards not needed but kept illustrative
            return True

    # Phrases like "free from"
    joined = " ".join(neighbor_tokens)
    if re.search(r"\bfree\s+from\b", joined):
        return True

    return False


def verify_no_banned(plan_json: Dict, banned_keywords: Sequence[str]):
    """
    Ensure no banned keywords (with basic inflection handling) appear in meal name or description,
    excluding safe/negated contexts (e.g., 'egg-free', 'no eggs', 'without egg').
    Returns:
      R_banned: 1.0 if clean, 0.0 if any violation
      violations: list of dicts with day, meal_name, offending_keyword, text_snippet
    """
    # Pre-compile patterns per keyword
    patterns = [(kw, _build_keyword_regex(kw)) for kw in banned_keywords]

    violations = []
    R_banned = 1.0

    for day in plan_json.get("dailyMealPlans", []):
        d = day.get("day")
        for m in day.get("meals", []):
            name = m.get("name", "") or ""
            desc = m.get("description", "") or ""
            text_raw = f"{name} {desc}"
            text = _normalize(text_raw)

            for kw, pat in patterns:
                for match in pat.finditer(text):
                    if _has_negation_context(text, match.span()):
                        # safe context; skip
                        continue
                    # Record violation with a short snippet for debugging
                    s, e = match.span()
                    lo = max(0, s - 25)
                    hi = min(len(text_raw), e + 25)
                    snippet = text_raw[lo:hi]
                    violations.append({
                        "day": d,
                        "meal_name": name,
                        "offending_keyword": kw,
                        "matched_form": text_raw[s:e],
                        "context": snippet
                    })
                    R_banned = 0.0
                    # don’t break immediately; collect all matches
    return R_banned, violations