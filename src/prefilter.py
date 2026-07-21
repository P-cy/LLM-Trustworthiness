"""Pre-filter: regex-only detection of obfuscation + language hint.

No model calls. Returns a PrefilterHit describing what (if anything) was
detected. The downstream triage treats a jailbreak-suspected hit as a strong
prior toward S_jailbreak (guard still gets final say, but ties -> stricter).
"""
import re
import unicodedata


class PrefilterHit:
    __slots__ = ("flag", "signals", "lang")

    def __init__(self, flag="ok", signals=None, lang="en"):
        # flag in {ok, suspicious, jailbreak}
        self.flag = flag
        self.signals = signals or []
        self.lang = lang


_HOMOGLYPHS = {
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "ο": "o", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "і": "i", "ѕ": "s", "ԁ": "d",
}

_BASE64 = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")
_HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{32,}\b")
_LEET = re.compile(r"(?i)\b[\w]*(?:[0-9]|[\\@\$£]{1})[\w]*\b")

_JAILBREAK_KEYWORDS = [
    # English
    "ignore previous", "ignore all previous", "disregard", "forget your instructions",
    "you are now", "act as", "roleplay", "pretend you are", "jailbreak",
    "DAN", "developer mode", "override", "new rules", "no restrictions",
    "do anything now", "unrestricted", "bypass", "system prompt", "reveal your prompt",
    "ignore the above", "in hypothetical",
    # Thai (rough)
    "ลืมคำสั่งเดิม", "เพิกเฉยต่อคำสั่ง", "แกล้งทำเป็น", "เล่นบทบาทเป็น",
    "ตอนนี้คุณคือ", "ไม่มีข้อจำกัด", "ข้ามระบบ", "โหมดนักพัฒนา",
]


def _homoglyph_count(s):
    return sum(1 for ch in s if ch in _HOMOGLYPHS)


def _has_leetspeak(s):
    # Only flag if a "word" mixes letters and digits/symbols unusually: at least
    # one letter+digit adjacent token of length>=4. Avoids flagging normal text.
    for tok in re.findall(r"\b\w+\b", s):
        if len(tok) >= 4 and re.search(r"[a-z]", tok, re.I) and re.search(r"[0-9]", tok):
            return True
    return False


def prefilter(query):
    if not query:
        return PrefilterHit(flag="ok", lang="en")
    lang = "th" if any("฀" <= c <= "๿" for c in query) else "en"
    signals = []

    if _BASE64.search(query):
        signals.append("base64")
    if _HEX.search(query):
        signals.append("hex")
    if _has_leetspeak(query):
        signals.append("leetspeak")
    if _homoglyph_count(query) >= 2:
        signals.append("homoglyph")

    ql = query.lower()
    kb_hits = [k.lower() for k in _JAILBREAK_KEYWORDS if k.lower() in ql]
    if kb_hits:
        signals.append("jb_keywords:%d" % len(kb_hits))

    # Decision: explicit jailbreak phrasing OR payload obfuscation + jb keywords
    # -> jailbreak. Obfuscation alone -> suspicious (let model decide).
    if kb_hits:
        flag = "jailbreak"
    elif signals:
        flag = "suspicious"
    else:
        flag = "ok"
    return PrefilterHit(flag=flag, signals=signals, lang=lang)
