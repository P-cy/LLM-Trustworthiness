"""CSV + postprocess helpers (shared)."""
import re


def postprocess(text):
    """Strip CoT/think tags and trailing role markers. Never return empty str.

    Removes <think>...</think>, dangling <think>, <im_end>, chat role markers.
    Collapses newlines to single space (CSV one-row invariant).
    """
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE)
    text = text.replace("</think>", "")
    for tok in ("<|im_start|>", "<|im_end|>"):
        text = text.replace(tok, "")
    text = text.strip()
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def lang_of(query):
    """Crude language detection: Thai if any Thai char, else English."""
    for ch in query:
        if "฀" <= ch <= "๿":
            return "th"
    return "en"


def fallback_for(category, lang):
    """Category + language-aware safe fallback strings."""
    if category.startswith("S_"):
        if lang == "th":
            return "ขออภัย ฉันไม่สามารถให้ความช่วยเหลือในเรื่องนี้ได้ หากคุณมีคำถามอื่นที่ปลอดภัย ฉันยินดีช่วย"
        return "I'm sorry, I can't help with that. If you have another safe question, I'd be glad to assist."
    if lang == "th":
        return "ขออภัย ฉันไม่สามารถให้ข้อมูลเฉพาะเจาะจงนี้ได้ในขณะนี้"
    return "I'm sorry, I can't provide that specific information right now."
