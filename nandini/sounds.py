"""Which sound classes each syllable contains, in SLP1 (see text.py)."""
from .text import SLP_VOWELS

CLASSES = {
    "aspirated": set("KGCJWQTDPB"),
    "retroflex": set("wWqQRz"),
    "sibilant": set("Szs"),
    "visarga": {"H"},
    "anusvara": {"M"},
    "long_vowel": set("AIUF"),  # आ ई ऊ ॠ
}


def syllable_classes(verse):
    """Per syllable: its SLP1 text and a dict of class -> bool, plus 'conjunct' (two or more consonants before its vowel)."""
    out = []
    for a, b in verse.spans:
        s = verse.slp[a:b].replace(" ", "")
        onset = 0
        for c in s:
            if c in SLP_VOWELS:
                break
            if c not in "'":
                onset += 1
        flags = {name: any(c in chars for c in s) for name, chars in CLASSES.items()}
        flags["conjunct"] = onset >= 2
        out.append((s, flags))
    return out
