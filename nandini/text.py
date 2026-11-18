"""Devanagari to SLP1, and each verse's syllables and comparison stream."""
import numpy as np

# Devanagari to SLP1, a spelling of Sanskrit that uses one Latin letter per sound, which makes comparing texts simple.
VOWELS = {"अ": "a", "आ": "A", "इ": "i", "ई": "I", "उ": "u", "ऊ": "U", "ऋ": "f", "ॠ": "F", "ऌ": "x", "ॡ": "X",
          "ए": "e", "ऐ": "E", "ओ": "o", "औ": "O"}
MATRAS = {"ा": "A", "ि": "i", "ी": "I", "ु": "u", "ू": "U", "ृ": "f", "ॄ": "F", "ॢ": "x", "ॣ": "X",
          "े": "e", "ै": "E", "ो": "o", "ौ": "O"}
CONS = {"क": "k", "ख": "K", "ग": "g", "घ": "G", "ङ": "N", "च": "c", "छ": "C", "ज": "j", "झ": "J", "ञ": "Y",
        "ट": "w", "ठ": "W", "ड": "q", "ढ": "Q", "ण": "R", "त": "t", "थ": "T", "द": "d", "ध": "D", "न": "n",
        "प": "p", "फ": "P", "ब": "b", "भ": "B", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "L", "व": "v",
        "श": "S", "ष": "z", "स": "s", "ह": "h"}
MARKS = {"ं": "M", "ः": "H", "ँ": "M"}
VIRAMA, NUKTA, AVAGRAHA, OM = "्", "़", "ऽ", "ॐ"
SLP_VOWELS = set("aAiIuUfFxXeEoO")
SLP_MARKS = set("MH")
# How each SLP1 sound is written for the aligner, whose model only knows Latin letters. Close sounds share a spelling (ष and श are both "sh").
LATIN = {"a": "a", "A": "aa", "i": "i", "I": "ii", "u": "u", "U": "uu", "f": "ri", "F": "rii", "x": "li", "X": "lii",
         "e": "e", "E": "ai", "o": "o", "O": "au",
         "k": "k", "K": "kh", "g": "g", "G": "gh", "N": "n", "c": "c", "C": "ch", "j": "j", "J": "jh", "Y": "n",
         "w": "t", "W": "th", "q": "d", "Q": "dh", "R": "n", "t": "t", "T": "th", "d": "d", "D": "dh", "n": "n",
         "p": "p", "P": "ph", "b": "b", "B": "bh", "m": "m", "y": "y", "r": "r", "l": "l", "v": "v", "S": "sh",
         "z": "sh", "s": "s", "h": "h", "M": "m", "H": "h"}


def to_slp1(text):
    """Spaces are kept, with runs collapsed; anything outside the table is dropped."""
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        # A consonant carries an inherent "a", unless a vowel sign or a virama follows it.
        if ch in CONS:
            out.append(CONS[ch])
            j = i + 1
            if j < len(text) and text[j] == NUKTA:
                j += 1
            if j < len(text) and text[j] in MATRAS:
                out.append(MATRAS[text[j]])
                j += 1
            elif j < len(text) and text[j] == VIRAMA:
                j += 1
            else:
                out.append("a")
            i = j
            continue
        if ch in VOWELS:
            out.append(VOWELS[ch])
        elif ch in MARKS:
            out.append(MARKS[ch])
        elif ch == AVAGRAHA:
            out.append("'")
        elif ch == OM:
            out.append("AUM")
        elif ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
        i += 1
    return "".join(out).strip()


def syllable_spans(slp):
    """(start, end) per vowel: the vowel with the consonants before it and any anusvara or visarga after it.
    Consonants that end a word join its last syllable."""
    spans, words, i = [], [], 0
    for w in slp.split(" "):
        words.append((i, i + len(w)))
        i += len(w) + 1
    for ws, we in words:
        start, k, word_spans = ws, ws, []
        while k < we:
            if slp[k] in SLP_VOWELS:
                e = k + 1
                while e < we and slp[e] in SLP_MARKS:
                    e += 1
                word_spans.append([start, e])
                start, k = e, e
                continue
            k += 1
        if word_spans:
            word_spans[-1][1] = we
        elif spans and we > ws:
            spans[-1][1] = we
        spans.extend(word_spans)
    return [tuple(s) for s in spans]


class Verse:
    """A verse's syllables, and the characters the transcripts are compared with."""

    def __init__(self, devanagari):
        self.slp = to_slp1(devanagari.replace(OM, "ओम्"))  # the aligner and ASR compare the spoken form of ॐ
        self.spans = syllable_spans(self.slp)
        syl_of = {i: k for k, (a, b) in enumerate(self.spans) for i in range(a, b)}
        keep = [i for i, c in enumerate(self.slp) if c not in " '"]
        # `chars` drops spaces, so a transcript that splits words differently isn't counted as wrong. `char_syl` gives each character's syllable.
        self.chars = "".join(self.slp[i] for i in keep)
        self.char_syl = np.array([syl_of[i] for i in keep])
        self.n = len(self.spans)


# SLP1 back to Devanagari and to IAST, for showing syllables and what the ASR heard.
_DEV_V = {v: k for k, v in VOWELS.items()}
_DEV_M = {v: k for k, v in MATRAS.items()}
_DEV_C = {v: k for k, v in CONS.items()}
_DEV_K = {"M": "ं", "H": "ः"}
_IAST = {"a": "a", "A": "ā", "i": "i", "I": "ī", "u": "u", "U": "ū", "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ", "e": "e",
         "E": "ai", "o": "o", "O": "au", "M": "ṃ", "H": "ḥ", "k": "k", "K": "kh", "g": "g", "G": "gh", "N": "ṅ",
         "c": "c", "C": "ch", "j": "j", "J": "jh", "Y": "ñ", "w": "ṭ", "W": "ṭh", "q": "ḍ", "Q": "ḍh", "R": "ṇ",
         "t": "t", "T": "th", "d": "d", "D": "dh", "n": "n", "p": "p", "P": "ph", "b": "b", "B": "bh", "m": "m",
         "y": "y", "r": "r", "l": "l", "v": "v", "S": "ś", "z": "ṣ", "s": "s", "h": "h", "'": "'"}


def deva(slp):
    """One syllable (consonants, vowel, marks; or word-final consonants) from SLP1 back to Devanagari."""
    out = []
    for i, c in enumerate(slp):
        if c in _DEV_C:
            out.append(_DEV_C[c])
            if i + 1 == len(slp) or slp[i + 1] not in _DEV_V:
                out.append(VIRAMA)
        elif c in _DEV_V:
            if i > 0 and slp[i - 1] in _DEV_C:
                out.append(_DEV_M.get(c, ""))
            else:
                out.append(_DEV_V[c])
        elif c in _DEV_K:
            out.append(_DEV_K[c])
        elif c == "'":
            out.append(AVAGRAHA)
    return "".join(out)


def iast(slp):
    return "".join(_IAST.get(c, c) for c in slp)
