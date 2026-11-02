"""Repairs for text extracted from PDFs.

Two kinds of damage are fixed, both without touching anything that could be real scientific notation:

1. Real Unicode ligature characters (the single characters for "fi", "fl", "ff", "ffi", "ffl") are
   spelled out as plain letters, so search and embeddings see ordinary words.

2. Some PDF fonts draw the "f" of an fi / fl / ff ligature with a glyph that pypdf decodes as an
   uppercase "H" or "T": "reHlecting" (reflecting), "efHiciency" (efficiency), "signiTicantly"
   (significantly). The rule is deliberately narrow: an uppercase H or T strictly INSIDE an otherwise
   lowercase word is not English, so it is replaced by "f".

   What it deliberately does not touch:
   * chemistry: BaTiO3, CH3NH3PbI3, ZnTe, Ti3C2Tx all begin with a capital or contain digits;
   * short words, so units such as kHz, mHz and mmHg are safe (MIN_WORD_LENGTH);
   * a damaged word that STARTS with the wrong letter ("Hind" for "find", "Tinite" for "finite"):
     capital H and T begin thousands of correct words (The, High, Temperature), so no rule can tell them apart.

The H/T mapping was read off one real PDF (arXiv 2510.24874). Other fonts may use other substitutes; those
words are simply left as they are.
"""
import re
from typing import Tuple

_LIGATURES = str.maketrans({
    "\ufb00": "ff",   # ff
    "\ufb01": "fi",   # fi
    "\ufb02": "fl",   # fl
    "\ufb03": "ffi",  # ffi
    "\ufb04": "ffl",  # ffl
    "\ufb05": "st",   # long s + t
    "\ufb06": "st",   # st
})

# A whole token made of lowercase letters in which at least one uppercase H or T sits between lowercase letters.
_DAMAGED_WORD = re.compile(r"(?<![A-Za-z0-9_])[a-z]+(?:[HT][a-z]+)+(?![A-Za-z0-9_])")

MIN_WORD_LENGTH = 5  # keeps kHz, mHz, mmHg, pTyr and similar notation out of reach of the repair


def repair_pdf_text(text: str) -> Tuple[str, int]:
    """(repaired text, number of words repaired). Safe to apply twice."""
    if not text:
        return "", 0

    text = text.translate(_LIGATURES)

    repaired = 0

    def fix(match: "re.Match[str]") -> str:
        nonlocal repaired
        word = match.group(0)
        if len(word) < MIN_WORD_LENGTH:
            return word
        repaired += 1
        return word.replace("H", "f").replace("T", "f")

    return _DAMAGED_WORD.sub(fix, text), repaired
