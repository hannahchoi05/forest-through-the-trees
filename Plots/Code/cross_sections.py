"""
The 36 cross-sections from Table D.1 of Bryzgalova-Pelger-Zhu (2020).

Each cross-section is a triple (Size, X, Y) where X and Y are drawn from the
9 non-size characteristics. C(9, 2) = 36. The Id column matches the paper's
labeling so cross-sections can be cross-referenced with Table D.1.

The "char_key" is a short, filesystem-safe identifier we use to name the
output directory for each cross-section's backtest. e.g. cross-section 16
(Size, Prof, Inv) lives in `backtest_results/size_op_inv/`.
"""

from dataclasses import dataclass


# Map paper characteristic names to our column names in the panel.
# Update if your panel uses different column names.
CHAR_NAME_MAP = {
    "Size":  "LME",
    "Val":   "BEME",
    "Mom":   "r12_2",
    "Prof":  "OP",
    "Inv":   "Investment",
    "SRev":  "ST_Rev",
    "LRev":  "LT_Rev",
    "Acc":   "AC",
    "IVol":  "IdioVol",
    "Turn":  "LTurnover",
}


@dataclass(frozen=True)
class CrossSection:
    id: int
    char1: str   # always "Size"
    char2: str
    char3: str

    @property
    def key(self) -> str:
        """Filesystem-safe key, e.g. 'size_op_inv'."""
        return "_".join(
            CHAR_NAME_MAP[c].lower() for c in (self.char1, self.char2, self.char3)
        )

    @property
    def panel_chars(self) -> tuple:
        """Column names in the data panel, e.g. ('LME', 'OP', 'Investment')."""
        return tuple(CHAR_NAME_MAP[c] for c in (self.char1, self.char2, self.char3))

    @property
    def label(self) -> str:
        return f"{self.char1} × {self.char2} × {self.char3}"


# Exact ordering and (Char 1, Char 2, Char 3) triples from Table D.1.
# Sorted by Id so the index in the list matches the paper's numbering.
CROSS_SECTIONS_RAW = [
    # Id, Char2, Char3
    ( 1, "Val",  "Mom"),
    ( 2, "Val",  "Prof"),
    ( 3, "Val",  "Inv"),
    ( 4, "Val",  "SRev"),
    ( 5, "Val",  "LRev"),
    ( 6, "Val",  "Acc"),
    ( 7, "Val",  "IVol"),
    ( 8, "Val",  "Turn"),
    ( 9, "Mom",  "Prof"),
    (10, "Mom",  "Inv"),
    (11, "Mom",  "SRev"),
    (12, "Mom",  "LRev"),
    (13, "Mom",  "Acc"),
    (14, "Mom",  "IVol"),
    (15, "Mom",  "Turn"),
    (16, "Prof", "Inv"),
    (17, "Prof", "SRev"),
    (18, "Prof", "LRev"),
    (19, "Prof", "Acc"),
    (20, "Prof", "IVol"),
    (21, "Prof", "Turn"),
    (22, "Inv",  "SRev"),
    (23, "Inv",  "LRev"),
    (24, "Inv",  "Acc"),
    (25, "Inv",  "IVol"),
    (26, "Inv",  "Turn"),
    (27, "SRev", "LRev"),
    (28, "SRev", "Acc"),
    (29, "SRev", "IVol"),
    (30, "SRev", "Turn"),
    (31, "LRev", "Acc"),
    (32, "LRev", "IVol"),
    (33, "LRev", "Turn"),
    (34, "Acc",  "IVol"),
    (35, "Acc",  "Turn"),
    (36, "IVol", "Turn"),
]

CROSS_SECTIONS = [
    CrossSection(id=i, char1="Size", char2=c2, char3=c3)
    for (i, c2, c3) in CROSS_SECTIONS_RAW
]


def get_by_id(cs_id: int) -> CrossSection:
    return CROSS_SECTIONS[cs_id - 1]


def get_by_chars(chars: tuple) -> CrossSection:
    """Look up cross-section by panel column names (any order, with LME present)."""
    chars_set = set(chars)
    for cs in CROSS_SECTIONS:
        if set(cs.panel_chars) == chars_set:
            return cs
    raise KeyError(f"No cross-section matching {chars}")


if __name__ == "__main__":
    print(f"Total cross-sections: {len(CROSS_SECTIONS)}")
    assert len(CROSS_SECTIONS) == 36
    print(f"\nCross-section #16 (paper's Size × Prof × Inv example):")
    cs16 = get_by_id(16)
    print(f"  label    = {cs16.label}")
    print(f"  key      = {cs16.key}")
    print(f"  panel    = {cs16.panel_chars}")
    print(f"\nLookup by chars: get_by_chars(('LME', 'OP', 'Investment')) ->")
    print(f"  {get_by_chars(('LME', 'OP', 'Investment'))}")