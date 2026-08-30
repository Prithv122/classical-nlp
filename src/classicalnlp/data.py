"""Corpora.

The English benchmark is 20 Newsgroups, loaded **with headers, footers and quoted text
removed**. That removal is not tidiness: the headers contain the newsgroup name, so a
classifier trained on the raw version learns to read the label off the document and scores
in the high nineties while having learned nothing. It is the most famous leak in a
standard dataset and it is one keyword argument away.

The multilingual samples are hand-written and tiny. They exist to exercise the
normalisation and tokenisation paths on Devanagari and Kannada, and no accuracy number is
computed from them -- a dozen sentences cannot support one.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.datasets import fetch_20newsgroups

#: Four groups, two easy pairs and one hard pair -- `talk.religion.misc` and
#: `soc.religion.christian` overlap heavily, which is what keeps macro-F1 honest.
DEFAULT_CATEGORIES = (
    "rec.sport.hockey",
    "sci.space",
    "soc.religion.christian",
    "talk.religion.misc",
)

REMOVE = ("headers", "footers", "quotes")


@dataclass(frozen=True)
class Corpus:
    documents: list[str]
    labels: list[int]
    label_names: list[str]

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def class_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(self.label_names, 0)
        for label in self.labels:
            counts[self.label_names[label]] += 1
        return counts


def load_newsgroups(
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    subset: str = "train",
    *,
    remove: tuple[str, ...] = REMOVE,
    min_characters: int = 40,
) -> Corpus:
    """Fetch 20 Newsgroups (cached in ~/scikit_learn_data after the first call).

    Documents shorter than ``min_characters`` are dropped: with headers removed, a few
    hundred posts are empty or a single line of quoted text, and they add nothing but
    noise to both models equally.
    """
    bunch = fetch_20newsgroups(
        subset=subset, categories=list(categories), remove=remove, random_state=42, shuffle=True
    )
    documents, labels = [], []
    for text, label in zip(bunch.data, bunch.target, strict=True):
        if len(text.strip()) >= min_characters:
            documents.append(text)
            labels.append(int(label))
    return Corpus(documents=documents, labels=labels, label_names=list(bunch.target_names))


#: (text, language) pairs for the normalisation demos and tests.
MULTILINGUAL_SAMPLES: tuple[tuple[str, str], ...] = (
    ("यह एक परीक्षण वाक्य है और यह ठीक काम करता है", "hi"),
    ("क़िला बहुत पुराना है ०१२ नंबर वाला", "hi"),
    ("ಇದು ಒಂದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಮತ್ತು ಇದು ಚೆನ್ನಾಗಿ ಕೆಲಸ ಮಾಡುತ್ತದೆ", "kn"),
    ("ಬೆಂಗಳೂರಿನ ಗ್ರಂಥಾಲಯದಲ್ಲಿ ೧೨೩ ಪುಸ್ತಕಗಳಿವೆ", "kn"),
    ("The quick brown fox jumps over the lazy dog", "en"),
    ("Mixed script: नमस्ते from Bengaluru, 42 books", "en"),
)
