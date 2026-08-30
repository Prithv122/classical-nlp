# Interview Prep — Classical NLP Baselines

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ A normalisation layer that handles Latin, Devanagari and Kannada — NFC composition,
Indic digit folding, script detection, and a tokenizer that classifies characters by
Unicode general category. Four representations built as scikit-learn pipelines — TF-IDF on
words, TF-IDF on 3–5 character n-grams, LSA (TF-IDF then 300-dim SVD), and an optional
sentence-transformer — each feeding the same logistic regression, so the comparison is of
features. Evaluation is 5-fold stratified CV with the vectorizer fitted inside the fold,
reported as macro-F1 with per-class breakdown, and differences between models get a paired
bootstrap CI and a McNemar test. Topic modelling is NMF over TF-IDF with c-TF-IDF labels,
scored by NPMI coherence.

### Q2. Which representation won?

_A:_ None of them, and that is the finding. LSA scored 0.8123 macro-F1 against 0.8035 for
word TF-IDF, but the paired bootstrap gives a 95% CI of [−0.005, +0.022] with p = 0.22, and
McNemar — 68 items where only LSA was right, 61 where only TF-IDF was — gives p = 0.60.
Both tests say the same thing: on this corpus there is no evidence the dense representation
is better. I would ship TF-IDF, because it is faster, inspectable, and I can tell you which
features drove a prediction. The honest headline is "the cheap baseline is not beaten",
which is the result you need before spending money on an embedding service.

### Q3. What's the weakest part of this, and what would break first under load?

_A:_ The weakest part is the multilingual claim. The Indic work is real — the
normalisation, the tokenisation, the script detection are tested — but the *classification*
numbers are English only. Six hand-written sentences cannot support an accuracy figure and
I do not report one; the missing piece is a real Indic benchmark, and it is the top open
question. Under load: `TfidfVectorizer` keeps vocabulary in memory and needs a full pass,
so past a few hundred thousand documents it becomes `HashingVectorizer` plus `partial_fit`,
losing IDF and inspectable feature names. And the bootstrap — 2,000 resamples of macro-F1 —
becomes slower than training the models it is comparing.

### Q4. How do you know it works? What did you measure, and against what baseline?

_A:_ Three baselines, deliberately. For representations, TF-IDF is the baseline the
expensive options have to beat, and they do not. For the topic model, the baseline is my
own first attempt: coherence 0.213 with every topic labelled "the, of, to", against 0.361
after fixing the c-TF-IDF weighting and adding a stoplist — the coherence number is what
caught it, because both outputs look like *some* output if you only read the word lists.
For evaluation hygiene, the baseline is the leaky version of the same pipeline, which is
how I found that the warning everyone repeats is quantitatively the small one.

### Q5. You measured data leakage and found it barely mattered. Explain.

_A:_ Fitting a TF-IDF vectorizer on the whole dataset before cross-validating inflated
macro-F1 by 0.0025 at n = 2,082 and 0.0011 at n = 400 — noise. The reason is that the
vectorizer is **unsupervised**: it never sees a label, so all it leaks is vocabulary and
IDF statistics, which are stable once you have a few hundred documents. So I built the
comparison that does matter: chi-squared feature selection, which *does* use the labels,
fitted outside the folds. That inflated by 0.0288 at n = 400 — more than ten times — and
shrank to 0.0052 at n = 2,082, because the more data you have, the less a leak buys. The
general rule I would take into a review is: it is not "fitting outside the fold" that
hurts, it is fitting a step that *saw the labels* outside the fold, and the damage is
largest exactly when the dataset is small enough that you were already going to be
optimistic.

---

## 30-second pitch

A comparison of classical text representations done so that it can produce a negative
result: TF-IDF, character n-grams and LSA on 20 Newsgroups, with paired bootstrap
confidence intervals and McNemar tests, which show the dense representation does *not*
beat TF-IDF. Underneath it is a text-normalisation layer that gets Devanagari and Kannada
right — including the bug where Python's `\w` silently deletes every combining mark — and a
topic model whose labels went from "the, of, to" to "god, jesus, christ" once the c-TF-IDF
weighting was fixed, caught by tracking NPMI coherence rather than by reading word lists.
