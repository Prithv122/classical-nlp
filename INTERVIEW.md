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

## Router study

### Q6. Walk me through the router study. What question does it answer, and how?

_A:_ Should a support ticket be routed by a small supervised classifier or by an LLM? Arm A
is word TF-IDF plus logistic regression, trained on 8,002 Banking77 tickets. Arm B is a local
7B model — `qwen2.5:7b-instruct` at 3-bit, through Ollama — prompted zero-shot with all 77
intents. Both return a label and a confidence, and the same isotonic calibration step, fitted
on 2,001 validation tickets, is applied to both. The decision is costed: a ticket is
auto-routed only if (1 − confidence) × ratio < 1, otherwise it goes to a human. A handoff
costs 1 unit and a misroute costs the ratio, 10 at the headline. The metric is routing
decision cost per 1,000 tickets, excluding compute, compared with a paired bootstrap on the
3,080-ticket official test split, which was scored once under a protocol committed beforehand.

### Q7. Which router won, and why?

_A:_ The TF-IDF router: 589.0 against 939.0 per 1,000 tickets at 10:1, a difference of −350.0
with a 95% CI of [−397.7, −299.3], p < 0.001, excluding compute. The mechanism is the part I
would stress: A did not win by making fewer mistakes. It misrouted 42 tickets to B's 15. It
won because it auto-routed 1,686 tickets, 54.7%, while the calibrated LLM cleared the 0.9 bar
on only 338, 11.0%, and handed the rest to a human. The supporting accuracy evidence points
the same way — on an exact McNemar test, 937 tickets only A got right against 135 only B got
right — and A was lower than B at all four cost ratios under both calibrations.

### Q8. So TF-IDF beats LLMs?

_A:_ No, and I would not say that. This is one LLM configuration: one 3-bit 7B model,
zero-shot, one prompt, strict parsing, so 5.4% of its replies did not name a valid intent and
became handoffs. The supervision is unequal by design — A learned from 8,002 labelled tickets
and B from none. Compute is excluded from the cost, the 10:1 ratio is an assumption rather
than a measured cost, and it is one dataset. What the study does show is narrower: for this
task and this setup, a cheap supervised baseline with calibrated confidence had the lower
routing cost, and the break-even analysis found no setting where B was cheaper even
before compute. A dedicated decision model, arm C, was planned and is reported as not
evaluated.

### Q9. Why does calibration matter here, and what did you find?

_A:_ The routing rule reads the confidence directly, so a miscalibrated confidence is a wrong
decision, not just a bad chart. The raw confidences were badly off in opposite directions: A
was under-confident, with ECE 0.404, and B-logprob over-confident, with ECE 0.244 — so on raw
confidence, B was actually the better-calibrated arm. After isotonic calibration A reached
0.007 and B 0.026; the difference, −0.019, had a 95% CI of [−0.030, −0.002] with p = 0.027,
an interval that narrowly excluded zero. It changed the costs a lot: A's 10:1 cost went from
970.1 raw to 589.0 calibrated, and raw B-logprob at 5, 10 and 20:1 cost more than sending
every ticket to a human.

### Q10. How do you know you didn't tune on the test set?

_A:_ The method was frozen in two commits before the test split was scored: the implementation,
then a protocol file listing the data checksums, the model and its digest, the exact commands,
the hypotheses and the reporting rules, including that a p-value shown as 0.000 is written
p < 0.001. It also banned smoke runs on test. Development used only the validation split —
that is where I found and fixed a floating-point bug at the routing threshold, before anything
was frozen. The test split was run once; when the LLM run was interrupted, I logged it and
resumed with the identical command, which the protocol allowed because every completed
response was reused byte-for-byte from the append-only cache. An integrity audit — row counts,
labels, cache prefix unchanged — passed 14 of 14 before the report was generated, and the
report files are committed with their SHA-256 hashes.

---

## 30-second pitch

A comparison of classical text representations done so that it can produce a negative
result: TF-IDF, character n-grams and LSA on 20 Newsgroups, with paired bootstrap
confidence intervals and McNemar tests, which show the dense representation does *not*
beat TF-IDF. Underneath it is a text-normalisation layer that gets Devanagari and Kannada
right — including the bug where Python's `\w` silently deletes every combining mark — and a
topic model whose labels went from "the, of, to" to "god, jesus, christ" once the c-TF-IDF
weighting was fixed, caught by tracking NPMI coherence rather than by reading word lists.

The second part is a cost-priced router study on Banking77, with the protocol committed
before the test split was scored once. On 3,080 held-out tickets at a 10:1
misroute-to-handoff ratio, a calibrated TF-IDF router cost 589 handoff-units per 1,000 tickets
against 939 for a calibrated local 7B LLM router — difference −350, 95% CI −397.7 to −299.3,
p < 0.001, excluding compute. It won by automating far more tickets, not by misrouting fewer,
and it is one LLM configuration, not a verdict on LLMs.
