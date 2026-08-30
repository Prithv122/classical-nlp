"""Console entry point: compare, leakage, topics, normalize."""

from __future__ import annotations

import argparse
import sys

from . import data, evaluate, topics, vectorizers
from .normalize import detect_script, normalize, tokenize


def _stdout_utf8() -> None:
    """Required here, not optional: this project prints Devanagari and Kannada."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _corpus(args: argparse.Namespace) -> data.Corpus:
    corpus = data.load_newsgroups(subset=args.subset)
    if args.limit:
        corpus = data.Corpus(
            documents=corpus.documents[: args.limit],
            labels=corpus.labels[: args.limit],
            label_names=corpus.label_names,
        )
    print(f"{len(corpus)} documents, classes: {corpus.class_counts}\n", file=sys.stderr)
    return corpus


def cmd_compare(args: argparse.Namespace) -> int:
    corpus = _corpus(args)
    names = args.models or list(vectorizers.OFFLINE_MODELS)

    results = []
    for name in names:
        result = evaluate.evaluate(
            name,
            vectorizers.build(name),
            corpus.documents,
            corpus.labels,
            corpus.label_names,
            n_splits=args.folds,
        )
        results.append(result)
        print(
            f"{name:<22} macro-F1 {result.macro_f1:.4f}  (fold sd {result.fold_std:.4f})"
            f"  accuracy {result.accuracy:.4f}"
        )
        for label, score in result.per_class_f1.items():
            print(f"    {label:<26} F1 {score:.3f}")

    if len(results) >= 2:
        best, runner_up = sorted(results, key=lambda r: -r.macro_f1)[:2]
        stats = evaluate.paired_bootstrap(corpus.labels, best.predictions, runner_up.predictions)
        test = evaluate.mcnemar(corpus.labels, best.predictions, runner_up.predictions)
        print(
            f"\n{best.name} - {runner_up.name}: {stats['difference']:+.4f} macro-F1"
            f"  95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]"
            f"  bootstrap p={stats['p_value']:.3f}"
        )
        print(
            f"McNemar: {best.name} alone correct on {test['only_a']} items,"
            f" {runner_up.name} alone on {test['only_b']}, p={test['p_value']:.3g}"
        )
    if args.detail:
        print("\n" + evaluate.report(results[0], corpus.label_names, corpus.labels))
    return 0


def cmd_leakage(args: argparse.Namespace) -> int:
    corpus = _corpus(args)

    leaky, honest = evaluate.leaky_vs_honest(
        corpus.documents, corpus.labels, vectorizers.tfidf_word(), n_splits=args.folds
    )
    print("unsupervised step (TF-IDF fitted outside the folds):")
    print(f"  leaky  macro-F1 {leaky:.4f}")
    print(f"  honest macro-F1 {honest:.4f}")
    print(f"  inflation       {leaky - honest:+.4f}")

    if args.supervised:
        leaky_s, honest_s = evaluate.leaky_supervised_selection(
            corpus.documents, corpus.labels, k=args.k, n_splits=args.folds
        )
        print(f"\nsupervised step (chi-squared top-{args.k} selected outside the folds):")
        print(f"  leaky  macro-F1 {leaky_s:.4f}")
        print(f"  honest macro-F1 {honest_s:.4f}")
        print(f"  inflation       {leaky_s - honest_s:+.4f}")
    return 0


def cmd_topics(args: argparse.Namespace) -> int:
    corpus = _corpus(args)
    model = topics.fit_topics(corpus.documents, n_topics=args.n_topics, language="en")
    print(f"{len(model)} topics, NPMI coherence {model.coherence:.4f}\n")
    for topic in model.topics:
        print(f"  topic {topic.index:>2} ({topic.size:>4} docs): {topic.label(6)}")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    samples = [(args.text, args.language)] if args.text else list(data.MULTILINGUAL_SAMPLES)
    for text, language in samples:
        print(f"raw        : {text}")
        print(f"script     : {detect_script(text)}")
        print(f"normalized : {normalize(text)}")
        print(f"tokens     : {tokenize(text, language=language)}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="classical-nlp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_corpus_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--subset", choices=("train", "test", "all"), default="train")
        p.add_argument("--limit", type=int, default=None, help="Use only the first N documents")
        p.add_argument("--folds", type=int, default=5)

    p_compare = sub.add_parser("compare", help="Cross-validated comparison of representations")
    p_compare.add_argument(
        "--models", nargs="+", choices=sorted(vectorizers.BUILDERS), default=None
    )
    p_compare.add_argument("--detail", action="store_true", help="Per-class report and confusion")
    add_corpus_args(p_compare)
    p_compare.set_defaults(func=cmd_compare)

    p_leak = sub.add_parser("leakage", help="What fitting a step outside the folds buys you")
    p_leak.add_argument(
        "--supervised",
        action="store_true",
        help="Also measure chi-squared feature selection fitted on all labels",
    )
    p_leak.add_argument("--k", type=int, default=2000, help="Features kept by the selector")
    add_corpus_args(p_leak)
    p_leak.set_defaults(func=cmd_leakage)

    p_topics = sub.add_parser("topics", help="NMF + c-TF-IDF topics with NPMI coherence")
    p_topics.add_argument("--n-topics", type=int, default=8)
    add_corpus_args(p_topics)
    p_topics.set_defaults(func=cmd_topics)

    p_norm = sub.add_parser("normalize", help="Show normalisation and tokenisation")
    p_norm.add_argument("--text", default=None)
    p_norm.add_argument("--language", default=None, help="Stopword list to apply, e.g. hi, kn")
    p_norm.set_defaults(func=cmd_normalize)

    return parser


def main(argv: list[str] | None = None) -> int:
    _stdout_utf8()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
