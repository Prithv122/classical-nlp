"""Console entry point: compare, leakage, topics, normalize, tickets-download,
tickets-summary, route."""

from __future__ import annotations

import argparse
import sys

from . import data, evaluate, router, study, tickets, topics, vectorizers
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


def cmd_tickets_download(args: argparse.Namespace) -> int:
    paths = tickets.download_banking77(args.dest)
    for split, path in paths.items():
        print(f"{split:<5} {path}  sha256 {tickets.sha256_file(path)}")
    print(f"Banking77 ({tickets.BANKING77_LICENCE}), source: {tickets.BANKING77_SOURCE}")
    return 0


def cmd_tickets_summary(args: argparse.Namespace) -> int:
    if args.fixture:
        corpus = tickets.load_fixture()
        print("source: synthetic fixture (hand-written, not Banking77)")
    else:
        # Train split only: nothing computed on the official test split is shown here.
        corpus = tickets.load_banking77(args.data_dir)[0]
        print(f"source: Banking77 train split ({tickets.BANKING77_LICENCE}), {args.data_dir}")
    fit, val = tickets.validation_split(corpus)

    def smallest(c: data.Corpus) -> int:
        return min(count for count in c.class_counts.values() if count)

    print(f"rows: {len(corpus)}")
    print(f"intents present: {sum(1 for n in corpus.class_counts.values() if n)}")
    print(f"fit/val: {len(fit)}/{len(val)}")
    print(f"smallest per-intent count: fit {smallest(fit)}, val {smallest(val)}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    test = None
    if args.fixture:
        if args.split == "test":
            raise ValueError("the fixture has no held-out test split; use --split val")
        fit, val = tickets.validation_split(tickets.load_fixture())
        print(
            "source: synthetic fixture (hand-written, not Banking77); "
            "these numbers test the code and are not results"
        )
    else:
        train, official_test = tickets.load_banking77(args.data_dir)
        fit, val = tickets.validation_split(train)
        print(f"source: Banking77 ({tickets.BANKING77_LICENCE}), {args.data_dir}")
        if args.split == "test":
            test = official_test
            print("official test split: run once, after the arms and calibration are final")

    arm = router.ARMS[args.arm]()
    runs = study.run_arm(arm, fit, val, test)
    evaluated = test if test is not None else val
    print(
        f"arm {arm.name}: {arm.description}; split {args.split}; "
        f"fit {len(fit)} rows, evaluated {len(evaluated)} rows"
    )
    print(study.format_runs(runs))
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

    p_dl = sub.add_parser("tickets-download", help="Fetch the pinned Banking77 CSVs (checksummed)")
    p_dl.add_argument("--dest", default=str(tickets.DEFAULT_DIRECTORY))
    p_dl.set_defaults(func=cmd_tickets_download)

    p_ts = sub.add_parser("tickets-summary", help="Sizes and class counts of a ticket corpus")
    source = p_ts.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", action="store_true", help="The synthetic 60-row fixture")
    source.add_argument("--data-dir", help="Directory holding the downloaded Banking77 CSVs")
    p_ts.set_defaults(func=cmd_tickets_summary)

    p_route = sub.add_parser("route", help="Calibrate a router arm and price its decisions")
    p_route.add_argument("--arm", choices=sorted(router.ARMS), required=True)
    p_route.add_argument("--split", choices=("val", "test"), default="val")
    route_source = p_route.add_mutually_exclusive_group(required=True)
    route_source.add_argument("--fixture", action="store_true", help="The synthetic 60-row fixture")
    route_source.add_argument("--data-dir", help="Directory holding the downloaded Banking77 CSVs")
    p_route.set_defaults(func=cmd_route)

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
