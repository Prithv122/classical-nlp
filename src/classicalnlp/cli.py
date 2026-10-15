"""Console entry point: compare, leakage, topics, normalize, tickets-download,
tickets-summary, route, route-report, jev-probe, jev-rtt."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from . import data, evaluate, partition, report, router, study, tickets, topics, vectorizers
from .normalize import detect_script, normalize, tokenize
from .router import decision_arm as dec
from .router.llm_arm import LLMClient, Provider, ResponseCache, llm_arms
from .router.ollama import DEFAULT_HOST, DEFAULT_MODEL, OllamaProvider
from .router.typesafe import TypeSafeProvider


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


def _make_provider(args: argparse.Namespace) -> Provider:
    """The model behind arm B. Constructing it makes no connection; tests replace this."""
    return OllamaProvider(args.model or DEFAULT_MODEL, args.host)


def _make_decision_provider(args: argparse.Namespace) -> dec.DecisionProvider:
    """The route behind arm C. Constructing it makes no connection and reads no credential;
    tests replace this."""
    return TypeSafeProvider()


def _abc_corpora(args: argparse.Namespace) -> tuple[data.Corpus, data.Corpus, data.Corpus]:
    """``(F', V, H)`` for the arm C partition; reads the train file only. Tests replace this."""
    return partition.abc_corpora(partition.load_train(args.data_dir))


def _check_route_args(args: argparse.Namespace) -> None:
    """Refuse bad combinations before any data is loaded or any model is called."""
    abc = args.partition == "abc"
    if args.arm == "C":
        if not abc:
            raise ValueError("arm C is evaluated on the abc partition only: add --partition abc")
        if not args.cache:
            raise ValueError("arm C needs --cache PATH: a paid call must never be repeated")
        if not args.ledger:
            raise ValueError("arm C needs --ledger PATH: every attempt is logged")
        if args.model is not None:
            raise ValueError("--model applies to arm B only")
    else:
        if args.arm == "B":
            if not args.cache:
                raise ValueError("arm B needs --cache PATH: a slow run must never be repeated")
        elif args.model is not None or args.cache is not None:
            raise ValueError("--model and --cache apply to arm B only")
        if args.ledger:
            raise ValueError("--ledger applies to arm C only")
    if abc:
        if args.split == "test":
            raise ValueError("the official test split is not read by this study")
        if args.fixture:
            raise ValueError("--partition abc needs the real Banking77 train file, not the fixture")
    elif args.split == "heldout":
        raise ValueError("--split heldout needs --partition abc")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        if args.split == "test":
            raise ValueError(
                "--limit is for smoke runs on the validation split; the test split is run whole"
            )
        if args.split == "heldout":
            raise ValueError("--limit is for smoke runs on the validation split; H is run whole")


def _require_v_cached(client: dec.DecisionClient, val: data.Corpus, path: str) -> None:
    """A heldout run must make no V call: every V request is already in the cache."""
    identity = client.provider.identity
    missing = sum(
        client.cache.get(dec.cache_key(identity, dec.build_request(text))) is None
        for text in val.documents
    )
    if missing:
        raise ValueError(
            f"a heldout run makes no V call: {missing} of {len(val)} V requests are not in "
            f"the cache {path}; finish the --split val run first"
        )


def _open_ledger(args: argparse.Namespace) -> dec.Ledger:
    """The study's one spend ledger. A path that does not exist is refused unless
    ``--new-ledger`` says this is the study's first ledger, so a mistyped path can never
    restart the study total at $0; ``--new-ledger`` on an existing file is refused too."""
    path = Path(args.ledger)
    if args.new_ledger and path.exists():
        raise ValueError(f"--new-ledger given but {path} already exists; drop the flag")
    if not args.new_ledger and not path.exists():
        raise ValueError(
            f"ledger {path} does not exist: every run must use the study's one ledger "
            "(pass --new-ledger only for the study's first run)"
        )
    ledger = dec.Ledger(path)
    print(
        f"ledger {path}: study total so far ${ledger.study_total:.4f} of ${dec.STUDY_CAP_USD:.2f}"
    )
    return ledger


def cmd_route(args: argparse.Namespace) -> int:
    _check_route_args(args)
    test = None
    split_name = "test"
    if args.partition == "abc":
        fit, val, heldout = _abc_corpora(args)
        print(
            f"source: Banking77 train split ({tickets.BANKING77_LICENCE}), {args.data_dir}; "
            "abc partition (fit F', validation V, held-out H)"
        )
        if args.split == "heldout":
            test, split_name = heldout, "heldout"
    elif args.fixture:
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
    if args.limit is not None:
        val = data.Corpus(
            documents=val.documents[: args.limit],
            labels=val.labels[: args.limit],
            label_names=val.label_names,
        )
        print(f"LIMITED to the first {args.limit} validation rows: a smoke run, not a result")

    client = cache = None
    if args.arm == "B":
        cache = ResponseCache(args.cache)
        client = LLMClient(_make_provider(args), cache)
        runs = []
        for arm in llm_arms(client):
            runs += study.run_arm(arm, fit, val, test, split_name=split_name)
        label = "B (variants B-logprob, B-verbal)"
    elif args.arm == "C":
        cache, ledger = dec.DecisionCache(args.cache), _open_ledger(args)
        client = dec.DecisionClient(
            _make_decision_provider(args), cache, ledger, price_per_mtok=args.price_per_mtok
        )
        if args.split == "heldout":
            _require_v_cached(client, val, args.cache)
        runs = []
        for arm in dec.decision_arms(client):
            runs += study.run_arm(arm, fit, val, test, split_name=split_name)
        label = "C (variants C-prob, C-vendor)"
    else:
        arm = router.ARMS[args.arm]()
        runs = study.run_arm(arm, fit, val, test, split_name=split_name)
        label = arm.name
    evaluated = test if test is not None else val
    print(
        f"arm {label}: {router.ARMS[args.arm].description}; split {args.split}; "
        f"fit {len(fit)} rows, evaluated {len(evaluated)} rows"
    )
    print(study.format_runs(runs))
    print(study.format_usage(runs))
    if args.arm == "B":
        print(
            f"llm calls: {client.calls} made, {client.hits} reused from the cache or the "
            f"other variant (cache: {args.cache})"
        )
        if cache.skipped > 0:
            print(f"cache: {cache.skipped} lines skipped")
        print(
            "latency is the model-side time Ollama reports, load time excluded, "
            "as recorded when each response was first generated"
        )
    if args.arm == "C":
        print(
            f"jev requests: {client.calls} sent ({client.attempts} attempts), {client.hits} "
            f"reused from the cache or the other variant (cache: {args.cache})"
        )
        if cache.skipped > 0:
            print(f"cache: {cache.skipped} lines skipped")
        print(
            f"transport failures: {client.transport_failures}; spend this run "
            f"${ledger.run_total:.4f}, study total ${ledger.study_total:.4f} of "
            f"${dec.STUDY_CAP_USD:.2f} (ledger: {args.ledger})"
        )
        print(
            "latency is client-side wall-clock of the successful attempt, "
            "as recorded when each response was first fetched"
        )
    if args.save:
        print(f"saved {study.save_runs(runs, args.save)} rows to {args.save}")
    return 0


def cmd_route_report(args: argparse.Namespace) -> int:
    if args.iterations < 100:
        raise ValueError("--iterations must be at least 100")
    runs = report.load_runs(args.runs, final=args.final)
    written = report.write_report(
        runs,
        args.out,
        source_note=args.source_note,
        iterations=args.iterations,
        plots=not args.no_plots,
        price_per_mtok=args.price_per_mtok,
    )
    print(f"wrote {len(written)} files to {args.out}")
    for path in written:
        print(path.name)
    return 0


def probe_positions(n: int, k: int = 20) -> np.ndarray:
    """Positions of the probe rows in V: the protocol's seeded draw."""
    return np.random.default_rng(partition.ABC_SEED).choice(n, k, replace=False)


def _log_response(path: str, kind: str, key: str, r: dec.DecisionResponse) -> None:
    record = {"kind": kind, "key": key, "response": r.body, "model_version": r.model_version}
    record |= {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens}
    record |= {"seconds": r.seconds, "cold_start": r.cold_start, "attempts": r.attempts}
    record["transport_failure"] = r.transport_failure
    dec.append_line(path, json.dumps(record))


def cmd_jev_probe(args: argparse.Namespace) -> int:
    """G2 tool: 20 probe rows of V into the cache, then the same 20 again for self-agreement."""
    ledger = _open_ledger(args)  # first, so a wrong ledger path fails before any data loads
    _, val, _ = _abc_corpora(args)
    requests = [dec.build_request(val.documents[i]) for i in probe_positions(len(val))]
    client = dec.DecisionClient(
        _make_decision_provider(args),
        dec.DecisionCache(args.cache),
        ledger,
        price_per_mtok=args.price_per_mtok,
        kind="probe",
    )
    first = [client.decide(request) for request in requests]
    second = []
    for request in requests:
        r = client.decide(request, kind="self_agreement", use_cache=False)
        _log_response(
            args.log, "self_agreement", dec.cache_key(client.provider.identity, request), r
        )
        second.append(r)

    summary = probe_summary(first, second, list(tickets.BANKING77_INTENTS))
    n = len(requests)
    ok = [r for r in first if not r.transport_failure]
    print(f"reported version: {', '.join(summary['versions']) or 'none'}")
    schema = summary["schema"]
    print(f"schema check: {schema[0]}/{n} first pass, {schema[1]}/{n} second pass")
    if ok:
        print(f"median input_tokens: {statistics.median(r.input_tokens for r in ok)}")
        print(f"p50 latency: {statistics.median(r.seconds for r in ok):.3f} s")
    # Self-agreement is about the model, so it reads every pair; schema validity is the line above.
    print(
        f"same choice: {summary['same']}/{n}; "
        f"largest absolute probability difference: {summary['largest']:.6f}"
    )
    if summary["compared"] < n:
        print(f"probability difference read on {summary['compared']}/{n} pairs")
    ledger = client.ledger
    print(f"spend this run ${ledger.run_total:.4f}; study total ${ledger.study_total:.4f}")
    passed = summary["versions"] == [dec.REQUESTED_MODEL] and schema == [n, n]
    print("G2 checks: " + ("pass" if passed else "FAIL"))
    return 0 if passed else 1


def _readable(response, names):
    """``(choice, probabilities)`` as sent, whatever the sum, or None if either is unreadable."""
    intent = ((response.body or {}).get("answers") or {}).get("intent")
    if not isinstance(intent, dict) or not isinstance(intent.get("choice"), str):
        return None
    probs = intent.get("probabilities")
    numbers = isinstance(probs, dict) and all(
        isinstance(probs.get(n), int | float) and not isinstance(probs.get(n), bool) for n in names
    )
    return intent["choice"], (probs if numbers else None)


def probe_summary(first, second, names):
    """The G2 figures from the probe's two passes over the same requests.

    ``schema`` counts the responses that pass the arm's usability check, per pass. ``same`` and
    ``largest`` read every pair (a pair with an unreadable choice is not the same choice), and
    ``largest`` only pairs whose probabilities both hold a number for each of the 77 names.
    """
    ok = [r for r in first if not r.transport_failure]
    seen = [(_readable(a, names), _readable(b, names)) for a, b in zip(first, second, strict=True)]
    same = sum(a is not None and b is not None and a[0] == b[0] for a, b in seen)
    both = [(a[1], b[1]) for a, b in seen if a and b and a[1] is not None and b[1] is not None]

    def usable(responses):
        return sum(dec.parse_answer(r.body, names)[0] is not None for r in responses)

    return {
        "versions": sorted({r.model_version for r in ok + second if r.model_version}),
        "schema": [usable(first), usable(second)],
        "same": same,
        "compared": len(both),
        "largest": max((abs(a[n] - b[n]) for a, b in both for n in names), default=0.0),
    }


def cmd_jev_rtt(args: argparse.Namespace) -> int:
    """Baseline round trip: n unbilled GET /v1/models requests on one persistent connection."""
    if args.n < 1:
        raise ValueError("--n must be at least 1")
    provider, ledger, times = _make_decision_provider(args), _open_ledger(args), []
    for attempt in range(1, args.n + 1):
        try:
            status, seconds = provider.ping()
        except dec.AttemptFailed as failure:
            status, seconds = failure.status or failure.error, None
        ledger.add("rtt", None, attempt, status, None, None, 0.0)
        dec.append_line(
            args.log, json.dumps({"attempt": attempt, "status": status, "seconds": seconds})
        )
        if status in (401, 403):
            raise dec.AuthError(f"HTTP {status}: check TYPESAFE_API_KEY")
        if seconds is not None:
            times.append(seconds)
    if not times:
        print("no request completed", file=sys.stderr)
        return 1
    median = statistics.median(times)
    print(f"median round trip over {len(times)} of {args.n} requests: {median:.4f} s")
    print("the first request includes opening the connection")
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
    p_route.add_argument("--split", choices=("val", "test", "heldout"), default="val")
    p_route.add_argument(
        "--partition",
        choices=("v2", "abc"),
        default="v2",
        help="v2: the shipped study; abc: the arm C partition (fit F', validation V, held-out H)",
    )
    route_source = p_route.add_mutually_exclusive_group(required=True)
    route_source.add_argument("--fixture", action="store_true", help="The synthetic 60-row fixture")
    route_source.add_argument("--data-dir", help="Directory holding the downloaded Banking77 CSVs")
    p_route.add_argument("--model", default=None, help=f"Arm B model (default {DEFAULT_MODEL})")
    p_route.add_argument("--host", default=DEFAULT_HOST, help="Arm B Ollama server address")
    p_route.add_argument("--cache", metavar="PATH", help="Arm B response cache (JSONL, required)")
    p_route.add_argument("--ledger", metavar="PATH", help="Arm C spend ledger (JSONL, required)")
    p_route.add_argument(
        "--new-ledger",
        action="store_true",
        help="Create the ledger: only for the study's first paid or logged run",
    )
    p_route.add_argument(
        "--price-per-mtok",
        type=float,
        default=dec.DEFAULT_PRICE_PER_MTOK,
        help="Arm C price, USD per million input tokens",
    )
    p_route.add_argument(
        "--limit", type=int, default=None, help="Smoke run: first N validation rows only"
    )
    p_route.add_argument("--save", metavar="PATH", help="Write per-ticket results (JSONL)")
    p_route.set_defaults(func=cmd_route)

    p_report = sub.add_parser(
        "route-report", help="Tables, comparisons and a report from route --save files"
    )
    p_report.add_argument(
        "--runs", nargs="+", required=True, metavar="PATH", help="route --save files"
    )
    p_report.add_argument("--out", required=True, metavar="DIR", help="Directory to write into")
    p_report.add_argument("--source-note", default=None, help="Where the data came from")
    p_report.add_argument(
        "--iterations", type=int, default=2000, help="Bootstrap resamples (>= 100)"
    )
    p_report.add_argument("--no-plots", action="store_true", help="Skip the reliability diagram")
    p_report.add_argument(
        "--final", action="store_true", help="Allow runs on the official test split"
    )
    p_report.add_argument(
        "--price-per-mtok",
        type=float,
        default=dec.DEFAULT_PRICE_PER_MTOK,
        help="Arm C price the runs were made at, USD per million input tokens",
    )
    p_report.set_defaults(func=cmd_route_report)

    p_probe = sub.add_parser("jev-probe", help="Arm C probe: 20 rows of V, then self-agreement")
    p_probe.add_argument("--data-dir", required=True)
    for flag in ("--cache", "--ledger", "--log"):
        p_probe.add_argument(flag, required=True, metavar="PATH")
    p_probe.add_argument("--price-per-mtok", type=float, default=dec.DEFAULT_PRICE_PER_MTOK)
    p_probe.add_argument(
        "--new-ledger",
        action="store_true",
        help="Create the ledger: only for the study's first paid or logged run",
    )
    p_probe.set_defaults(func=cmd_jev_probe)

    p_rtt = sub.add_parser("jev-rtt", help="Baseline round trip to the arm C host (unbilled)")
    for flag in ("--ledger", "--log"):
        p_rtt.add_argument(flag, required=True, metavar="PATH")
    p_rtt.add_argument("--n", type=int, default=20)
    p_rtt.add_argument(
        "--new-ledger",
        action="store_true",
        help="Create the ledger: only for the study's first paid or logged run",
    )
    p_rtt.set_defaults(func=cmd_jev_rtt)

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
