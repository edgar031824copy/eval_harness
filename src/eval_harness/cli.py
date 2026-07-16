import argparse
import subprocess
import sys

from eval_harness.anthropic_client import AnthropicClient
from eval_harness.baseline_store import BaselineStore
from eval_harness.cost_tracker import CostTracker
from eval_harness.embedding_filter import EmbeddingFilter
from eval_harness.golden_set import load_golden_set
from eval_harness.judge import judge_pair, JudgeVerdict
from eval_harness.prompt_registry import load_registry, find_entry_for_changed_file
from eval_harness.replay import replay_prompt
from eval_harness.report import build_report, render_markdown, RunReport
from eval_harness.sampler import sample_examples
from eval_harness.stats_gate import evaluate_gate


def run_eval(
    prompt_file: str,
    old_prompt_text: str,
    new_prompt_text: str,
    registry_path: str = "prompt_registry.yaml",
    baseline_path: str = "baseline_store.json",
    full_run: bool = False,
    seed: int = 0,
    api_key: str | None = None,
) -> RunReport:
    registry = load_registry(registry_path)
    entry = find_entry_for_changed_file(registry, prompt_file)
    if entry is None:
        raise ValueError(f"{prompt_file} is not registered in {registry_path} — add it before it can be gated")

    golden_set = load_golden_set(entry.golden_set_file)
    examples = sample_examples(golden_set.examples, entry.sample_size, seed=seed, full_run=full_run)

    client = AnthropicClient(api_key=api_key)
    embedding_filter = EmbeddingFilter()
    cost_tracker = CostTracker()

    old_results = replay_prompt(client, old_prompt_text, examples)
    new_results = replay_prompt(client, new_prompt_text, examples)

    for r in old_results + new_results:
        cost_tracker.add_replay_call(model="claude-sonnet-5", input_tokens=r.input_tokens, output_tokens=r.output_tokens)

    verdicts: list[JudgeVerdict] = []

    for example, old_r, new_r in zip(examples, old_results, new_results):
        cost_tracker.add_embedding_call()
        if embedding_filter.is_likely_unchanged(old_r.output, new_r.output):
            verdicts.append(JudgeVerdict(score=1.0, reasoning="embedding pre-filter: outputs near-identical, judge skipped", input_tokens=0, output_tokens=0))
            continue
        verdict = judge_pair(client, example.input, example.expected, old_r.output, new_r.output, model=entry.judge_model)
        cost_tracker.add_judge_call(model=entry.judge_model, input_tokens=verdict.input_tokens, output_tokens=verdict.output_tokens)
        verdicts.append(verdict)

    baseline_store = BaselineStore()
    baseline_scores = baseline_store.load(baseline_path).get(golden_set.name, [])
    new_scores = [v.score for v in verdicts]
    gate_decision = evaluate_gate(new_scores, baseline_scores)

    mean_score = sum(new_scores) / len(new_scores) if new_scores else 0.0
    baseline_store.append_run(baseline_path, golden_set.name, mean_score)

    return build_report(
        prompt_name=golden_set.name,
        example_ids=[e.id for e in examples],
        verdicts=verdicts,
        gate_decision=gate_decision,
        cost_breakdown=cost_tracker.breakdown(),
    )


def _read_prompt_at_ref(prompt_file: str, git_ref: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{git_ref}:{prompt_file}"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def main():
    parser = argparse.ArgumentParser(prog="eval-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Replay a prompt change against its golden set")
    run_parser.add_argument("--prompt-file", required=True)
    run_parser.add_argument("--old-ref", default="origin/main", help="git ref to read the OLD prompt version from")
    run_parser.add_argument("--registry-path", default="prompt_registry.yaml")
    run_parser.add_argument("--baseline-path", default="baseline_store.json")
    run_parser.add_argument("--full-run", action="store_true")
    run_parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "run":
        old_text = _read_prompt_at_ref(args.prompt_file, args.old_ref)
        with open(args.prompt_file) as f:
            new_text = f.read()

        import os
        report = run_eval(
            prompt_file=args.prompt_file,
            old_prompt_text=old_text,
            new_prompt_text=new_text,
            registry_path=args.registry_path,
            baseline_path=args.baseline_path,
            full_run=args.full_run,
            seed=args.seed,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        print(render_markdown(report))
        sys.exit(1 if report.gate_decision.is_regression else 0)


if __name__ == "__main__":
    main()
