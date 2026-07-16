from eval_harness.prompt_registry import load_registry, find_entry_for_changed_file, PromptEntry

REGISTRY_YAML = """
prompts:
  - prompt_file: demo_agent/faq_agent_prompt.md
    golden_set_file: golden_sets/faq-demo-agent.yaml
    sample_size: 5
    judge_model: claude-opus-4-8
  - prompt_file: .claude/skills/jira-ticket-kickoff/SKILL.md
    golden_set_file: golden_sets/jira-ticket-kickoff.yaml
    sample_size: 3
    judge_model: claude-opus-4-8
"""


def test_load_registry_parses_all_entries(tmp_path):
    p = tmp_path / "registry.yaml"
    p.write_text(REGISTRY_YAML)

    registry = load_registry(str(p))

    assert len(registry) == 2
    assert registry[0].prompt_file == "demo_agent/faq_agent_prompt.md"
    assert registry[0].sample_size == 5


def test_find_entry_for_changed_file_matches():
    registry = [
        PromptEntry("demo_agent/faq_agent_prompt.md", "golden_sets/faq-demo-agent.yaml", 5, "claude-opus-4-8"),
    ]

    entry = find_entry_for_changed_file(registry, "demo_agent/faq_agent_prompt.md")

    assert entry is not None
    assert entry.golden_set_file == "golden_sets/faq-demo-agent.yaml"


def test_find_entry_for_changed_file_returns_none_when_unregistered():
    registry = [
        PromptEntry("demo_agent/faq_agent_prompt.md", "golden_sets/faq-demo-agent.yaml", 5, "claude-opus-4-8"),
    ]

    entry = find_entry_for_changed_file(registry, "some/other/file.py")

    assert entry is None
