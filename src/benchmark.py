from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unicodedata

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 dataset and reject malformed evaluation inputs."""
    conversations = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(conversations, list):
        raise ValueError(f"{path}: expected a list of conversations")
    for index, conversation in enumerate(conversations):
        where = f"{path}: conversation {index}"
        if not isinstance(conversation, dict):
            raise ValueError(f"{where}: expected an object")
        for key in ("id", "user_id"):
            if not isinstance(conversation.get(key), str) or not conversation[key].strip():
                raise ValueError(f"{where}: {key} must be a nonempty string")
        turns = conversation.get("turns")
        if not isinstance(turns, list) or not all(isinstance(turn, str) for turn in turns):
            raise ValueError(f"{where}: turns must be a list of strings")
        questions = conversation.get("recall_questions")
        if not isinstance(questions, list):
            raise ValueError(f"{where}: recall_questions must be a list")
        for question in questions:
            if not isinstance(question, dict) or not isinstance(question.get("question"), str):
                raise ValueError(f"{where}: each recall question needs question text")
            expected = question.get("expected_contains")
            if not isinstance(expected, list) or not expected or not all(
                isinstance(fact, str) and fact.strip() for fact in expected
            ):
                raise ValueError(f"{where}: expected_contains must contain nonempty strings")
    return conversations


def _fact_hits(answer: str, expected: list[str]) -> int:
    normalized = unicodedata.normalize("NFC", answer).casefold()
    return sum(bool(fact.strip()) and unicodedata.normalize("NFC", fact).casefold() in normalized
               for fact in expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """All facts = 1, some facts = 0.5, no facts (or no expectations) = 0."""
    hits = _fact_hits(answer, expected)
    if not hits:
        return 0.0
    return 1.0 if hits == len(expected) else 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Fact coverage in [0, 1], a proxy rather than an LLM quality judgment.

    Uses the same formula for both agents, with no bonus for length or style.
    Substring matching cannot detect contradictions or assess helpfulness.
    """
    return _fact_hits(answer, expected) / len(expected) if expected else 0.0


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Feed each conversation, then evaluate recall in a separate thread.

    Token and compaction columns count dataset conversation turns only;
    recall probing is excluded. Memory growth is final minus initial bytes,
    once per distinct user, not the sum of sizes after each conversation.
    Callers should supply a fresh agent and isolated state (as main does).
    """
    users = {conversation["user_id"] for conversation in conversations}
    memory_size = getattr(agent, "memory_file_size", lambda user_id: 0)
    initial_bytes = sum(memory_size(user_id) for user_id in users)
    tokens = prompts = compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    # Numbered IDs prevent duplicate dataset IDs/users from sharing history.
    for index, conversation in enumerate(conversations):
        thread_id = f"benchmark:conversation:{index}"
        recall_thread = f"benchmark:recall:{index}"
        user_id = conversation["user_id"]
        for message in conversation["turns"]:
            result = agent.reply(user_id, thread_id, message)
            tokens += result["token_usage"]
            prompts += result["prompt_tokens_processed"]
        compactions += agent.compaction_count(thread_id)
        for question in conversation["recall_questions"]:
            answer = agent.reply(user_id, recall_thread, question["question"])["response"]
            expected = question["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=tokens,
        prompt_tokens_processed=prompts,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=sum(memory_size(user_id) for user_id in users) - initial_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render the agent label and all six required metrics as Markdown."""
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
               "Response quality", "Memory growth (bytes)", "Compactions"]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] + ["---:"] * 6) + " |"]
    for row in rows:
        name = row.agent_name.replace("|", "\\|").replace("\n", " ")
        values = [name, str(row.agent_tokens_only), str(row.prompt_tokens_processed),
                  f"{row.recall_score:.2%}", f"{row.response_quality:.2%}",
                  str(row.memory_growth_bytes), str(row.compactions)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Run reproducible offline suites without reading existing user profiles."""
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [("Standard Benchmark", "conversations.json"),
              ("Long-Context Stress Benchmark", "advanced_long_context.json")]
    print("Mode: deterministic offline; token counts are heuristic estimates.")
    print("Tokens/compactions: conversation turns only (recall probes excluded).")
    print("Recall: mean 0/0.5/1 per question; quality: mean expected-fact coverage.")
    print("Memory growth: final minus initial profile bytes, counted once per user.")
    print(f"Compaction: threshold={config.compact_threshold_tokens} tokens, "
          f"keep={config.compact_keep_messages} messages.\n")
    for title, filename in suites:
        conversations = load_conversations(config.data_dir / filename)
        # A new directory and new agents for every suite and invocation prevent
        # later facts from leaking backwards into earlier recall questions.
        with TemporaryDirectory(prefix="benchmark-", dir=config.state_dir) as state:
            suite_config = replace(config, state_dir=Path(state))
            rows = [run_agent_benchmark(name, agent_type(suite_config, force_offline=True),
                                        conversations, suite_config)
                    for name, agent_type in [("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)]]
        print(f"## {title}\n")
        print(format_rows(rows))
        print()


if __name__ == "__main__":
    main()
