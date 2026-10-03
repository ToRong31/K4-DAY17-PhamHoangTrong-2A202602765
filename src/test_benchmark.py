from dataclasses import replace
import json
from pathlib import Path
import unicodedata

import pytest

import benchmark
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, load_conversations, recall_points, run_agent_benchmark
from config import load_config


@pytest.mark.parametrize('answer,expected,recall,quality', [
    ('unknown', ['name'], 0, 0),
    ('Alice', ['Alice', 'Hue', 'engineer'], 0.5, 1 / 3),
    ('ALICE Hue engineer', ['alice', 'Hue', 'engineer'], 1, 1),
    ('', [], 0, 0),
    (unicodedata.normalize('NFD', 'Huế'), ['Huế'], 1, 1),
])
def test_shared_scoring(answer, expected, recall, quality):
    assert recall_points(answer, expected) == recall
    assert heuristic_quality(answer, expected) == pytest.approx(quality)


def test_load_utf8_and_reject_invalid_expectations(tmp_path):
    path = tmp_path / 'dataset.json'
    data = [{'id': 'c', 'user_id': 'u', 'turns': ['Mình ở Huế.'],
             'recall_questions': [{'question': 'Ở đâu?', 'expected_contains': ['Huế']}]}]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8-sig')
    assert load_conversations(path) == data
    data[0]['recall_questions'][0]['expected_contains'] = []
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='expected_contains'):
        load_conversations(path)
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='list of conversations'):
        load_conversations(path)


class TrackingAgent:
    def __init__(self):
        self.calls = []
        self.sizes = {'u': 10, 'v': 20}

    def reply(self, user_id, thread_id, message):
        self.calls.append((user_id, thread_id, message))
        self.sizes[user_id] += 1
        probe = ':recall:' in thread_id
        return {'response': 'fact', 'token_usage': 1000 if probe else 2,
                'prompt_tokens_processed': 2000 if probe else 3}

    def compaction_count(self, thread_id):
        return 4

    def memory_file_size(self, user_id):
        return self.sizes[user_id]


def test_accounting_threads_and_same_input_for_both_agents():
    question = {'question': 'fact?', 'expected_contains': ['fact']}
    data = [{'id': 'duplicate', 'user_id': user, 'turns': ['one', 'two'],
             'recall_questions': [question, question]} for user in ['u', 'u', 'v']]
    first, second = TrackingAgent(), TrackingAgent()
    row = run_agent_benchmark('A', first, data, None)
    run_agent_benchmark('B', second, data, None)
    assert first.calls == second.calls
    training = {thread for _, thread, _ in first.calls if ':conversation:' in thread}
    probes = {thread for _, thread, _ in first.calls if ':recall:' in thread}
    assert len(training) == len(probes) == 3
    assert training.isdisjoint(probes)
    assert row.agent_tokens_only == 12
    assert row.prompt_tokens_processed == 18
    assert row.compactions == 12
    assert row.memory_growth_bytes == 12  # final minus initial; repeated u counted once
    assert row.recall_score == row.response_quality == 1
    empty = run_agent_benchmark('A', TrackingAgent(), [], None)
    assert empty.recall_score == empty.response_quality == empty.memory_growth_bytes == 0


def test_corrections_are_evaluated_before_later_conversations(tmp_path):
    config = replace(load_config(), state_dir=tmp_path)
    data = [{'id': 'c', 'user_id': 'u', 'turns': [f'Mình ở {place}.'],
             'recall_questions': [{'question': 'Mình ở đâu?', 'expected_contains': [place]}]}
            for place in ['Huế', 'Đà Nẵng']]
    baseline = run_agent_benchmark('Baseline', BaselineAgent(config, True), data, config)
    advanced = run_agent_benchmark('Advanced', AdvancedAgent(config, True), data, config)
    assert baseline.recall_score == 0
    assert baseline.compactions == baseline.memory_growth_bytes == 0
    assert advanced.recall_score == 1


@pytest.mark.parametrize('filename,stress', [
    ('conversations.json', False), ('advanced_long_context.json', True),
])
def test_real_datasets(tmp_path, filename, stress):
    config = replace(load_config(), state_dir=tmp_path,
                     compact_threshold_tokens=900, compact_keep_messages=6)
    data = load_conversations(config.data_dir / filename)
    baseline = run_agent_benchmark('Baseline', BaselineAgent(config, True), data, config)
    advanced = run_agent_benchmark('Advanced', AdvancedAgent(config, True), data, config)
    assert baseline.recall_score == 0
    assert advanced.recall_score == advanced.response_quality == 1
    assert advanced.memory_growth_bytes > 0
    if stress:
        assert advanced.compactions > 0
        assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed
    else:
        assert advanced.prompt_tokens_processed > baseline.prompt_tokens_processed


def test_main_repeatable_isolated_suites_and_all_columns(tmp_path, monkeypatch, capsys):
    config = replace(load_config(), state_dir=tmp_path,
                     compact_threshold_tokens=900, compact_keep_messages=6)
    sentinel = tmp_path / 'profiles' / 'existing' / 'User.md'
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('Keep my profile', encoding='utf-8')
    monkeypatch.setattr(benchmark, 'load_config', lambda *_: config)
    benchmark.main()
    first = capsys.readouterr().out
    benchmark.main()
    assert capsys.readouterr().out == first
    assert first.count('## Standard Benchmark') == 1
    assert first.count('## Long-Context Stress Benchmark') == 1
    for column in ['Agent tokens only', 'Prompt tokens processed', 'Cross-session recall',
                   'Response quality', 'Memory growth (bytes)', 'Compactions']:
        assert first.count(column) == 2
    assert first.count('| Baseline |') == first.count('| Advanced |') == 2
    assert sentinel.read_text(encoding='utf-8') == 'Keep my profile'
    assert not list(tmp_path.glob('benchmark-*'))
