from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, estimate_tokens


def make_config(tmp_path: Path):
    return replace(load_config(), state_dir=tmp_path / 'state',
                   compact_threshold_tokens=100, compact_keep_messages=2)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / 'profiles')
    assert store.read_text('u') == ''
    path = store.write_text('u', '# User\n- location: Huế\n')
    assert path.exists()
    assert store.edit_text('u', 'Huế', 'Đà Nẵng')
    assert 'Đà Nẵng' in store.read_text('u')
    assert not store.edit_text('u', 'missing', 'x')


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for _ in range(10):
        agent.reply('u', 'long', 'Nội dung dài. ' * 80)
    assert agent.compaction_count('long') > 0
    assert len(agent.compact_memory.context('long')['messages']) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for agent in (advanced, baseline):
        agent.reply('same-user', 'first', 'Mình tên là An. Mình ở Huế.')
        assert 'An' in agent.reply('same-user', 'first', 'Mình tên gì?')['response']
    assert 'An' in advanced.reply('same-user', 'new', 'Mình tên gì?')['response']
    assert 'An' not in baseline.reply('same-user', 'new', 'Mình tên gì?')['response']
    assert 'Huế' not in baseline.reply('same-user', 'new', 'Mình ở đâu?')['response']


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for _ in range(20):
        message = 'Nội dung dài về kỹ thuật. ' * 100
        assert baseline.reply('u', 'long', message)['response'] == advanced.reply('u', 'long', message)['response']
    assert advanced.prompt_token_usage('long') < baseline.prompt_token_usage('long')
    assert baseline.compaction_count('long') == 0
    assert len(baseline.sessions['long'].messages) == 40


def test_baseline_counters_accumulate_each_turn(tmp_path):
    agent = BaselineAgent(make_config(tmp_path), force_offline=True)
    assert agent.token_usage('absent') == agent.prompt_token_usage('absent') == 0
    assert agent.sessions == {}
    first = agent.reply('u', 't', 'Chào bạn')
    assert first['prompt_tokens_processed'] == estimate_tokens('Chào bạn')
    assert first['token_usage'] == estimate_tokens(first['response'])
    second = agent.reply('u', 't', 'Tiếp tục')
    expected = sum(estimate_tokens(x) for x in ('Chào bạn', first['response'], 'Tiếp tục'))
    assert second['prompt_tokens_processed'] == expected
    assert agent.token_usage('t') == first['token_usage'] + second['token_usage']
    assert agent.prompt_token_usage('t') == first['prompt_tokens_processed'] + expected
    other = agent.reply('u', 'other', 'Chào bạn')
    assert other == first
    assert agent.prompt_token_usage('t') == first['prompt_tokens_processed'] + expected


def test_baseline_never_reads_or_writes_profiles(tmp_path):
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / 'profiles')
    store.upsert_fact('u', 'name', 'SecretName')
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    agent = BaselineAgent(config, force_offline=True)
    assert 'SecretName' not in agent.reply('u', 't', 'Mình tên gì?')['response']
    agent.reply('u', 't', 'Mình tên là An.')
    agent.reply('u', 't', 'Mình ở Huế.')
    agent.reply('u', 't', 'Giờ mình đang ở Đà Nẵng chứ không còn ở Huế nữa.')
    answer = agent.reply('u', 't', 'Nơi ở hiện tại của mình là đâu?')['response']
    assert 'Đà Nẵng' in answer and 'Huế' not in answer
    after = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert before == after
    fresh = BaselineAgent(config, force_offline=True)
    assert 'An' not in fresh.reply('u', 't', 'Mình tên gì?')['response']


def test_baseline_live_routing_and_thread_history(tmp_path):
    calls = []

    class FakeModel:
        def invoke(self, messages):
            calls.append([dict(m) for m in messages])
            return SimpleNamespace(content=[{'type': 'text', 'text': 'Live answer'}])

    agent = BaselineAgent(make_config(tmp_path))
    agent.langchain_agent = FakeModel()
    first = agent.reply('u', 'a', 'First')
    second = agent.reply('u', 'a', 'Second')
    agent.reply('u', 'b', 'Fresh')
    assert first['response'] == 'Live answer'
    assert calls[1] == [{'role': 'user', 'content': 'First'},
                        {'role': 'assistant', 'content': 'Live answer'},
                        {'role': 'user', 'content': 'Second'}]
    assert calls[2] == [{'role': 'user', 'content': 'Fresh'}]
    assert agent.token_usage('a') == first['token_usage'] + second['token_usage']
    assert agent.prompt_token_usage('a') == first['prompt_tokens_processed'] + second['prompt_tokens_processed']
    assert agent.compaction_count('a') == 0
    agent.force_offline = True
    agent.reply('u', 'offline', 'Hello')
    assert len(calls) == 3


def test_baseline_live_failure_does_not_record_turn(tmp_path):
    class FailingModel:
        def invoke(self, messages):
            messages[0]['content'] = 'mutated'
            raise RuntimeError('Unavailable')

    agent = BaselineAgent(make_config(tmp_path))
    agent.reply('u', 't', 'Original')
    before = [dict(m) for m in agent.sessions['t'].messages]
    counters = (agent.token_usage('t'), agent.prompt_token_usage('t'))
    agent.langchain_agent = FailingModel()
    with pytest.raises(RuntimeError, match='Unavailable'):
        agent.reply('u', 't', 'Failed')
    assert agent.sessions['t'].messages == before
    assert (agent.token_usage('t'), agent.prompt_token_usage('t')) == counters


def test_baseline_optional_builder_uses_configured_provider(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    model = object()
    calls = []

    def build_model(provider_config):
        calls.append(provider_config)
        return model

    monkeypatch.setattr('agent_baseline.build_chat_model', build_model)
    assert BaselineAgent(config)._maybe_build_langchain_agent() is model
    assert calls == [config.model]
    assert BaselineAgent(config, force_offline=True)._maybe_build_langchain_agent() is None
    assert calls == [config.model]
