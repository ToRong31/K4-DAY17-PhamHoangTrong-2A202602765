from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_advanced import AdvancedAgent
from config import load_config
from memory_store import estimate_tokens


def make_agent(tmp_path, threshold=10000, keep=2, offline=True):
    config = replace(load_config(), state_dir=tmp_path,
                     compact_threshold_tokens=threshold, compact_keep_messages=keep)
    return AdvancedAgent(config, force_offline=offline)


def test_offline_turn_accounts_all_memory_and_only_generated_output(tmp_path):
    agent = make_agent(tmp_path)
    first = agent.reply('u', 't', 'Mình tên là An. Mình ở Huế.')
    profile_tokens = estimate_tokens(agent.profile_store.read_text('u'))
    assert first['prompt_tokens_processed'] == profile_tokens + estimate_tokens('Mình tên là An. Mình ở Huế.')
    assert first['token_usage'] == estimate_tokens(first['response'])
    question = 'Mình tên gì?'
    old_messages = agent.compact_memory.context('t')['messages']
    second = agent.reply('u', 't', question)
    expected_prompt = profile_tokens + sum(estimate_tokens(m['content']) for m in old_messages) + estimate_tokens(question)
    assert second['response'] == 'An'
    assert second['prompt_tokens_processed'] == expected_prompt
    assert agent.token_usage('t') == first['token_usage'] + second['token_usage']
    assert agent.prompt_token_usage('t') == first['prompt_tokens_processed'] + second['prompt_tokens_processed']
    assert agent.token_usage('unknown') == agent.prompt_token_usage('unknown') == 0
    assert agent.compaction_count('unknown') == 0
    assert agent.compact_memory.context('new')['messages'] == []


def test_estimator_includes_nonempty_profile_summary_and_recent(tmp_path):
    agent = make_agent(tmp_path, threshold=100)
    agent.profile_store.upsert_fact('u', 'name', 'An')
    for index in range(6):
        agent.compact_memory.append('t', 'user', f'Topic {index}: ' + 'x' * 600)
    context = agent.compact_memory.context('t')
    profile = estimate_tokens(agent.profile_store.read_text('u'))
    summary = estimate_tokens(context['summary'])
    recent = sum(estimate_tokens(item['content']) for item in context['messages'])
    assert profile > 0 and summary > 0 and recent > 0
    assert agent._estimate_prompt_context_tokens('u', 't') == profile + summary + recent


def test_corrections_survive_compaction_and_restart(tmp_path):
    agent = make_agent(tmp_path, threshold=50)
    agent.reply('u', 't', 'Mình tên là An. Mình ở Huế và đang làm backend engineer.')
    agent.reply('u', 't', 'Mình đang ở Đà Nẵng chứ không còn ở Huế nữa.')
    agent.reply('u', 't', 'Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.')
    for index in range(20):
        agent.reply('u', 't', f'Topic {index}: ' + 'x' * 600)
    profile = agent.profile_store.read_text('u')
    assert profile.count('- location:') == profile.count('- profession:') == 1
    assert 'Huế' not in profile and 'backend engineer' not in profile
    assert agent.memory_file_size('u') > 0 and agent.compaction_count('t') > 0
    assert 'An' not in agent.compact_memory.context('t')['summary']
    reloaded = make_agent(tmp_path)
    answer = reloaded.reply('u', 'fresh', 'Nhắc lại tên, nghề và nơi ở hiện tại của mình?')['response']
    assert all(value in answer for value in ('An', 'MLOps engineer', 'Đà Nẵng'))
    assert reloaded.profile_store.read_text('u') == profile
    assert reloaded.compaction_count('fresh') == 0


def test_style_keeps_tradeoff_and_generates_three_bullets(tmp_path):
    agent = make_agent(tmp_path)
    agent.reply('u', 't', 'Mình muốn bạn trả lời ngắn gọn thành 3 bullet, ưu tiên trade-off.')
    agent.reply('u', 't', 'Mình muốn bạn trả lời ngắn gọn theo 3 bullet có ví dụ thực chiến.')
    style = agent.profile_store.facts('u')['response_style']
    assert 'trade-off' in style and '3 bullet' in style and 'ví dụ thực chiến' in style
    answer = make_agent(tmp_path).reply('u', 'fresh', 'Nhắc lại style trả lời mình thích')['response']
    assert len(answer.splitlines()) == 3
    assert all(line.startswith('- ') for line in answer.splitlines())
    assert '3 bullet' in answer and 'Trade-off' in answer


def test_advanced_does_not_leak_between_users(tmp_path):
    agent = make_agent(tmp_path)
    agent.reply('one', 't1', 'Mình tên là SecretName.')
    assert 'SecretName' not in agent.reply('two', 't2', 'Mình tên gì?')['response']
    before = agent.compact_memory.context('t1')
    with pytest.raises(ValueError, match='single user_id'):
        agent.reply('two', 't1', 'Mình tên là WrongName.')
    assert agent.compact_memory.context('t1') == before
    assert agent.profile_store.read_text('two') == ''


def test_live_prompt_uses_profile_summary_recent_and_counts_actual_payload(tmp_path):
    agent = make_agent(tmp_path, threshold=80, offline=False)
    agent.reply('u', 't', 'Mình tên là An.')
    for index in range(5):
        agent.reply('u', 't', f'Topic {index}: ' + 'x' * 600)
    calls = []

    class FakeModel:
        def invoke(self, prompt):
            calls.append(prompt)
            return SimpleNamespace(content=[{'type': 'text', 'text': 'Live answer'}])

    agent.langchain_agent = FakeModel()
    output_before = agent.token_usage('t')
    prompts_before = agent.prompt_token_usage('t')
    result = agent.reply('u', 't', 'Mình tên gì?')
    prompt = calls[0]
    assert any('User profile:' in m['content'] and 'An' in m['content'] for m in prompt)
    assert any('Earlier thread summary:' in m['content'] for m in prompt)
    assert prompt[-1] == {'role': 'user', 'content': 'Mình tên gì?'}
    assert len([m for m in prompt if m['role'] != 'system']) <= 2
    assert result['prompt_tokens_processed'] == sum(estimate_tokens(m['content']) for m in prompt)
    assert agent.token_usage('t') == output_before + estimate_tokens('Live answer')
    assert agent.prompt_token_usage('t') == prompts_before + result['prompt_tokens_processed']
    agent.force_offline = True
    assert agent.reply('u', 'other', 'Mình tên gì?')['response'] == 'An'
    assert len(calls) == 1


@pytest.mark.parametrize('existing', [True, False])
def test_live_failure_rolls_back_profile_thread_and_counters(tmp_path, existing):
    agent = make_agent(tmp_path, threshold=50, offline=False)
    if existing:
        agent.reply('u', 't', 'Mình ở Huế.')
    profile_before = agent.profile_store.read_text('u')
    profile_exists = agent.profile_store.path_for('u').exists()
    context_before = agent.compact_memory.context('t')
    counters_before = (agent.token_usage('t'), agent.prompt_token_usage('t'))
    owners_before = dict(agent.thread_users)

    class FailingModel:
        def invoke(self, prompt):
            prompt[-1]['content'] = 'mutated'
            raise RuntimeError('Unavailable')

    agent.langchain_agent = FailingModel()
    with pytest.raises(RuntimeError, match='Unavailable'):
        agent.reply('u', 't', 'Mình đang ở Đà Nẵng.')
    assert agent.profile_store.read_text('u') == profile_before
    assert agent.profile_store.path_for('u').exists() == profile_exists
    assert agent.compact_memory.context('t') == context_before
    assert (agent.token_usage('t'), agent.prompt_token_usage('t')) == counters_before
    assert agent.thread_users == owners_before


def test_advanced_builder_uses_provider_without_initializing_implicitly(tmp_path, monkeypatch):
    calls = []
    model = object()

    def build(provider_config):
        calls.append(provider_config)
        return model

    monkeypatch.setattr('agent_advanced.build_chat_model', build)
    agent = make_agent(tmp_path, offline=False)
    assert agent.langchain_agent is None and calls == []
    assert agent._maybe_build_langchain_agent() is model
    assert calls == [agent.config.model]
    agent.force_offline = True
    assert agent._maybe_build_langchain_agent() is None
    assert len(calls) == 1
