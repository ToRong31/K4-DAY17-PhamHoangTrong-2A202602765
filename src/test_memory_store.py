from dataclasses import replace
import json
from pathlib import Path
import unicodedata

import pytest

from agent_advanced import AdvancedAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, summarize_messages


def test_token_estimator():
    assert estimate_tokens('') == estimate_tokens('   \n') == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens('a') == 1
    assert estimate_tokens('abcde') == 1
    assert estimate_tokens('abcdefgh') == 2
    counts = [estimate_tokens('a' * length) for length in range(100)]
    assert counts == sorted(counts)
    assert estimate_tokens('Xin chào') == estimate_tokens('Xin chào') > 0


def test_profile_storage_and_correction(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.read_text('u') == ''
    assert store.file_size('u') == 0
    path = store.write_text('u', '# User\nHuế Huế\n')
    assert path.name == 'User.md'
    assert store.file_size('u') == len(path.read_bytes())
    assert store.edit_text('u', 'Huế', 'Đà Nẵng')
    assert store.read_text('u') == '# User\nĐà Nẵng Huế\n'
    assert not store.edit_text('u', 'missing', 'x')
    assert not store.edit_text('u', '', 'x')
    assert not store.edit_text('u', 'Huế', 'Huế')
    store.upsert_fact('u', 'location', 'Huế')
    store.upsert_fact('u', 'location', 'Đà Nẵng')
    assert UserProfileStore(tmp_path).facts('u') == {'location': 'Đà Nẵng'}
    assert store.read_text('u').count('- location:') == 1


@pytest.mark.parametrize('user_id', ['../../outside', r'..\..\outside', 'C:/outside', '/outside', '', 'CON', 'a:b', 'a/b'])
def test_safe_paths(tmp_path, user_id):
    store = UserProfileStore(tmp_path)
    assert store.write_text(user_id, 'profile').resolve().is_relative_to(tmp_path.resolve())
    assert store.path_for('a/b') != store.path_for('a:b')
    assert store.path_for('User') != store.path_for('user')


@pytest.mark.parametrize('message', [
    'Mình tên gì?', 'Đồ uống yêu thích của mình là gì?',
    'Mình ở đâu?', 'Mình muốn bạn trả lời như thế nào?',
    'Mình vừa bay ra Hà Nội họp hai ngày, không phải nơi ở hiện tại.',
    'Mình là product manager chỉ là câu nói đùa.',
    'Mình làm product manager, nhưng chỉ nói đùa thôi.',
    'Nếu mình ở Hà Nội thì sao?', 'Ví dụ cũ: mình ở Đà Nẵng.',
    'Mình không còn ở Huế.',
    'Nhắc lại giúp mình tên và style trả lời mình thích.',
])
def test_noisy_messages_do_not_create_facts(message):
    assert extract_profile_updates(message) == {}


def test_extraction_and_corrections():
    facts = extract_profile_updates('Mình tên là DũngCT. Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.')
    assert facts == {'name': 'DũngCT', 'location': 'Đà Nẵng', 'profession': 'backend engineer'}
    assert extract_profile_updates('Giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.') == {'location': 'Huế'}
    assert extract_profile_updates('Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.') == {'profession': 'MLOps engineer'}
    assert extract_profile_updates(unicodedata.normalize('NFD', 'Mình ở Huế.')) == {'location': 'Huế'}


def test_compaction_keeps_recent_messages_and_is_bounded():
    memory = CompactMemoryManager(100, 2)
    history = []
    previous = 0
    for index in range(40):
        content = f'{index}: ' + 'long text ' * 100
        history.append({'role': 'user', 'content': content})
        memory.append('thread', 'user', content)
        context = memory.context('thread')
        if context['compactions'] > previous:
            assert context['messages'] == history[-2:]
            assert len(context['summary']) <= 6 * 180
        previous = context['compactions']
    assert previous > 1
    assert memory.context('other') == {'messages': [], 'summary': '', 'compactions': 0}
    context['messages'][0]['content'] = 'external mutation'
    assert memory.context('thread')['messages'] == history[-2:]
    assert summarize_messages([], 0) == ''
    assert summarize_messages(history) == summarize_messages(history)


def test_zero_keep_and_previous_summary():
    memory = CompactMemoryManager(1, 0)
    memory.append('t', 'user', 'important first fact')
    memory.append('t', 'user', 'second fact')
    assert memory.context('t')['messages'] == []
    assert 'important first fact' in memory.context('t')['summary']
    assert memory.compaction_count('t') == 2


def test_advanced_long_context_checkpoint(tmp_path):
    config = replace(load_config(), state_dir=tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    data = json.loads((config.data_dir / 'advanced_long_context.json').read_text(encoding='utf-8'))
    conversation = data[0]
    user = conversation['user_id']
    thread = conversation['id']
    baseline_prompt_load = 0
    full_history = []
    for message in conversation['turns']:
        full_history.append(message)
        baseline_prompt_load += sum(estimate_tokens(item) for item in full_history)
        result = agent.reply(user, thread, message)
        full_history.append(result['response'])
    profile = agent.profile_store.facts(user)
    assert profile['location'] == 'Đà Nẵng'
    assert profile['profession'] == 'MLOps engineer'
    assert 'Hà Nội' not in agent.profile_store.read_text(user)
    assert 'product manager' not in agent.profile_store.read_text(user)
    assert agent.memory_file_size(user) > 0
    assert agent.compaction_count(thread) > 0
    assert agent.prompt_token_usage(thread) < baseline_prompt_load
    reloaded = AdvancedAgent(config, force_offline=True)
    original = reloaded.profile_store.read_text(user)
    for index, question in enumerate(conversation['recall_questions']):
        answer = reloaded.reply(user, f'recall-{index}', question['question'])['response']
        assert all(value in answer for value in question['expected_contains'])
    assert reloaded.profile_store.read_text(user) == original


def test_standard_dataset_recall(tmp_path):
    config = replace(load_config(), state_dir=tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    data = json.loads((config.data_dir / 'conversations.json').read_text(encoding='utf-8'))
    for conversation in data:
        for message in conversation['turns']:
            agent.reply(conversation['user_id'], conversation['id'], message)
        for index, question in enumerate(conversation['recall_questions']):
            answer = agent.reply(conversation['user_id'], f"{conversation['id']}-recall-{index}", question['question'])['response']
            assert all(value in answer for value in question['expected_contains']), answer
