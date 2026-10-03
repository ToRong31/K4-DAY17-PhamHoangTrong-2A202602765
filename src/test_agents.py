from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build all seven fields without loading .env or touching repo state."""
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / 'data',
        state_dir=tmp_path / 'state',
        compact_threshold_tokens=80,  # Small threshold triggers compaction early.
        compact_keep_messages=2,
        model=ProviderConfig(provider='openai', model_name='stub', temperature=0.0),
        judge_model=ProviderConfig(provider='openai', model_name='stub', temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / 'profiles')
    original = '# User profile\n- name: Lan\n- location: Huế\n'
    assert store.read_text('user-1') == ''
    path = store.write_text('user-1', original)
    assert path.is_relative_to(tmp_path)
    assert path.name == 'User.md'
    assert path.read_text(encoding='utf-8') == original
    assert store.read_text('user-1') == original
    assert store.edit_text('user-1', 'Huế', 'Đà Nẵng') is True
    expected = original.replace('Huế', 'Đà Nẵng')
    assert store.read_text('user-1') == expected
    assert path.read_text(encoding='utf-8') == expected


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    assert memory.compaction_count('long-thread') == 0
    for index in range(10):
        memory.append('long-thread', 'user', f'Lượt {index}: ' + 'Nội dung kỹ thuật dài. ' * 40)
    assert memory.compaction_count('long-thread') > 0
    context = memory.context('long-thread')
    assert context['summary'].strip()
    assert len(context['messages']) <= config.compact_keep_messages
    assert context['messages'][-1]['content'].startswith('Lượt 9:')


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    user_id = 'same-user'
    name = 'LanMemoryTest'
    declaration = f'Mình tên là {name}.'
    question = 'Mình tên gì?'
    for agent in (advanced, baseline):
        agent.reply(user_id, 'first-thread', declaration)
        assert name in agent.reply(user_id, 'first-thread', question)['response']
    assert name in advanced.reply(user_id, 'second-thread', question)['response']
    assert name not in baseline.reply(user_id, 'second-thread', question)['response']
    # A new instance must also recover the fact from the persisted profile.
    restarted = AdvancedAgent(config, force_offline=True)
    assert name in restarted.reply(user_id, 'third-thread', question)['response']


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    thread_id = 'long-thread'
    for index in range(20):
        message = f'Lượt {index}: ' + 'Nội dung dài về kỹ thuật. ' * 100
        # Matching answers keep differences in output length out of this comparison.
        a = advanced.reply('user-1', thread_id, message)
        b = baseline.reply('user-1', thread_id, message)
        assert a['response'] == b['response']
    assert advanced.compaction_count(thread_id) > 0
    assert baseline.compaction_count(thread_id) == 0
    assert 0 < advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
