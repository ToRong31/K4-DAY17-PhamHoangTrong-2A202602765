from pathlib import Path
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    return LabConfig(
        base_dir=tmp_path, data_dir=tmp_path / 'data', state_dir=tmp_path / 'state',
        compact_threshold_tokens=80, compact_keep_messages=2,
        model=ProviderConfig('openai', 'stub', 0.0),
        judge_model=ProviderConfig('openai', 'stub', 0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(make_config(tmp_path).state_dir / 'profiles')
    assert store.read_text('u') == ''
    path = store.write_text('u', '# User\n- location: Huế\n')
    assert path.exists()
    assert store.edit_text('u', 'Huế', 'Đà Nẵng')
    assert 'Đà Nẵng' in store.read_text('u')
    assert not store.edit_text('u', 'missing', 'x')


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    assert memory.compaction_count('long') == 0
    for index in range(10):
        memory.append('long', 'user', f'Topic {index}: ' + 'x' * 400)
    assert memory.compaction_count('long') > 0
    assert memory.context('long')['summary'].strip()
    assert len(memory.context('long')['messages']) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for agent in (advanced, baseline):
        agent.reply('same-user', 'first', 'Mình tên là An. Mình ở Huế.')
        assert 'An' in agent.reply('same-user', 'first', 'Mình tên gì?')['response']
    assert 'An' in advanced.reply('same-user', 'new', 'Mình tên gì?')['response']
    restarted = AdvancedAgent(config, force_offline=True)
    assert 'An' in restarted.reply('same-user', 'restart', 'Mình tên gì?')['response']
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
    assert advanced.compaction_count('long') > 0
