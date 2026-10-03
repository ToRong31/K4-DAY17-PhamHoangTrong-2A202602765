from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Uncompressed, in-process history keyed exclusively by thread ID.

    No profile files, tools, summaries or cross-thread fact cache are used.
    The optional live adapter must be stateless: this class owns its history.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        # Offline by default. A live model can be explicitly attached by the caller.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the answer and per-turn heuristic token counts."""
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({'role': 'user', 'content': message})
        prompt_tokens = sum(estimate_tokens(item['content']) for item in session.messages)
        answer = self._offline_response(session.messages, message)
        return self._record_reply(session, answer, prompt_tokens)

    def _offline_response(self, messages: list[dict[str, str]], message: str) -> str:
        """Re-read only this thread's user declarations for deterministic recall."""
        lower = message.casefold()
        recall = '?' in message or any(word in lower for word in ('nhắc lại', 'tóm tắt', 'nhớ lại'))
        if not recall:
            return 'Mình đã nhận thông tin của bạn.'
        # Reconstruct temporary facts from raw history, never cache them by user.
        facts = {}
        for item in messages:
            if item['role'] == 'user':
                facts.update(extract_profile_updates(item['content']))
        topics = {
            'name': ('tên', 'là ai'),
            'location': ('ở đâu', 'nơi ở', 'huế', 'đà nẵng', 'hà nội'),
            'profession': ('nghề', 'công việc', 'là ai'),
            'response_style': ('style', 'kiểu trả lời', 'trả lời như'),
            'favorite_drink': ('đồ uống',),
            'favorite_food': ('món ăn',),
            'pet': ('nuôi', 'con gì'),
            'interests': ('quan tâm', 'kỹ thuật'),
        }
        values = [facts[key] for key, words in topics.items()
                  if key in facts and any(word in lower for word in words)]
        return '; '.join(values) if values else 'Mình chưa có thông tin đó trong cuộc trò chuyện này.'

    def _record_reply(self, session: SessionState, answer: str, prompt_tokens: int) -> dict[str, Any]:
        output_tokens = estimate_tokens(answer)
        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({'role': 'assistant', 'content': answer})
        return {
            'response': answer,
            'token_usage': output_tokens,
            'prompt_tokens_processed': prompt_tokens,
            'compactions': 0,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        pending = [dict(item) for item in session.messages]
        pending.append({'role': 'user', 'content': message})
        prompt_tokens = sum(estimate_tokens(item['content']) for item in pending)
        result = self.langchain_agent.invoke(pending)
        content = result.content
        if isinstance(content, str):
            answer = content
        elif isinstance(content, list):
            answer = ''.join(block if isinstance(block, str) else block.get('text', '')
                             for block in content)
        else:
            raise TypeError('Live chat model must return text or text content blocks')
        session.messages.append({'role': 'user', 'content': message})
        return self._record_reply(session, answer, prompt_tokens)

    def _maybe_build_langchain_agent(self):
        """Build a stateless chat model without profile tools or a checkpointer.

        Opt in with agent.langchain_agent = agent._maybe_build_langchain_agent().
        Missing credentials or integrations propagate to the caller.
        """
        if self.force_offline:
            return None
        return build_chat_model(self.config.model)
