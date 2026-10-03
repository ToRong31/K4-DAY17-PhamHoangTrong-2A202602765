from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import unicodedata


def estimate_tokens(text: str) -> int:
    """Deterministic character-count heuristic, not a provider tokenizer."""
    stripped = (text or '').strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """UTF-8 profiles stored at root_dir/<safe user id>/User.md."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        normalized = unicodedata.normalize('NFC', user_id)
        slug = re.sub(r'[^A-Za-z0-9_-]', '_', normalized)[:80] or 'user'
        # Prevent sanitization, case folding and Windows reserved-name collisions.
        digest = hashlib.sha256(user_id.encode('utf-8')).hexdigest()[:16]
        root = Path(self.root_dir).resolve()
        path = (root / f'user-{slug}-{digest}' / 'User.md').resolve()
        if not path.is_relative_to(root):
            raise ValueError('Profile path escapes root_dir')
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding='utf-8') if path.exists() else ''

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8', newline='\n')
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content or search_text == replacement:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r'^- ([\w-]+): (.+)$', self.read_text(user_id), re.M))

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        if not re.fullmatch(r'[\w-]+', key) or not value.strip() or '\n' in value or '\r' in value:
            raise ValueError('Facts require a simple key and a nonempty single-line value')
        content = self.read_text(user_id) or '# User profile\n'
        line = f'- {key}: {value.strip()}'
        pattern = re.compile(rf'^- {re.escape(key)}: .*?$', re.M)
        if pattern.search(content):
            # Replace the existing line and discard legacy duplicate keys.
            lines = content.splitlines()
            found = False
            result = []
            for existing in lines:
                if pattern.fullmatch(existing):
                    if not found:
                        result.append(line)
                        found = True
                else:
                    result.append(existing)
            content = '\n'.join(result) + '\n'
        else:
            content = content.rstrip() + '\n' + line + '\n'
        return self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative Vietnamese declarations; ignore questions, trips and jokes.

    This is a lab heuristic, not a general natural-language fact extractor.
    """
    updates: dict[str, str] = {}
    subject = r'(?:mình|tôi|tớ)'
    patterns = {
        'name': rf'{subject}\s+tên\s+(?:là\s+)?(.+?)(?=[,.!?;]|$)',
        'location': rf'(?:{subject}\s+(?:(?:hiện tại|hiện|giờ|vẫn|đang)\s+)*|(?<=, )hiện\s+)ở\s+(?:là\s+)?(.+?)(?=\s+(?:và|chứ|nhưng|chưa|để|trong|dù)|[,.!?;]|$)',
        'profession': rf'(?:{subject}|và)\s+(?:(?:hiện tại|vẫn|đang)\s+)*làm\s+(?:nghề\s+)?(.+?)(?=\s+(?:cho|chứ|nhưng)|[,.!?;]|$)',
        'favorite_drink': r'đồ uống yêu thích\s+(?:của mình\s+)?là\s+(.+?)(?=[.!?;]|$)',
        'favorite_food': r'món ăn yêu thích\s+(?:của mình\s+)?là\s+(.+?)(?=[.!?;]|$)',
        'pet': rf'{subject}\s+nuôi\s+(.+?)(?=[.!?;]|$)',
        'interests': rf'{subject}\s+(?:(?:vẫn|còn|đang)\s+)*(?:thích|quan tâm (?:nhiều )?đến)\s+(.+?)(?=[.!?;]|$)',
    }
    message = unicodedata.normalize('NFC', message)
    message = re.sub(r'"[^"\n]*"|“[^”\n]*”', '', message)
    for sentence in re.split(r'(?<=[.!?;])\s+', message):
        lower = sentence.casefold()
        if '?' in sentence or re.search(r'\b(?:đùa|giả sử|nếu|tạm thời|nhắc lại giúp mình|nhắc lại xem)\b', lower):
            continue
        for key, pattern in patterns.items():
            for match in re.finditer(pattern, sentence, re.I):
                value = match.group(1).strip()
                if key == 'profession' and not re.fullmatch(r'[\w +#-]+(?:engineer|developer|manager|designer|teacher|bác sĩ|giáo viên)', value, re.I):
                    continue
                if key == 'location' and re.search(r'\b(?:họp|du lịch|hai ngày|hôm nay|quán|ví dụ)\b', lower):
                    continue
                if value and not re.search(r'\b(?:gì|đâu|không|như thế nào)\b', value, re.I):
                    updates[key] = value
        correction = re.search(r'giờ\s+(?:mình\s+)?chuyển sang\s+([\w +#-]+engineer)', sentence, re.I)
        if correction:
            updates['profession'] = correction.group(1).strip()
        if re.search(r'(?:muốn|hãy).*?(?:trả lời|câu trả lời|style)', lower):
            style = []
            if re.search(r'ngắn|gọn', lower):
                style.append('ngắn gọn')
            if '3 bullet' in lower:
                style.append('3 bullet')
            elif 'bullet' in lower:
                style.append('bullet')
            if 'ví dụ' in lower:
                style.append('ví dụ thực chiến' if 'thực chiến' in lower else 'ví dụ thực tế')
            if 'trade-off' in lower:
                style.append('trade-off')
            if style:
                updates['response_style'] = ', '.join(style)
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Bounded deterministic excerpts; summaries are lossy by design."""
    if max_items <= 0:
        return ''
    items = []
    for message in messages:
        if message['role'] == 'summary':
            items.extend(message['content'].splitlines())
        else:
            content = ' '.join(message['content'].split())
            if content:
                items.append(f"{message['role']}: {content[:160]}")
    return '\n'.join(list(dict.fromkeys(items))[-max_items:])


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages < 0:
            raise ValueError('threshold_tokens must be positive; keep_messages must be nonnegative')

    def append(self, thread_id: str, role: str, content: str) -> None:
        context = self.state.setdefault(thread_id, {'messages': [], 'summary': '', 'compactions': 0})
        messages = context['messages']
        messages.append({'role': role, 'content': content})
        tokens = estimate_tokens(context['summary']) + sum(estimate_tokens(m['content']) for m in messages)
        if tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split = len(messages) - self.keep_messages
            previous = [{'role': 'summary', 'content': context['summary']}] if context['summary'] else []
            context['summary'] = summarize_messages(previous + messages[:split])
            context['messages'] = messages[split:]
            context['compactions'] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        context = self.state.get(thread_id, {'messages': [], 'summary': '', 'compactions': 0})
        return {**context, 'messages': [dict(m) for m in context['messages']]}

    def compaction_count(self, thread_id: str) -> int:
        return self.state.get(thread_id, {}).get('compactions', 0)
