# Memory layer and agent checkpoints

Implemented: `src/memory_store.py`, `src/agent_baseline.py` and
`src/agent_advanced.py`, following section 6 of
`Day17_Memory_Systems_for_AI_Agent_formatted.md`, README, Guide and Rubric.

## Three memory layers

- Short-term memory: messages isolated by thread ID in process memory. Baseline
  keeps all messages. Advanced retains recent messages after compaction.
- Persistent memory: UTF-8 `User.md` at
  `<state_dir>/profiles/user-<sanitized-id>-<digest>/User.md`. Sanitization and a
  digest prevent path traversal and collisions. Advanced replaces facts by key
  on corrections; Baseline never reads or writes profiles.
- Compact memory: when summary plus messages exceed the heuristic threshold,
  older messages become up to six short excerpts. The most recent
  `keep_messages` remain verbatim. Summaries are bounded and lossy; old news
  details are not guaranteed to survive. Profile facts remain separate.

`estimate_tokens` returns 0 for empty/whitespace input (including `None`), and
otherwise `max(1, len(text.strip()) // 4)`, matching the Codelab example.
The threshold triggers compaction; it is not a hard token cap when the retained
messages alone exceed it. Zero kept messages is supported.

## Advanced Agent: section 6

Each offline turn extracts facts, upserts them into `User.md`, appends the user
message into compact memory, measures profile + summary + recent messages,
generates an answer from saved facts, then appends the assistant answer and
increments both counters. Corrections leave one current value per key.

Profile extraction ignores questions, recall requests, temporary trips and
obvious jokes. Unspecified facts are not guessed. Independent style preferences
are preserved when only part of the style is repeated. A saved `3 bullet` style
produces exactly three bullets and includes the saved style; a saved trade-off
preference produces an explicit recall/token trade-off note.

Persistent recall works in a fresh thread and after creating a new agent
instance. Different users have independent profiles. Reusing an Advanced thread
ID for a different user raises an error before mutating memory.

## Token accounting and live mode

Returned token fields describe the current turn; accessors return accumulated
thread totals. Output tokens count only newly generated assistant text. Offline
prompt tokens include profile, summary and recent user/assistant messages before
response generation. Baseline counts its entire history plus the incoming
message. Per-turn prompt load grows with history; cumulative load can grow
quadratically for equally sized turns.

Both agents default to deterministic offline mode. Opt in to a stateless live
chat model with:

```python
agent.langchain_agent = agent._maybe_build_langchain_agent()
```

The builder uses `build_chat_model(config.model)`; `force_offline=True` suppresses
live calls. Advanced sends a dynamic prompt containing profile, summary and
recent messages, and counts the actual payload including system instructions.
Failed live calls restore its prior profile and thread state without adding
output/prompt counters. Live routing was tested with fake models; no external
LLM request was made. The optional LangGraph tool/checkpointer/middleware bonus
is not implemented; compaction is handled by the existing local manager.

## Verification

From the repository root:

```powershell
.venv\Scripts\python.exe -m pytest src -q
```

Result: **44 passed**, covering storage, paths, deterministic estimates,
corrections/noise, repeated compaction, restart recall, prompt components,
per-turn counters, user/thread isolation, saved style and live failure handling.

Offline dataset checks with default settings and initially empty profiles:

| Dataset | Agent | Conversation output tokens | Conversation prompt tokens | Compactions | Fresh-thread facts recalled | Final profile bytes |
|---|---|---:|---:|---:|---:|---:|
| Standard | Baseline | 727 | 11,181 | 0 | 0/33 | 0 |
| Standard | Advanced | 774 | 17,509 | 0 | 33/33 | 328 |
| Stress | Baseline | 127 | 21,194 | 0 | 0/8 | 0 |
| Stress | Advanced | 653 | 12,074 | 12 | 8/8 | 246 |

Token and compaction columns above include conversation turns only; recall
questions are measured separately by expected-fact hits. Profile bytes are final
file sizes, not cumulative write volume. Advanced carries extra profile context
in short conversations and reduces prompt load by about 43% on the stress input.
Its output grows because it follows the requested three-bullet style.

Generated checkpoint profiles are in `state/advanced_agent_checkpoint/profiles/`.
The benchmark runner remains a scaffold; these are direct dataset checks, not a
completed six-column benchmark with a response-quality score.
