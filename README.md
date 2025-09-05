# Orchestrator

A simple configurable orchestration system that chains Codex and OpenAI model calls.

## Usage

Create a JSON configuration describing each step. Each item requires a `type` (`"codex"` or `"openai"`) and a `prompt`.

```json
[
  {"type": "openai", "prompt": "Write a limerick about orchestration."},
  {"type": "openai", "prompt": "Summarize the previous output."}
]
```

Run the orchestrator with the path to this file:

```bash
python orchestrator.py path/to/config.json
```

Each step receives the output of the previous step appended to its prompt. The final output is printed to stdout.

