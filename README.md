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
python orchestrator.py path/to/config.json --parallel 10
```

The `--parallel` flag controls how many end-to-end flows are executed in parallel.
While running, the orchestrator logs a live view of the number of active flows at
each step. For a configuration such as `openai -> codex -> openai`, the log might
look like:

```
openai: 1 -> codex: 0 -> openai: 1
```

indicating one flow is at the first OpenAI step and another is nearing completion
at the final OpenAI step.

Each step receives the output of the previous step appended to its prompt. The final output is printed to stdout.

