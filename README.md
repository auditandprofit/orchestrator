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

The `--parallel` flag sets a **maximum** number of flows that may run
concurrently. When more flows are scheduled, they queue until a slot becomes
available.

You can also supply files whose contents are interpolated into the prompts of
each flow. Each `--key` flag specifies a placeholder and a text file containing
line-separated file paths. The contents of those files are inserted wherever the
placeholder appears in the prompt. The number of lines across the referenced
files determines how many total flows are executed.

```bash
python orchestrator.py config.json --parallel 10 --key foo:paths.txt
```

While running, the orchestrator logs a live view of the number of active flows at
each step, along with overall progress `finished/total`. For a configuration such
as `openai -> codex -> openai`, the log might look like:

```
openai: 1 -> codex: 0 -> openai: 1
```

indicating one flow is at the first OpenAI step and another is nearing completion
at the final OpenAI step.

Each step receives the output of the previous step appended to its prompt. The final output is printed to stdout.

