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
line-separated file paths. Placeholders must be wrapped in triple braces (e.g.
`{{{foo}}}`). The contents of those files are inserted wherever the matching
placeholder appears in the prompt. Flows are generated for every combination of
lines across the supplied files, so the total number of flows equals the
product of the line counts for each key file.

```bash
python orchestrator.py config.json --parallel 10 --key foo:paths.txt
```

Any prompt containing `{{{foo}}}` will have that placeholder replaced with the
contents of each file listed in `paths.txt`.

While running, the orchestrator logs a live view of the number of active flows at
each step, along with overall progress `finished/total`. For a configuration such
as `openai -> codex -> openai`, the log might look like:

```
openai: 1 -> codex: 0 -> openai: 1
```

indicating one flow is at the first OpenAI step and another is nearing completion
at the final OpenAI step.

Each step receives the output of the previous step appended to its prompt. When
the final step is handled by the codex CLI, its concluding message is written to
a file inside the `generated` directory so it remains available after the run.
The script prints the path so downstream code can read the message:

```python
with open("generated/codex_exec_xxxx/final_message.txt") as f:
    final_message = f.read()
```

The directory is left intact for logging.

