# Orchestrator

A simple configurable orchestration system that chains Codex and OpenAI model calls.

## Usage

Create a JSON configuration describing each step. For built-in model calls, each
item requires a `type` (`"codex"` or `"openai"`) and either a `prompt` or a
`prmpt_file` pointing to a file containing the prompt. You can optionally
include a `name` to use in the live progress output instead of the step type.
Steps may also include a `cmd` field to run an arbitrary shell
command; the previous step's output is piped to the command's standard input and
its standard output is passed to the next step.

```json
[
  {"type": "openai", "name": "draft", "prmpt_file": "prompts/draft.txt"},
  {"type": "openai", "name": "summary", "prompt": "Summarize the previous output."},
  {"type": "shout", "cmd": "tr '[:lower:]' '[:upper:]'"}
]
```

Run the orchestrator with the path to this file:

```bash
python orchestrator.py path/to/config.json --parallel 10
```

The `--parallel` flag sets a **maximum** number of flows that may run
concurrently. When more flows are scheduled, they queue until a slot becomes
available.

Use `--timeout` to specify a timeout (in seconds) for each Codex CLI invocation:

```bash
python orchestrator.py config.json --timeout 30 --workdir repo
```

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
contents of each file listed in `paths.txt`. Placeholders are also substituted in
`prmpt_file` paths and `cmd` strings. Use `--append-filepath` to append the path
of each interpolated file after its contents in the prompt.

While running, the orchestrator logs a live view of the number of active flows at
each step, along with overall progress `finished/total`. If a step includes a
`name`, that value appears in the log instead of the step's `type`. For a
configuration such as `draft -> codex -> summary`, the log might look like:

```
draft: 1 -> codex: 0 -> summary: 1
```

indicating one flow is at the first step and another is nearing completion at
the final step.

Each step receives the output of the previous step appended to its prompt. Each
orchestrator invocation creates a dedicated `run_xxxx` directory inside
`generated`, and flows are stored beneath that run directory (for example,
`generated/run_1234/flow_5678`). Codex invocations create subdirectories within
each flow. During execution, the Codex process's standard output is streamed to
`stdout.txt` in its subdirectory. When the final step is handled by the codex
CLI, its concluding message is written to `final_message.txt` in the same
location so it remains available after the run. The script prints the path so
downstream code can read the message:

```python
with open("generated/run_xxxx/flow_xxxx/codex_exec_xxxx/final_message.txt") as f:
    final_message = f.read()
```

Each flow directory is left intact for logging.

### Branching with arrays

Add an `"array": true` field to any step that is expected to produce a JSON
array. The orchestrator parses the array and runs the remaining steps once for
each element in parallel. Each element is passed to the next phase as the
previous step's output, and the final results from all branches are returned as
separate flow outputs.

