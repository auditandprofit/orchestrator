import threading
from pathlib import Path
import orchestrator

def test_cmd_output_saved(tmp_path):
    config = [
        {"type": "cmd", "cmd": "printf '[\"a\",\"b\"]'", "array": True},
        {"type": "cmd", "cmd": "cat"},
    ]
    results, failed = orchestrator._run_flow(
        config, [0, 0], threading.Lock(), tmp_path, tmp_path
    )

    assert (tmp_path / "step_0_cmd.txt").read_text() == '["a","b"]'
    assert (tmp_path / "branch_0" / "step_1_cmd.txt").read_text() == "a"
    assert (tmp_path / "branch_1" / "step_1_cmd.txt").read_text() == "b"
    assert sorted(r[0] for r in results) == ["a", "b"]
    assert not failed


def test_cmd_respects_process_cwd(tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    default_dir = tmp_path / "default"
    default_dir.mkdir()
    data_file = default_dir / "data.txt"
    data_file.write_text("hello", encoding="utf-8")

    monkeypatch.chdir(default_dir)

    config = [{"type": "cmd", "cmd": "cat data.txt"}]

    results, failed = orchestrator._run_flow(
        config, [0], threading.Lock(), workdir, tmp_path
    )

    assert not failed
    assert results[0][0] == "hello"


def test_cmd_inputs_can_select_named_steps(tmp_path):
    config = [
        {"type": "cmd", "cmd": "printf alpha", "name": "first"},
        {"type": "cmd", "cmd": "printf beta", "name": "second"},
        {"type": "cmd", "cmd": "cat", "inputs": ["second", "first"]},
    ]

    results, failed = orchestrator._run_flow(
        config, [0, 0, 0], threading.Lock(), tmp_path, tmp_path
    )

    assert not failed
    assert results[0][0] == "beta\nalpha"


def test_cmd_inputs_accept_step_indexes(tmp_path):
    config = [
        {"type": "cmd", "cmd": "printf foo"},
        {"type": "cmd", "cmd": "printf bar"},
        {"type": "cmd", "cmd": "cat", "inputs": [1, 0]},
    ]

    results, failed = orchestrator._run_flow(
        config, [0, 0, 0], threading.Lock(), tmp_path, tmp_path
    )

    assert not failed
    assert results[0][0] == "bar\nfoo"


def test_cmd_allows_stdin_manifest(tmp_path):
    stdin_source = tmp_path / "input.txt"
    stdin_source.write_text("hello from file", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{stdin_source}\n", encoding="utf-8")

    base_config = [
        {"type": "cmd", "cmd": "cat", "stdin_file": str(manifest)},
    ]

    flow_configs = orchestrator._generate_flow_configs(base_config, {})
    assert len(flow_configs) == 1
    flow_config = flow_configs[0]
    assert flow_config.interpolated_paths == (str(stdin_source),)

    flow_dir = tmp_path / "flow"
    flow_dir.mkdir()

    results, failed = orchestrator._run_flow(
        flow_config, [0], threading.Lock(), tmp_path, flow_dir
    )

    assert not failed
    assert results[0][0] == "hello from file"
    output_path = flow_dir / "step_0_cmd.txt"
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == "hello from file"


def test_generate_flow_configs_expands_stdin_manifests(tmp_path):
    data_a = tmp_path / "data_a.txt"
    data_a.write_text("AAA", encoding="utf-8")
    data_b = tmp_path / "data_b.txt"
    data_b.write_text("BBB", encoding="utf-8")
    stdin_manifest = tmp_path / "stdin_manifest.txt"
    stdin_manifest.write_text(f"{data_a}\n{data_b}\n", encoding="utf-8")

    key_data_one = tmp_path / "key_one.txt"
    key_data_one.write_text("first", encoding="utf-8")
    key_data_two = tmp_path / "key_two.txt"
    key_data_two.write_text("second", encoding="utf-8")
    key_manifest = tmp_path / "key_manifest.txt"
    key_manifest.write_text(
        f"{key_data_one}\n{key_data_two}\n", encoding="utf-8"
    )

    base_config = [
        {"type": "cmd", "cmd": "cat", "stdin_file": str(stdin_manifest)},
        {"type": "cmd", "cmd": "printf '{{{label}}}'"},
    ]

    flow_configs = orchestrator._generate_flow_configs(
        base_config,
        {"label": key_manifest},
    )

    assert len(flow_configs) == 4
    generated_pairs = set()
    for config in flow_configs:
        stdin_path = config[0]["stdin_file"]
        assert Path(stdin_path).read_text(encoding="utf-8") in {"AAA", "BBB"}
        assert config.interpolated_paths[0] in {str(key_data_one), str(key_data_two)}
        assert config.interpolated_paths[1] in {str(data_a), str(data_b)}
        generated_pairs.add((config.interpolated_paths[0], config.interpolated_paths[1]))
        assert config[1]["cmd"] in {"printf 'first'", "printf 'second'"}

    assert generated_pairs == {
        (str(key_data_one), str(data_a)),
        (str(key_data_one), str(data_b)),
        (str(key_data_two), str(data_a)),
        (str(key_data_two), str(data_b)),
    }
