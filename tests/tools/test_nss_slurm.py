# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for script/slurm/nss_slurm.py, the login-node Slurm CLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_cli(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("nss_slurm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def repo_root(pytestconfig: pytest.Config) -> Path:
    return Path(pytestconfig.rootpath)


@pytest.fixture(scope="module")
def cli(repo_root: Path) -> Any:
    return _load_cli(repo_root / "script" / "slurm" / "nss_slurm.py")


@pytest.fixture
def settings(cli: Any, tmp_path: Path) -> Any:
    nss_dir = tmp_path / "repo"
    slurm_dir = nss_dir / "script" / "slurm"
    config_dir = slurm_dir / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "tiny.yaml").write_text("training:\n  pretrained_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n")
    (nss_dir / ".python-version").write_text("3.13\n")
    (nss_dir / "uv.lock").write_bytes(b"lock-v1")
    lustre_dir = tmp_path / "lustre" / "jdoe"
    lustre_dir.mkdir(parents=True)
    (lustre_dir / ".api_tokens.sh").write_text("export NSS_INFERENCE_KEY=x\n")
    shared_dir = tmp_path / "shared"
    (shared_dir / "images").mkdir(parents=True)
    image = shared_dir / "images" / "nss-slurm-py3.13-abc.sqsh"
    image.write_bytes(b"")
    base_image = shared_dir / "images" / "base.sqsh"
    base_image.write_bytes(b"")
    return cli.Settings(
        user_name="jdoe",
        lustre_dir=lustre_dir,
        shared_dir=shared_dir,
        nss_dir=nss_dir,
        slurm_dir=slurm_dir,
        config_dir=config_dir,
        base_log_dir=lustre_dir / "nss_results",
        image=image,
        base_image=base_image,
        models_dir=shared_dir / "models",
        default_configs=("tiny",),
    )


class TestSettings:
    def test_from_env_reads_paths_and_defaults(self, cli: Any):
        env = {
            "USER_NAME": "jdoe",
            "LUSTRE_DIR": "/l/jdoe",
            "NSS_SHARED_DIR": "/s",
            "NSS_DIR": "/home/jdoe/code/nss",
            "NSS_SLURM_DIR": "/home/jdoe/code/nss/script/slurm",
            "CONFIG_DIR": "/home/jdoe/code/nss/script/slurm/configs",
            "BASE_LOG_DIR": "/l/jdoe/nss_results",
            "NSS_IMAGE": "/s/images/a.sqsh",
            "NSS_BASE_IMAGE": "/s/images/base.sqsh",
            "NSS_MODELS_DIR": "/s/models",
            "NSS_DEFAULT_CONFIGS": "a, b,,c",
        }
        settings = cli.Settings.from_env(env)
        assert settings.account == cli.DEFAULT_ACCOUNT
        assert settings.default_configs == ("a", "b", "c")
        assert settings.image == Path("/s/images/a.sqsh")

    def test_from_env_requires_user_name(self, cli: Any):
        with pytest.raises(cli.CliError, match="USER_NAME"):
            cli.Settings.from_env({})


class TestMatrix:
    def test_pack_matrix_orders_config_then_dataset_then_run(self, cli: Any):
        configs, datasets = cli.pack_matrix(["a", "b"], ["x", "y"], 2)
        assert configs == "a,a,a,a,b,b,b,b"
        assert datasets == "x,x,y,y,x,x,y,y"

    @pytest.mark.parametrize(
        ("configs", "datasets", "runs"),
        [(["a"], ["x"], 0), ([], ["x"], 1), (["a,b"], ["x"], 1), (["a"], ["x,y"], 1)],
    )
    def test_pack_matrix_rejects_bad_input(self, cli: Any, configs: list[str], datasets: list[str], runs: int):
        with pytest.raises(cli.CliError):
            cli.pack_matrix(configs, datasets, runs)

    def test_array_spec(self, cli: Any):
        assert cli.array_spec(3) == "0-2"
        assert cli.array_spec(3, 2) == "0-2%2"
        with pytest.raises(cli.CliError):
            cli.array_spec(0)
        with pytest.raises(cli.CliError):
            cli.array_spec(3, 0)

    def test_resolve_datasets(self, cli: Any):
        assert cli.resolve_datasets("adult, /data/x.csv", None) == ("adult", "/data/x.csv")
        assert cli.resolve_datasets(None, "long") == cli.DATASET_GROUPS["long"]
        for urls, group in [("adult", "short"), (None, None), (None, "medium")]:
            with pytest.raises(cli.CliError):
                cli.resolve_datasets(urls, group)


class TestImageAndModels:
    def test_image_name_uses_python_version_and_lock_hash(self, cli: Any):
        expected = hashlib.sha256(b"lock").hexdigest()[:12]
        assert cli.image_name("3.13\n", b"lock") == f"nss-slurm-py3.13-{expected}.sqsh"

    def test_pinned_uv_release_parses_mise_lock(self, cli: Any):
        text = (
            '[[tools.uv]]\nversion = "0.9.30"\nbackend = "aqua:astral-sh/uv"\n\n'
            '[tools.uv."platforms.linux-x64"]\nchecksum = "sha256:abc123"\n'
            'url = "https://example.com/uv-x86_64.tar.gz"\n'
        )
        assert cli.pinned_uv_release(text) == ("https://example.com/uv-x86_64.tar.gz", "abc123")
        with pytest.raises(cli.CliError):
            cli.pinned_uv_release(text, platform="linux-arm64")

    def test_pinned_uv_release_reads_repo_lockfile(self, cli: Any, repo_root: Path):
        url, checksum = cli.pinned_uv_release((repo_root / "mise.lock").read_text())
        assert url.endswith("x86_64-unknown-linux-musl.tar.gz")
        assert len(checksum) == 64

    def test_configured_models(self, cli: Any, repo_root: Path):
        texts = ['training:\n  pretrained_model: "Org/A"  # comment\n', "training:\n  pretrained_model: Org/B\n", ""]
        assert cli.configured_models(texts) == ["Org/A", "Org/B"]
        repo_configs = [p.read_text() for p in (repo_root / "script" / "slurm" / "configs").glob("*.yaml")]
        assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in cli.configured_models(repo_configs)

    def test_compare_image_metadata(self, cli: Any):
        assert cli.compare_image_metadata("3.13\n", b"a", "3.13", b"a") is None
        assert "uv.lock differs" in cli.compare_image_metadata("3.13", b"a", "3.13", b"b")
        with pytest.raises(cli.CliError, match="Python 3.14"):
            cli.compare_image_metadata("3.13", b"a", "3.14", b"a")


class TestSbatchArgs:
    def test_end_to_end_args(self, cli: Any, settings: Any, tmp_path: Path):
        common = cli.submit_common_args(settings, "polar4", tmp_path / "logs", "0-3%2", dry_run=False)
        argv = cli.phase_args(common, settings, "end_to_end", "01:00:00")
        assert argv[:2] == ["sbatch", "--parsable"]
        assert argv[argv.index("--array") + 1] == "0-3%2"
        assert argv[argv.index("--output") + 1] == str(tmp_path / "logs" / "slurm_%A_%a.out")
        assert "--test-only" not in argv
        assert "--time=01:00:00" in argv
        assert argv[-2:] == ["--export=ALL,NSS_PHASE=end_to_end", str(settings.slurm_dir / "slurm_srun.sh")]

    def test_generate_depends_on_train(self, cli: Any, settings: Any, tmp_path: Path):
        common = cli.submit_common_args(settings, "polar4", tmp_path, "0-0", dry_run=True)
        argv = cli.phase_args(common, settings, "generate", "00:30:00", dependency="123")
        assert "--test-only" in argv
        assert "--dependency=aftercorr:123" in argv
        assert argv[argv.index("--job-name") + 1] == "nss_generate"

    @pytest.mark.parametrize(("force", "wait"), [(False, False), (True, True)])
    def test_build_image_call(self, cli: Any, settings: Any, force: bool, wait: bool):
        call = cli.build_image_call(
            settings,
            "img.sqsh",
            ("https://u", "sha"),
            partition="cpu_short",
            time_limit="01:00:00",
            force=force,
            wait=wait,
        )
        assert call.argv[-2:] == [str(settings.slurm_dir / "build_image.sh"), "--job"]
        assert ("--wait" in call.argv) is wait
        assert call.env["NSS_FORCE"] == ("1" if force else "0")
        assert call.env["NSS_IMAGES_DIR"] == str(settings.shared_dir / "images")
        assert call.env["REPO_ROOT"] == str(settings.nss_dir)

    def test_download_models_call(self, cli: Any, settings: Any):
        call = cli.download_models_call(settings, ["Org/A", "Org/B"], partition="cpu_short", wait=False)
        assert call.argv[-2:] == [str(settings.slurm_dir / "download_models.sh"), "--job"]
        assert call.env["NSS_DOWNLOAD_MODELS"] == "Org/A Org/B"


def _submit_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "configs": None,
        "dataset_urls": "adult",
        "dataset_group": None,
        "runs": 1,
        "partition": "polar4",
        "exp_name": "exp",
        "pipeline_mode": "end_to_end",
        "wandb_project": None,
        "max_concurrent_slurm_jobs": None,
        "time_limit": "04:00:00",
        "train_time_limit": None,
        "generate_time_limit": None,
        "job_id_file": None,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestCommands:
    @pytest.fixture
    def sbatch_calls(self, cli: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        calls: list[Any] = []

        def fake_run_sbatch(call: Any, base_env: Any) -> str:
            calls.append(call)
            return str(100 + len(calls))

        monkeypatch.setattr(cli, "run_sbatch", fake_run_sbatch)
        monkeypatch.setattr(cli, "read_image_metadata", lambda image: ("3.13\n", b"lock-v1"))
        return calls

    def test_two_stage_submit_links_generate_to_train(
        self, cli: Any, settings: Any, sbatch_calls: list[Any], tmp_path: Path
    ):
        job_id_file = tmp_path / "ids" / "job_id"
        args = _submit_args(pipeline_mode="two_stage", runs=2, job_id_file=str(job_id_file))
        assert cli.cmd_submit(args, settings, {}) == 0
        train, generate = sbatch_calls
        assert "--export=ALL,NSS_PHASE=train" in train.argv
        assert "--dependency=aftercorr:101" in generate.argv
        assert train.env["PACKED_CONFIGS"] == "tiny,tiny"
        assert train.env["WANDB_PROJECT"] == "exp"
        assert job_id_file.read_text() == "102\n"

    def test_submit_rejects_missing_config(self, cli: Any, settings: Any, sbatch_calls: list[Any]):
        with pytest.raises(cli.CliError, match="Config not found"):
            cli.cmd_submit(_submit_args(configs="missing"), settings, {})
        assert sbatch_calls == []

    def test_build_image_skips_existing_image_unless_forced(self, cli: Any, settings: Any, sbatch_calls: list[Any]):
        name = cli.image_name("3.13", b"lock-v1")
        (settings.shared_dir / "images" / name).write_bytes(b"")
        (settings.nss_dir / "mise.lock").write_text(
            '[[tools.uv]]\nversion = "0.9.30"\n[tools.uv."platforms.linux-x64"]\nchecksum = "sha256:abc"\nurl = "https://u"\n'
        )
        args = argparse.Namespace(partition="cpu_short", time="01:00:00", wait=False, force=False, dry_run=False)
        assert cli.cmd_build_image(args, settings, {}) == 0
        assert sbatch_calls == []

        args.force = True
        assert cli.cmd_build_image(args, settings, {}) == 0
        assert sbatch_calls[0].env["NSS_FORCE"] == "1"


def _settings_env(settings: Any) -> dict[str, str]:
    return {
        "USER_NAME": settings.user_name,
        "LUSTRE_DIR": str(settings.lustre_dir),
        "NSS_SHARED_DIR": str(settings.shared_dir),
        "NSS_DIR": str(settings.nss_dir),
        "NSS_SLURM_DIR": str(settings.slurm_dir),
        "CONFIG_DIR": str(settings.config_dir),
        "BASE_LOG_DIR": str(settings.base_log_dir),
        "NSS_IMAGE": str(settings.image),
        "NSS_BASE_IMAGE": str(settings.base_image),
        "NSS_MODELS_DIR": str(settings.models_dir),
        "NSS_DEFAULT_CONFIGS": ",".join(settings.default_configs),
    }


class TestSideEffects:
    """Process-launching helpers, with subprocess replaced by fakes."""

    def test_from_env_requires_paths(self, cli: Any, settings: Any):
        env = _settings_env(settings)
        del env["NSS_IMAGE"]
        with pytest.raises(cli.CliError, match="NSS_IMAGE"):
            cli.Settings.from_env(env)

    def test_read_image_metadata(self, cli: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            meta = Path(argv[argv.index("-d") + 1]) / "opt" / "nss-image"
            meta.mkdir(parents=True)
            (meta / ".python-version").write_text("3.13\n")
            (meta / "uv.lock").write_bytes(b"lock")
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        assert cli.read_image_metadata(tmp_path / "img.sqsh") == ("3.13\n", b"lock")

    def test_read_image_metadata_rejects_non_nss_image(self, cli: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", ""))
        with pytest.raises(cli.CliError, match="Could not read"):
            cli.read_image_metadata(tmp_path / "img.sqsh")

    def test_run_sbatch(self, cli: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        seen: dict[str, Any] = {}

        def fake_run(argv: list[str], *, env: dict[str, str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            seen["env"] = env
            return subprocess.CompletedProcess(argv, 0, "4242\n", "sbatch: note")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        call = cli.SbatchCall(argv=["sbatch"], env={"EXTRA": "1"})
        assert cli.run_sbatch(call, {"BASE": "x"}) == "4242"
        assert seen["env"] == {"BASE": "x", "EXTRA": "1"}
        assert "sbatch: note" in capsys.readouterr().err

        monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", ""))
        with pytest.raises(cli.CliError, match="sbatch failed"):
            cli.run_sbatch(call, {})

    def test_run_sbatch_wait(self, cli: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        class FakePopen:
            def __init__(self, argv: list[str], **kwargs: Any) -> None:
                self.stdout = io.StringIO("777\n")

            def wait(self) -> int:
                return 3

        monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)
        assert cli.run_sbatch_wait(cli.SbatchCall(argv=["sbatch"]), {}, "/logs/build_%j.out") == 3
        assert "Submitted job 777; log: /logs/build_777.out" in capsys.readouterr().out


class TestCommandPaths:
    @pytest.fixture
    def fakes(self, cli: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        state: dict[str, Any] = {"calls": [], "wait_code": 0, "lock": b"lock-v1"}

        def fake_run_sbatch(call: Any, base_env: Any) -> str:
            state["calls"].append(call)
            return "500"

        def fake_run_sbatch_wait(call: Any, base_env: Any, log_hint: str) -> int:
            state["calls"].append(call)
            if state.get("create") is not None:
                state["create"].write_bytes(b"")
            return state["wait_code"]

        monkeypatch.setattr(cli, "run_sbatch", fake_run_sbatch)
        monkeypatch.setattr(cli, "run_sbatch_wait", fake_run_sbatch_wait)
        monkeypatch.setattr(cli, "read_image_metadata", lambda image: ("3.13\n", state["lock"]))
        return state

    def test_main_runs_end_to_end_dry_run(
        self, cli: Any, settings: Any, fakes: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ):
        fakes["lock"] = b"other-lock"
        argv = ["submit", "-d", "adult", "-e", "dry", "--dry-run"]
        assert cli.main(argv, _settings_env(settings)) == 0
        out = capsys.readouterr().out
        assert "uv.lock differs" in out
        assert "Dry run completed without errors" in out
        assert "--test-only" in fakes["calls"][0].argv
        assert not (settings.base_log_dir / "dry").exists()

    def test_main_reports_errors(self, cli: Any, capsys: pytest.CaptureFixture[str]):
        assert cli.main(["submit", "-d", "adult"], {}) == 1
        assert "ERROR: USER_NAME is not set" in capsys.readouterr().err

    def test_submit_checks_tokens_and_image(self, cli: Any, settings: Any, fakes: dict[str, Any]):
        settings.image.unlink()
        with pytest.raises(cli.CliError, match="NSS image not found"):
            cli.cmd_submit(_submit_args(), settings, {})
        (settings.lustre_dir / ".api_tokens.sh").unlink()
        with pytest.raises(cli.CliError, match="api_tokens"):
            cli.cmd_submit(_submit_args(), settings, {})

    def test_submit_end_to_end(self, cli: Any, settings: Any, fakes: dict[str, Any]):
        assert cli.cmd_submit(_submit_args(configs="tiny", runs=3), settings, {}) == 0
        (call,) = fakes["calls"]
        assert "--export=ALL,NSS_PHASE=end_to_end" in call.argv
        assert call.env["PACKED_DATASETS"] == "adult,adult,adult"
        assert (settings.base_log_dir / "exp").is_dir()

    def _build_args(self, **overrides: Any) -> argparse.Namespace:
        values = {"partition": "cpu_short", "time": "01:00:00", "wait": False, "force": False, "dry_run": False}
        values.update(overrides)
        return argparse.Namespace(**values)

    @pytest.fixture
    def mise_lock(self, settings: Any) -> None:
        (settings.nss_dir / "mise.lock").write_text(
            '[[tools.uv]]\n[tools.uv."platforms.linux-x64"]\nchecksum = "sha256:abc"\nurl = "https://u"\n'
        )

    def test_build_image_paths(self, cli: Any, settings: Any, fakes: dict[str, Any], mise_lock: None):
        final_image = settings.shared_dir / "images" / cli.image_name("3.13", b"lock-v1")

        assert cli.cmd_build_image(self._build_args(dry_run=True), settings, {}) == 0
        assert fakes["calls"] == []

        assert cli.cmd_build_image(self._build_args(), settings, {}) == 0
        assert len(fakes["calls"]) == 1

        fakes["create"] = final_image
        assert cli.cmd_build_image(self._build_args(wait=True, force=True), settings, {}) == 0

        fakes["wait_code"] = 1
        with pytest.raises(cli.CliError, match="Image build failed"):
            cli.cmd_build_image(self._build_args(wait=True, force=True), settings, {})

        settings.base_image.unlink()
        with pytest.raises(cli.CliError, match="Base image not found"):
            cli.cmd_build_image(self._build_args(force=True), settings, {})

    def test_download_models_paths(
        self, cli: Any, settings: Any, fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ):
        args = argparse.Namespace(models=[], partition="cpu_short", wait=False)
        assert cli.cmd_download_models(args, settings, {}) == 0
        assert fakes["calls"][0].env["NSS_DOWNLOAD_MODELS"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert settings.models_dir.is_dir()

        args = argparse.Namespace(models=["Org/A"], partition="cpu_short", wait=True)
        assert cli.cmd_download_models(args, settings, {}) == 0
        fakes["wait_code"] = 1
        with pytest.raises(cli.CliError, match="Model download failed"):
            cli.cmd_download_models(args, settings, {})

        monkeypatch.setattr(cli.os, "access", lambda path, mode: False)
        with pytest.raises(cli.CliError, match="Cannot write"):
            cli.cmd_download_models(args, settings, {})
        monkeypatch.setattr(cli.os, "access", lambda path, mode: True)

        settings.image.unlink()
        with pytest.raises(cli.CliError, match="NSS image not found"):
            cli.cmd_download_models(args, settings, {})
        for config in settings.config_dir.glob("*.yaml"):
            config.unlink()
        with pytest.raises(cli.CliError, match="No models"):
            cli.cmd_download_models(argparse.Namespace(models=[], partition="p", wait=False), settings, {})
