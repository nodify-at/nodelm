from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from nodelm.models import VerificationStatus
from nodelm.training.config import TrainingSmokeConfig, load_training_smoke_config
from nodelm.training.lifecycle import TrainingLifecycleConfig, run_training_lifecycle
from nodelm.training.transformers_backend import (
    TransformersSmokeBackend,
    TransformersSmokeSettings,
)


class FakeBackend:
    def __init__(self, *, fail_at: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} failed")

    def load(self, config: TrainingLifecycleConfig) -> None:
        self._call("load")

    def tokenize(self, samples: tuple[str, ...]) -> None:
        self._call("tokenize")

    def train_step(self) -> None:
        self._call("train_step")

    def save_checkpoint(self, path: Path) -> None:
        self._call("save_checkpoint")

    def reload_checkpoint(self, path: Path) -> None:
        self._call("reload_checkpoint")

    def resume_train_step(self) -> None:
        self._call("resume_train_step")

    def infer(self, prompt: str) -> str:
        self._call("infer")
        return "ok"


def test_training_lifecycle_exercises_every_required_stage(tmp_path: Path) -> None:
    backend = FakeBackend()
    config = TrainingLifecycleConfig(
        model_id="owner/model",
        revision="a" * 40,
        output_dir=tmp_path / "checkpoint",
        seed=7,
    )

    report = run_training_lifecycle(backend, config, samples=("sample",), prompt="hello")

    assert report.status is VerificationStatus.PASS
    assert backend.calls == [
        "load",
        "tokenize",
        "train_step",
        "save_checkpoint",
        "reload_checkpoint",
        "resume_train_step",
        "infer",
    ]
    assert set(report.stage_durations_seconds) == set(backend.calls)
    assert report.resumed_optimizer_step_completed is True
    assert report.wall_clock_seconds >= 0


def test_training_lifecycle_stops_and_reports_failed_stage(tmp_path: Path) -> None:
    backend = FakeBackend(fail_at="train_step")
    config = TrainingLifecycleConfig(
        model_id="owner/model",
        revision="a" * 40,
        output_dir=tmp_path / "checkpoint",
    )

    report = run_training_lifecycle(backend, config, samples=("sample",), prompt="hello")

    assert report.status is VerificationStatus.FAIL
    assert report.failed_stage == "train_step"
    assert backend.calls == ["load", "tokenize", "train_step"]
    assert report.resumed_optimizer_step_completed is False


def test_training_lifecycle_reports_a_failed_resumed_optimizer_step(tmp_path: Path) -> None:
    backend = FakeBackend(fail_at="resume_train_step")
    config = TrainingLifecycleConfig(
        model_id="owner/model",
        revision="a" * 40,
        output_dir=tmp_path / "checkpoint",
    )

    report = run_training_lifecycle(backend, config, samples=("sample",), prompt="hello")

    assert report.status is VerificationStatus.FAIL
    assert report.failed_stage == "resume_train_step"
    assert report.resumed_optimizer_step_completed is False
    assert backend.calls[-2:] == ["reload_checkpoint", "resume_train_step"]


def test_transformers_smoke_requires_immutable_model_revision() -> None:
    with pytest.raises(ValidationError, match="revision"):
        TransformersSmokeSettings(
            model_id="owner/model",
            revision="main",
            padding_policy="require-existing",
        )


def test_lora_smoke_requires_explicit_target_modules() -> None:
    with pytest.raises(ValidationError, match="target_modules"):
        TransformersSmokeSettings(
            model_id="owner/model",
            revision="a" * 40,
            padding_policy="require-existing",
            use_lora=True,
        )


def test_checked_in_training_smoke_config_matches_strict_contract() -> None:
    config = load_training_smoke_config(Path("configs/training/tiny-lora.yaml"))

    assert config.status is VerificationStatus.UNVERIFIED
    assert config.max_steps == 1
    assert config.checkpoint.reload is True
    assert config.checkpoint.resume is True


def test_checked_in_pilot_lora_config_matches_strict_contract() -> None:
    config = load_training_smoke_config(Path("configs/training/pilot-lora.yaml"))

    assert config.status is VerificationStatus.UNVERIFIED
    assert config.purpose == "pilot LoRA"
    assert config.model.repository_id is None
    assert config.model.revision is None
    assert config.model.license is None
    assert config.runtime is None
    assert config.precision is None
    assert config.max_steps == 1
    assert config.batch_size == 1
    assert config.gradient_accumulation_steps == 1
    assert config.checkpoint.save is True
    assert config.checkpoint.reload is True
    assert config.checkpoint.resume is True
    assert config.inference_after_reload is True


def test_training_smoke_config_forbids_unknown_fields() -> None:
    payload = load_training_smoke_config(Path("configs/training/tiny-lora.yaml")).model_dump(
        mode="json"
    )
    payload["unexpected"] = "ignored settings are unsafe"

    with pytest.raises(ValidationError, match="unexpected"):
        TrainingSmokeConfig.model_validate(payload)


def test_training_smoke_config_requires_explicit_checkpoint_resume() -> None:
    payload = load_training_smoke_config(Path("configs/training/tiny-lora.yaml")).model_dump(
        mode="json"
    )
    del payload["checkpoint"]["resume"]

    with pytest.raises(ValidationError, match="resume"):
        TrainingSmokeConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", VerificationStatus.FAIL),
        ("checkpoint", {"save": False, "reload": True, "resume": True}),
        ("checkpoint", {"save": True, "reload": False, "resume": True}),
        ("checkpoint", {"save": True, "reload": True, "resume": False}),
        ("inference_after_reload", False),
    ),
)
def test_training_config_is_not_runnable_when_a_required_gate_is_disabled(
    field: str,
    value: object,
) -> None:
    payload = {
        **load_training_smoke_config(Path("configs/training/tiny-lora.yaml")).model_dump(
            mode="json"
        ),
        "status": VerificationStatus.UNVERIFIED,
        "model": {
            "repository_id": "owner/model",
            "revision": "a" * 40,
            "license": "Apache-2.0",
        },
        "runtime": {
            "backend": "transformers-peft",
            "device": "cpu",
            "padding_policy": "require-existing",
            "max_length": 128,
            "max_new_tokens": 16,
            "use_lora": False,
            "target_modules": [],
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
        },
        "precision": "float32",
        field: value,
    }

    assert TrainingSmokeConfig.model_validate(payload).is_runnable is False


def test_training_model_license_must_not_be_empty() -> None:
    payload = load_training_smoke_config(Path("configs/training/tiny-lora.yaml")).model_dump(
        mode="json"
    )
    payload["model"]["license"] = ""

    with pytest.raises(ValidationError, match="license"):
        TrainingSmokeConfig.model_validate(payload)


def test_checkpoint_save_includes_optimizer_state(tmp_path: Path) -> None:
    events: dict[str, Any] = {}

    class Savable:
        def save_pretrained(self, path: Path) -> None:
            path.mkdir()

    class Optimizer:
        def state_dict(self) -> dict[str, Any]:
            return {"state": {0: {"step": 1}}, "param_groups": [{"params": [0]}]}

    class Torch:
        @staticmethod
        def save(value: object, path: Path) -> None:
            events["saved_value"] = value
            events["saved_path"] = path
            path.write_bytes(b"optimizer-state")

    backend = TransformersSmokeBackend(
        TransformersSmokeSettings(
            model_id="owner/model",
            revision="a" * 40,
            padding_policy="require-existing",
        )
    )
    backend._torch = Torch()
    backend._model = Savable()
    backend._tokenizer = Savable()
    backend._optimizer = Optimizer()
    backend.optimizer_steps = 1

    checkpoint = tmp_path / "checkpoint"
    backend.save_checkpoint(checkpoint)

    assert events["saved_path"] == checkpoint / "optimizer.pt"
    assert events["saved_value"] == backend._optimizer.state_dict()


def test_lora_checkpoint_reload_is_trainable_and_uses_weights_only_optimizer_state(
    tmp_path: Path,
) -> None:
    events: dict[str, Any] = {}
    optimizer_state = {"state": {0: {"step": 1}}, "param_groups": [{"params": [0]}]}

    class Parameter:
        requires_grad = True

        def numel(self) -> int:
            return 2

    class Weight:
        shape = (3, 4)

    class Embeddings:
        weight = Weight()

    class Model:
        def __init__(self) -> None:
            self.parameter = Parameter()

        def get_input_embeddings(self) -> Embeddings:
            return Embeddings()

        def get_nb_trainable_parameters(self) -> tuple[int, int]:
            return 2, 10

        def parameters(self) -> tuple[Parameter, ...]:
            return (self.parameter,)

        def to(self, device: str) -> Model:
            events["model_device"] = device
            return self

    class Tokenizer:
        def __len__(self) -> int:
            return 3

    class Optimizer:
        def load_state_dict(self, value: object) -> None:
            events["loaded_optimizer_state"] = value

    optimizer = Optimizer()

    class Optim:
        @staticmethod
        def AdamW(parameters: list[Parameter], *, lr: float) -> Optimizer:
            events["optimizer_parameter_count"] = len(parameters)
            events["optimizer_learning_rate"] = lr
            return optimizer

    class Torch:
        float32 = object()
        optim = Optim()

        @staticmethod
        def load(path: Path, *, map_location: str, weights_only: bool) -> object:
            events["loaded_path"] = path
            events["map_location"] = map_location
            events["weights_only"] = weights_only
            return optimizer_state

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path) -> Tokenizer:
            return Tokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> Model:
            events["released_before_model_load"] = (
                backend._model is None and backend._optimizer is None
            )
            return Model()

    class SavedPeftConfig:
        base_model_name_or_path = "owner/model"
        revision = "a" * 40

    class PeftConfig:
        @staticmethod
        def from_pretrained(path: Path) -> SavedPeftConfig:
            return SavedPeftConfig()

    class PeftModel:
        @staticmethod
        def from_pretrained(base: Model, path: Path, *, is_trainable: bool) -> Model:
            events["is_trainable"] = is_trainable
            return base

    backend = TransformersSmokeBackend(
        TransformersSmokeSettings(
            model_id="owner/model",
            revision="a" * 40,
            padding_policy="require-existing",
            use_lora=True,
            target_modules=("q_proj",),
        )
    )
    backend._torch = Torch()
    backend._transformers = SimpleNamespace(
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
    )
    backend._peft = SimpleNamespace(PeftConfig=PeftConfig, PeftModel=PeftModel)
    backend._tokenizer = Tokenizer()
    backend._model = Model()
    backend._optimizer = object()
    backend._learning_rate = 0.001
    backend.trainable_parameters = 2
    backend.total_parameters = 10
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"safe fake state")

    backend.reload_checkpoint(checkpoint)

    assert events["is_trainable"] is True
    assert events["released_before_model_load"] is True
    assert events["weights_only"] is True
    assert events["map_location"] == "cpu"
    assert events["loaded_optimizer_state"] == optimizer_state
    assert backend.optimizer_state_reloaded is True


def test_backend_runs_a_measured_optimizer_step_after_reload() -> None:
    class Truth:
        def all(self) -> Truth:
            return self

        def item(self) -> bool:
            return True

    class Parameter:
        requires_grad = True
        grad: Truth | None = None

    parameter = Parameter()

    class Loss:
        def __init__(self, value: float) -> None:
            self.value = value

        def backward(self) -> None:
            parameter.grad = Truth()

        def detach(self) -> Loss:
            return self

        def cpu(self) -> Loss:
            return self

        def item(self) -> float:
            return self.value

    losses = iter((1.0, 0.5))

    class Model:
        def parameters(self) -> tuple[Parameter, ...]:
            return (parameter,)

        def train(self) -> None:
            pass

        def __call__(self, **batch: object) -> SimpleNamespace:
            assert batch
            return SimpleNamespace(loss=Loss(next(losses)))

    class Optimizer:
        def __init__(self) -> None:
            self.steps = 0

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            parameter.grad = None

        def step(self) -> None:
            self.steps += 1

    optimizer = Optimizer()

    class Optim:
        @staticmethod
        def AdamW(parameters: list[Parameter], *, lr: float) -> Optimizer:
            assert parameters == [parameter]
            assert lr == 0.001
            return optimizer

    class Torch:
        optim = Optim()

        @staticmethod
        def isfinite(value: object) -> Truth:
            return Truth()

        @staticmethod
        def stack(values: list[Truth]) -> Truth:
            assert values
            return Truth()

    backend = TransformersSmokeBackend(
        TransformersSmokeSettings(
            model_id="owner/model",
            revision="a" * 40,
            padding_policy="require-existing",
        )
    )
    backend._torch = Torch()
    backend._model = Model()
    backend._tokenizer = object()
    backend._batch = {"input_ids": object()}
    backend._learning_rate = 0.001

    backend.train_step()
    backend.optimizer_state_reloaded = True
    backend.resume_train_step()

    assert optimizer.steps == 2
    expected = {
        "loss": 0.5,
        "initial_loss": 1.0,
        "resumed_loss": 0.5,
        "forward_passes": 2,
        "backward_passes": 2,
        "optimizer_steps": 2,
        "resumed_optimizer_steps": 1,
        "optimizer_state_reloaded": True,
    }
    evidence = backend.measured_evidence()
    assert {key: evidence[key] for key in expected} == expected
