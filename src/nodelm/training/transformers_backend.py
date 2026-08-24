from __future__ import annotations

import gc
import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodelm.training.lifecycle import TrainingLifecycleConfig


class TransformersSmokeSettings(BaseModel):
    """Fail-closed settings for a real Transformers/PEFT lifecycle smoke test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    padding_policy: Literal["require-existing", "reuse-eos", "add-token"]
    added_padding_token: str | None = None
    max_length: int = Field(default=512, ge=8)
    max_new_tokens: int = Field(default=16, ge=1)
    use_lora: bool = False
    target_modules: tuple[str, ...] = ()
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_architecture_decisions(self) -> TransformersSmokeSettings:
        if self.use_lora and not self.target_modules:
            raise ValueError("LoRA target_modules must be explicitly selected for the model")
        if self.padding_policy == "add-token" and not self.added_padding_token:
            raise ValueError("add-token padding policy requires added_padding_token")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU smoke tests require float32 unless explicitly revalidated")
        return self


class TransformersSmokeBackend:
    """Real one-batch causal-LM lifecycle backend.

    This backend intentionally has no architecture defaults. The caller must supply an exact
    commit, padding policy, device/dtype, and explicit LoRA targets when LoRA is enabled.
    """

    def __init__(self, settings: TransformersSmokeSettings) -> None:
        self.settings = settings
        self._torch: Any = None
        self._transformers: Any = None
        self._peft: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._batch: dict[str, Any] | None = None
        self._learning_rate: float | None = None
        self._optimizer: Any = None
        self.loss: float | None = None
        self.initial_loss: float | None = None
        self.resumed_loss: float | None = None
        self.forward_passes = 0
        self.backward_passes = 0
        self.optimizer_steps = 0
        self.resumed_optimizer_steps = 0
        self.optimizer_state_reloaded = False
        self.trainable_parameters: int | None = None
        self.total_parameters: int | None = None

    def load(self, config: TrainingLifecycleConfig) -> None:
        if config.model_id != self.settings.model_id or config.revision != self.settings.revision:
            raise ValueError("lifecycle config does not match pinned Transformers settings")
        self._torch = importlib.import_module("torch")
        self._transformers = importlib.import_module("transformers")
        if self.settings.use_lora:
            self._peft = importlib.import_module("peft")
        if self.settings.device == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self._torch.manual_seed(config.seed)
        if self.settings.device == "cuda":
            self._torch.cuda.manual_seed_all(config.seed)
            self._torch.cuda.reset_peak_memory_stats()
        self._learning_rate = config.learning_rate
        self._optimizer = None
        self.loss = None
        self.initial_loss = None
        self.resumed_loss = None
        self.forward_passes = 0
        self.backward_passes = 0
        self.optimizer_steps = 0
        self.resumed_optimizer_steps = 0
        self.optimizer_state_reloaded = False

        dtype = getattr(self._torch, self.settings.dtype)
        tokenizer = self._transformers.AutoTokenizer.from_pretrained(
            self.settings.model_id,
            revision=self.settings.revision,
            trust_remote_code=False,
        )
        model = self._transformers.AutoModelForCausalLM.from_pretrained(
            self.settings.model_id,
            revision=self.settings.revision,
            dtype=dtype,
            trust_remote_code=False,
        )
        model.to(self.settings.device)

        if tokenizer.pad_token_id is None:
            if self.settings.padding_policy == "require-existing":
                raise ValueError("tokenizer has no pad token and policy forbids changing it")
            if self.settings.padding_policy == "reuse-eos":
                if tokenizer.eos_token_id is None:
                    raise ValueError("tokenizer has neither pad nor EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                added = tokenizer.add_special_tokens(
                    {"pad_token": self.settings.added_padding_token}
                )
                if added != 1:
                    raise ValueError("expected exactly one padding token to be added")
                model.resize_token_embeddings(len(tokenizer))

        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            raise ValueError("tokenizer vocabulary and model embeddings are incompatible")

        if self.settings.use_lora:
            lora_config = self._peft.LoraConfig(
                task_type=self._peft.TaskType.CAUSAL_LM,
                r=self.settings.lora_rank,
                lora_alpha=self.settings.lora_alpha,
                lora_dropout=self.settings.lora_dropout,
                target_modules=list(self.settings.target_modules),
                bias="none",
            )
            model = self._peft.get_peft_model(
                model,
                lora_config,
                revision=self.settings.revision,
                low_cpu_mem_usage=False,
            )
            trainable, total = model.get_nb_trainable_parameters()
            if not 0 < trainable < total:
                raise ValueError("LoRA did not produce a strict trainable parameter subset")
            self.trainable_parameters = int(trainable)
            self.total_parameters = int(total)
        else:
            parameters = tuple(model.parameters())
            self.trainable_parameters = sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )
            self.total_parameters = sum(parameter.numel() for parameter in parameters)

        self._tokenizer = tokenizer
        self._model = model

    def tokenize(self, samples: tuple[str, ...]) -> None:
        self._require_loaded()
        if not samples or any(not sample.strip() for sample in samples):
            raise ValueError("training samples must be non-empty strings")
        self._tokenizer.padding_side = "right"
        batch = self._tokenizer(
            list(samples),
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        non_padding = batch["attention_mask"].sum(dim=1)
        if bool((non_padding < 2).any().item()):
            raise ValueError("each sample requires at least two non-padding tokens")
        moved = {name: tensor.to(self.settings.device) for name, tensor in batch.items()}
        labels = moved["input_ids"].clone()
        labels.masked_fill_(moved["attention_mask"] == 0, -100)
        moved["labels"] = labels
        self._batch = moved

    def train_step(self) -> None:
        self._require_loaded()
        if self.optimizer_steps != 0 or self._optimizer is not None:
            raise RuntimeError("the initial optimizer step has already run")
        self._optimizer = self._new_optimizer()
        self._run_optimizer_step(resumed=False)

    def save_checkpoint(self, path: Path) -> None:
        self._require_loaded()
        if self._optimizer is None or self.optimizer_steps != 1:
            raise RuntimeError("an optimizer step must complete before saving a checkpoint")
        path.mkdir(parents=True, exist_ok=False)
        self._model.save_pretrained(path / "model")
        self._tokenizer.save_pretrained(path / "tokenizer")
        optimizer_path = path / "optimizer.pt"
        self._torch.save(self._optimizer.state_dict(), optimizer_path)
        if not optimizer_path.is_file() or optimizer_path.stat().st_size == 0:
            raise RuntimeError("optimizer checkpoint was not written completely")

    def reload_checkpoint(self, path: Path) -> None:
        self._require_loaded()
        self._model = None
        self._optimizer = None
        self._tokenizer = None
        self.optimizer_state_reloaded = False
        gc.collect()
        if self.settings.device == "cuda":
            self._torch.cuda.empty_cache()
        dtype = getattr(self._torch, self.settings.dtype)
        tokenizer = self._transformers.AutoTokenizer.from_pretrained(path / "tokenizer")
        if self.settings.use_lora:
            saved = self._peft.PeftConfig.from_pretrained(path / "model")
            if saved.base_model_name_or_path != self.settings.model_id:
                raise ValueError("saved LoRA adapter references an unexpected base model")
            base = self._transformers.AutoModelForCausalLM.from_pretrained(
                self.settings.model_id,
                revision=self.settings.revision,
                dtype=dtype,
                trust_remote_code=False,
            )
            if len(tokenizer) != base.get_input_embeddings().weight.shape[0]:
                if self.settings.padding_policy != "add-token":
                    raise ValueError(
                        "reloaded tokenizer vocabulary and base embeddings are incompatible"
                    )
                base.resize_token_embeddings(len(tokenizer))
            base.to(self.settings.device)
            model = self._peft.PeftModel.from_pretrained(
                base,
                path / "model",
                is_trainable=True,
            )
        else:
            model = self._transformers.AutoModelForCausalLM.from_pretrained(
                path / "model", dtype=dtype, trust_remote_code=False
            )
            model.to(self.settings.device)
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            raise ValueError("reloaded tokenizer vocabulary and model embeddings are incompatible")
        model.to(self.settings.device)
        if self.settings.use_lora:
            trainable, total = model.get_nb_trainable_parameters()
        else:
            parameters = tuple(model.parameters())
            trainable = sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )
            total = sum(parameter.numel() for parameter in parameters)
        if not 0 < trainable <= total:
            raise ValueError("reloaded checkpoint has no trainable parameters")
        if (
            self.trainable_parameters is not None
            and self.total_parameters is not None
            and (int(trainable), int(total)) != (self.trainable_parameters, self.total_parameters)
        ):
            raise ValueError("reloaded checkpoint changed the trainable parameter set")
        self._tokenizer = tokenizer
        self._model = model
        optimizer_state = self._torch.load(
            path / "optimizer.pt",
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(optimizer_state, Mapping)
            or not isinstance(optimizer_state.get("state"), Mapping)
            or not optimizer_state["state"]
            or not isinstance(optimizer_state.get("param_groups"), list)
            or not optimizer_state["param_groups"]
        ):
            raise ValueError("optimizer checkpoint has an invalid state-dict structure")
        optimizer = self._new_optimizer()
        optimizer.load_state_dict(dict(optimizer_state))
        self._optimizer = optimizer
        self.optimizer_state_reloaded = True

    def resume_train_step(self) -> None:
        self._require_loaded()
        if self._optimizer is None or not self.optimizer_state_reloaded:
            raise RuntimeError("reload_checkpoint must restore optimizer state before resume")
        if self.resumed_optimizer_steps != 0 or self.optimizer_steps != 1:
            raise RuntimeError("the resumed optimizer step must run exactly once")
        self._run_optimizer_step(resumed=True)

    def infer(self, prompt: str) -> str:
        self._require_loaded()
        if not prompt.strip():
            raise ValueError("inference prompt must not be empty")
        self._tokenizer.padding_side = "left"
        batch = self._tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        moved = {name: tensor.to(self.settings.device) for name, tensor in batch.items()}
        self._model.eval()
        if not self._model.can_generate():
            raise ValueError("selected model cannot generate")
        generated = self._model.generate(
            **moved,
            max_new_tokens=self.settings.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        prompt_length = int(moved["input_ids"].shape[1])
        return cast(
            str,
            self._tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True),
        )

    def measured_evidence(self) -> dict[str, Any]:
        peak_gpu_memory_bytes: int | None = None
        if self._torch is not None and self.settings.device == "cuda":
            peak_gpu_memory_bytes = int(self._torch.cuda.max_memory_allocated())
        return {
            "loss": self.loss,
            "initial_loss": self.initial_loss,
            "resumed_loss": self.resumed_loss,
            "forward_passes": self.forward_passes,
            "backward_passes": self.backward_passes,
            "optimizer_steps": self.optimizer_steps,
            "resumed_optimizer_steps": self.resumed_optimizer_steps,
            "optimizer_state_reloaded": self.optimizer_state_reloaded,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        }

    def _new_optimizer(self) -> Any:
        if self._learning_rate is None:
            raise RuntimeError("load must establish the learning rate before training")
        trainable = [parameter for parameter in self._model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        return self._torch.optim.AdamW(trainable, lr=self._learning_rate)

    def _run_optimizer_step(self, *, resumed: bool) -> None:
        if self._batch is None:
            raise RuntimeError("tokenize must run before an optimizer step")
        if self._optimizer is None:
            raise RuntimeError("optimizer must be initialized before an optimizer step")
        self._model.train()
        trainable = [parameter for parameter in self._model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        self._optimizer.zero_grad(set_to_none=True)
        output = self._model(**self._batch)
        loss = output.loss
        if loss is None or not bool(self._torch.isfinite(loss).item()):
            raise ValueError("training loss is missing or non-finite")
        self.forward_passes += 1
        loss.backward()
        self.backward_passes += 1
        gradients = [parameter.grad for parameter in trainable]
        if any(gradient is None for gradient in gradients):
            raise ValueError("trainable gradients are missing or non-finite")
        finite_gradients = self._torch.stack(
            [self._torch.isfinite(gradient).all() for gradient in gradients]
        )
        if not bool(finite_gradients.all().item()):
            raise ValueError("trainable gradients are missing or non-finite")
        self._optimizer.step()
        observed_loss = float(loss.detach().cpu().item())
        self.loss = observed_loss
        self.optimizer_steps += 1
        if resumed:
            self.resumed_loss = observed_loss
            self.resumed_optimizer_steps += 1
        else:
            self.initial_loss = observed_loss

    def _require_loaded(self) -> None:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("load must complete before this stage")
