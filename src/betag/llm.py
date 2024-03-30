from typing import Literal, Callable
from pprint import pprint
import os
import re
import sys
import warnings
import abc

import torch
import torch.utils
import torch.utils.data

from pydantic import BaseModel, ConfigDict

from transformers import (
    Trainer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainerCallback, TrainerState, TrainingArguments, TrainerControl,
    BatchEncoding
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")
_re_checkpoint = re.compile(r"^ckpt(\-\S+)?-(\d+)$")


def get_last_checkpoint(folder, model_name: str | None = None):
    content = os.listdir(folder)

    def is_valid_ckpt(path: str):
        match = _re_checkpoint.search(path)
        if match is not None and os.path.isdir(os.path.join(folder, path)):
            if model_name is None:
                return True
            if match.group(1) == f'-{model_name}':
                return True
        return False

    checkpoints = list(filter(is_valid_ckpt, content))
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).group(2)))
    )


class _WrapDataset(torch.utils.data.Dataset):

    def __init__(self, ref_dataset: torch.utils.data.Dataset, hook):
        self.ref = ref_dataset
        self.hook = hook
        return

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        return self.hook(self.ref[index])


class SaveCheckpointCallback(TrainerCallback):
    """Save checkpoint on epoch end.
    Note the normal checkpoints saved by trainer are buggy, as they cannot used to
    resume training.

    """

    def __init__(
        self,
        model,
        ckpts_root_dir: str,
        ckpt_name_template: str,
        cur_epoch: int,
        further_callbacks: list[Callable] | None = None,
    ):
        self.model = model
        self.ckpts_root_dir = ckpts_root_dir
        self.cur_epoch = cur_epoch
        self.ckpt_name_template = ckpt_name_template
        self.further_callbacks = further_callbacks or []
        return

    def save(self, state: TrainerState, comment: str | None = None):
        ckpt_dir = os.path.join(
            self.ckpts_root_dir,
            self.ckpt_name_template.format(epoch=self.cur_epoch)
        )
        if comment is not None:
            ckpt_dir += comment
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
        print(f"A new checkpoint has been saved in {ckpt_dir}")
        for further_callback in self.further_callbacks:
            further_callback(ckpt_dir)
        return

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.cur_epoch += 1
        self.save(state)
        return


class DefaultLoraConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = [
        "q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj",
        "v_proj"
    ]
    bias: Literal["none"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"


PADDING_MAX_LEN = 1024


class DefaultTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    save_strategy: Literal["no", "steps"] = "steps"
    fp16: bool = False
    """May cause overflow."""

    warmup_steps: int = 0
    save_steps: int = 100
    logging_steps: int = 100
    save_total_limit: int = 10
    evaluation_strategy: Literal["epoch", "no"] | None = None
    max_steps: int | None = None


class _Base(abc.ABC):

    @property
    @abc.abstractmethod
    def PADDING_MAX_LEN(self) -> int:
        ...

    def __init__(
        self,
        model_name: str,
        instruction: str | Callable,
        ckpt_name: str = None,
        inference_only: bool = True,
        step_ckpt_dir: str = None,
        epoch_ckpt_dir: str = None,
        cache_dir: str = ".cache",
        add_eos_token: bool | None = None,
    ):
        self.model_name = model_name
        self.model_basename = os.path.basename(model_name)
        self.instruction = instruction
        self.cache_dir = cache_dir
        self.step_ckpt_dir = step_ckpt_dir
        # NOTE: for some weird reason, continuing training from step_ckpt results in
        # parameters reinitialization.
        # But here, we still save them for logs (and they can still be used for inference)
        self.epoch_ckpt_dir = epoch_ckpt_dir
        if epoch_ckpt_dir is not None and not os.path.exists(epoch_ckpt_dir):
            print(f"'{epoch_ckpt_dir}' does not exist, thus creating..")
            os.makedirs(epoch_ckpt_dir)
        self.ckpt_name = ckpt_name or get_last_checkpoint(
            epoch_ckpt_dir, self.model_basename
        )
        if self.ckpt_name is None:
            self.cur_ckpt_count = 0
        else:
            cur_ckpt = _re_checkpoint.search(os.path.basename(self.ckpt_name))
            self.cur_ckpt_count = cur_ckpt and int(cur_ckpt.group(2))
        # self.prompt_template = PROMPT
        self.inference_only = inference_only
        self.add_eos_token = add_eos_token
        self._init(model_name)
        return

    def _init(self, model_name: str):

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            quantization_config=nf4_config,
            # low_cpu_mem_usage=True
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True,
            cache_dir=self.cache_dir,
            quantization_config=nf4_config,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # NOTE: llama3 does use pad_id = -1, since it uses only mask.
        # https://github.com/meta-llama/llama3/issues/42#issuecomment-2066943089
        # Here the pad_token is set as eos_token, but they need to be processed later.

        if self.inference_only:
            assert self.ckpt_name is not None, (
                "the inference_only=True is passed. However, no ckpt_name is provided"
            )
            print(f"loading checkpoint from {self.ckpt_name}")
            self.model = PeftModel.from_pretrained(
                self.model, self.ckpt_name, is_trainable=False
            )
        return

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        num_epoch: int,
        training_config: DefaultTrainingConfig,
        val_dataset: torch.utils.data.Dataset = None,
        lora_config: DefaultLoraConfig = None,
        add_eos_token: bool | None = None,
        save_checkpoint_callbacks: list[Callable] | None = None,
    ):
        # pylint: disable=attribute-defined-outside-init

        if not val_dataset:
            assert getattr(training_config, "evaluation_strategy", None) in [
                None, "no"
            ], (
                "Please explictly turn off evaulation_stratey if val_dataset is not used."
            )
        assert self.step_ckpt_dir is not None
        assert self.epoch_ckpt_dir is not None
        assert self.cur_ckpt_count is not None
        if add_eos_token is not None:
            self.add_eos_token = add_eos_token
        else:
            assert self.add_eos_token is not None, (
                "add_eos_token is not set. This determine whether"
                " add eos_token after outputs during training."
                " set add_eos_token=False lead to non-stop generation"
                ", which may be a desired property in our case."
            )
        os.makedirs(self.step_ckpt_dir, exist_ok=True)

        self.model = prepare_model_for_kbit_training(self.model)
        if self.ckpt_name is None:
            if lora_config is None:
                lora_config = DefaultLoraConfig()
            print("LoraConfig:")
            pprint(lora_config.model_dump())
            self.model = get_peft_model(
                self.model, LoraConfig(**lora_config.model_dump())
            )
            print(
                "Warning: no ckpt_name found. The finetuning will start from scratch."
            )
        else:
            print(f"loading checkpoint from {self.ckpt_name}")
            self.model = PeftModel.from_pretrained(
                self.model, self.ckpt_name, is_trainable=True
            )

        self.model.print_trainable_parameters()
        print("TrainingConfig:")
        pprint(training_config.model_dump())

        save_checkpoint_callback = SaveCheckpointCallback(
            self.model,
            self.epoch_ckpt_dir,
            f"ckpt-{self.model_basename}-{{epoch}}",
            self.cur_ckpt_count,
            further_callbacks=save_checkpoint_callbacks,
        )
        trainer = Trainer(
            model=self.model,
            train_dataset=_WrapDataset(train_dataset, self._pre_collate),
            eval_dataset=(
                _WrapDataset(val_dataset, self._pre_collate)
                if val_dataset else None
            ),
            callbacks=[save_checkpoint_callback],
            args=TrainingArguments(
                output_dir=self.step_ckpt_dir,
                ddp_find_unused_parameters=None,
                num_train_epochs=num_epoch,
                **training_config.model_dump(),
                # max_steps=10,
            ),
        )

        self.model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        try:
            trainer.train()
        except KeyboardInterrupt as e:
            save_checkpoint_callback.save(trainer.state, comment="-aborted")
            raise e

        return

    @abc.abstractmethod
    def _pre_collate(self, data_point) -> dict:
        ...

    def eval(self, dataset: list, candidates: list[str], batch_size: int):
        self.model.eval()
        PADDING_SIDE = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        candidates = self.tokenizer(
            [f" {c}" for c in candidates],  # add leading space
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = PADDING_SIDE
        candidates["labels"] = candidates["input_ids"].clone()
        candidates["labels"][candidates["labels"] ==
                             self.tokenizer.pad_token_id] = -100

        for data_point in dataset:
            tag_loss = self.eval_sample(
                data_point, candidates, batch_size=batch_size
            )
            yield data_point, tag_loss
        return

    @torch.no_grad()
    def eval_sample(
        self,
        data_point: dict,
        candidates: list[str] | dict[str, torch.Tensor],
        batch_size: int,
    ):
        self.model.eval()
        if not isinstance(candidates, (dict, BatchEncoding)):
            PADDING_SIDE = self.tokenizer.padding_side
            self.tokenizer.padding_side = "right"
            candidates = self.tokenizer(
                candidates,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = PADDING_SIDE

        prompt = self.instruction(self.tokenizer, data_point)
        input_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_tokens["labels"] = torch.full_like(
            input_tokens["input_ids"], -100
        )
        input_tokens = input_tokens.to("cuda")
        bos_candidate = {k: v[:, -1:] for k, v in input_tokens.items()}
        input_tokens = {k: v[:, :-1] for k, v in input_tokens.items()}
        input_cache = self.model.forward(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            use_cache=True,
            # past_key_values=tuple(handle_cache(cache)),
        ).past_key_values
        input_cache = tuple(
            (k.repeat(batch_size, 1, 1, 1), v.repeat(batch_size, 1, 1, 1))
            for k, v in input_cache
        )

        def cal_losses(candidate_tokens: dict):
            nonlocal input_cache
            candidate_tokens = {
                k: v.cuda()
                for k, v in candidate_tokens.items()
            }
            this_batch_size = len(candidate_tokens["input_ids"])
            if this_batch_size < batch_size:
                input_cache = tuple(
                    (k[:this_batch_size], v[:this_batch_size])
                    for k, v in input_cache
                )
            batch = {}
            for key, candidate_data in candidate_tokens.items():
                batch[key] = torch.concat(
                    [
                        bos_candidate[key].repeat(len(candidate_data), 1),
                        candidate_data,
                    ], dim=1
                )
            labels = batch.pop("labels")
            attention_mask = torch.concat(
                [
                    input_tokens["attention_mask"].repeat(this_batch_size, 1),
                    batch["attention_mask"],
                ], dim=1
            )
            out = self.model.forward(
                input_ids=batch["input_ids"],
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=input_cache,
            )
            logits = out["logits"]
            # following model wrapper in huggingface
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            losses = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                reduction="none",
            )
            losses_ = losses.view(this_batch_size, -1).sum(dim=1)
            return losses_.cpu()

        def batch_forward():
            for i in range(0, len(candidates["input_ids"]), batch_size):
                yield cal_losses(
                    {
                        k: v[i:i + batch_size]
                        for k, v in candidates.items()
                    }
                )

        return torch.concat(list(batch_forward()), dim=0)

    def _inference(
        self,
        data_points: list[dict],
        generation_config,
        **_kwargs,
    ):
        prompts = [self.instruction(self.tokenizer, d) for d in data_points]
        assert self.tokenizer.padding_side == "left"
        inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].cuda()
        if generation_config.num_beams > 1:
            assert generation_config.num_return_sequences is not None

        generation_output = self.model.generate(
            input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        raw_outputs = self.tokenizer.batch_decode(generation_output.sequences)
        batch_size = len(generation_output.sequences) // len(data_points)
        outputs = [
            raw.split(self.tokenizer.bos_token, maxsplit=1)[1]
            for raw in raw_outputs
        ]
        return [
            outputs[i:(i + batch_size)]
            for i in range(0, len(outputs), batch_size)
        ]

    def inference(
        self,
        data_point: dict,
        generation_config,
        **kwargs,
    ):
        """inference a single sample"""
        return self._inference([data_point], generation_config, **kwargs)[0]

    def batch_inference(
        self,
        dataset: list[dict],
        generation_config,
        batch_size,
        **kwargs,
    ):
        """inference a dataset by batch"""
        assert self.inference_only is True

        for i in range(0, len(dataset), batch_size):
            batch = [
                dataset[i]
                for i in range(i, min(i + batch_size, len(dataset)))
            ]
            res = self._inference(
                batch,
                generation_config,
                **kwargs,
            )
            for data_point, results in zip(batch, res):
                yield data_point, results


class LlamaInstruct(_Base):

    PADDING_MAX_LEN: int = 1024

    def _pre_collate(self, data_point):
        # construct full input prompt
        # use bos as sep token
        input_prompt, generation_prompt = self.instruction(
            self.tokenizer,
            data_point,
            include_generation_prompt=False,
        )
        example = (
            input_prompt + self.tokenizer.bos_token + generation_prompt +
            data_point["output"]
        )
        assert self.add_eos_token is not None
        if self.add_eos_token:
            example += self.tokenizer.eos_token

        full_tokens = self.tokenizer(
            example,
            truncation=True,
            max_length=self.PADDING_MAX_LEN + 1,
            padding="max_length",
            add_special_tokens=False,
        )
        bos_idx = full_tokens["input_ids"].index(self.tokenizer.bos_token_id)
        sep_idx = full_tokens["input_ids"].index(
            self.tokenizer.bos_token_id, bos_idx + 1
        )
        assert sep_idx, f"should not used id '{self.tokenizer.bos_token_id}' for padding!"
        for vals in full_tokens.values():
            vals.pop(sep_idx)
        if full_tokens["input_ids"][0] == self.tokenizer.bos_token_id:
            warnings.warn(
                f"An input may have been truncated due to {self.PADDING_MAX_LEN = }"
            )

        full_tokens["labels"] = (
            [-100] * sep_idx + full_tokens["input_ids"].copy()[sep_idx:]
        )
        return full_tokens

    def _inference(
        self,
        data_points: list[dict],
        generation_config,
        **kwargs,
    ):
        to_remove_tokens = [
            r"<\|eot_id\|>",
            r"<\|begin_of_text\|>",
            r"<\|end_of_text\|>",
        ]

        assert "generation_sep" in kwargs

        def post_process(out: str):
            out = out.split(kwargs["generation_sep"], maxsplit=1)
            assert len(out) == 2
            out = out[1]
            for token in to_remove_tokens:
                out = re.sub(token, "", out, flags=re.IGNORECASE)
            return out

        return [
            [post_process(o) for o in outputs]
            for outputs in super()._inference(data_points, generation_config)
        ]
