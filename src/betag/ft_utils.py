import json
import os
from typing import Literal
from pydantic import FilePath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cached_property
import naive_flow as nf
from .llm import DefaultLoraConfig, DefaultTrainingConfig


class BEFTConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        extra='forbid',
        frozen=True,
    )
    inters_path: FilePath
    base_tags_path: FilePath
    llm_name: str
    ckpt_dir: str
    max_seq_len: int
    min_seq_len: int
    n_tags_per_item: tuple[int, int] | tuple[float, float]

    num_epoch: int

    lora_config: DefaultLoraConfig
    training_config: DefaultTrainingConfig

    @field_validator('n_tags_per_item', mode='before')
    @classmethod
    def load_list(cls, v: str):
        if not isinstance(v, str):
            return v
        return json.loads(v)

    @cached_property
    def base_tags(self) -> dict[int, list[str]]:
        with open(self.base_tags_path, 'r', encoding='utf8') as fin:
            return {int(pid): tags for pid, tags in json.load(fin).items()}

    @cached_property
    def inters(self) -> list[list[int]]:
        with open(self.inters_path, 'r', encoding='utf8') as fin:
            return json.load(fin)

    def dump(self, path_or_dir: str):
        if os.path.isdir(path_or_dir):
            name = f'{self.__class__.__name__}.env'
            path_or_dir = os.path.join(path_or_dir, name)
        return nf.dump_config(self, path_or_dir)
