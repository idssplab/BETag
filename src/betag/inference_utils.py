import warnings
import re
import json
import hashlib
import os
from pydantic import FilePath, field_validator, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cached_property


class GenerationConfig(BaseSettings):
    # same as transformers.GenerationConfig

    model_config = SettingsConfigDict(extra='allow', frozen=True)
    do_sample: bool = False,
    temperature: float = 4.
    top_p: float = 0.95
    top_k: float = 25
    no_repeat_ngram_size: int = 0

    # multi-beam args
    num_beams: int
    num_beam_groups: int
    num_return_sequences: int
    diversity_penalty: float = 1.

    # Length of the generated text
    min_new_tokens: int
    max_new_tokens: int
    # pad_token_id = llm.tokenizer.pad_token_id

    @property
    def hash(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:8]


class BEInferenceConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        extra='forbid',
        frozen=True,
    )
    results_dir: str
    max_n_tags: int = 15
    base_tags_path: FilePath
    batch_size: int = 1
    ckpt_name: DirectoryPath
    llm_name: str
    sep: str = '| '

    generation_config: GenerationConfig

    @cached_property
    def base_tags(self) -> dict[int, list[str]]:
        with open(self.base_tags_path, 'r', encoding='utf8') as fin:
            return {int(pid): tags for pid, tags in json.load(fin).items()}

    @field_validator('results_dir', mode='after')
    @classmethod
    def mkdirs(cls, v: str):
        os.makedirs(v, exist_ok=True)
        return v


def parse_raw_predict(raw_predict_file: str, n_beams, sep):

    with open(raw_predict_file, 'r', encoding='utf8') as fin:
        raw_predicts = {
            int(pid): lines
            for pid, lines in json.load(fin).items()
        }

    line_2_r = re.compile(r'\n2\.\s*?([^\n]+)')

    def parse(raws: list[str]):
        lines = [re.search(r'1\. ([^\n]+)', raws[0]).group(1)]
        assert len(raws) >= n_beams, (f'{len(raws) = }, {n_beams = }')
        for raw in raws[:n_beams]:
            match = line_2_r.search(raw)
            if match is not None:
                lines.append(match.group(1))
            else:
                warnings.warn(
                    f'line: {repr(raw)} does not seem to have line 2'
                )
        return lines

    raw_tags = {}
    for pid, raw_predict in raw_predicts.items():
        raw_predict: list[str]
        lines = parse(raw_predict)
        tags = []
        for line in lines:
            tags.append(
                [tag.strip() for tag in line.split(sep) if tag.strip()]
            )

        raw_tags[pid] = tags

    return raw_tags
