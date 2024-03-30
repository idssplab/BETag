import argparse
from typing import Callable
import json
import os

from transformers import GenerationConfig
from tqdm import tqdm
import naive_flow as nf

from betag.llm import LlamaInstruct
from betag.dataset import InferenceDataset
from betag.inference_utils import BEInferenceConfig, parse_raw_predict
from betag.instructions import llama3_instruct_en_in_context


def main():

    arg_parser = argparse.ArgumentParser('BETag-Generation')
    arg_parser.add_argument(
        '-e',
        '--env',
        type=str,
        required=True,
        help='path to the .env file',
    )
    arg_parser.add_argument(
        '--check',
        action='store_true',
        help='check whether the config is valid',
    )
    args = arg_parser.parse_args()
    env_path = args.env
    assert os.path.isfile(env_path), 'No env file found'

    env_data = {
        field: val
        for field, val in nf.load_env_file(env_path).items()
        if not field.startswith('_')
    }

    config = BEInferenceConfig.model_validate_strings(env_data)
    print(nf.strfconfig(config))

    if args.check:
        return

    llm = LlamaInstruct(
        model_name=config.llm_name,
        instruction=llama3_instruct_en_in_context,
        ckpt_name=config.ckpt_name,
        inference_only=True,
    )

    dataset = InferenceDataset(
        base_tags=config.base_tags,
        max_n_tags_per_item=config.max_n_tags,
        sep=config.sep,
    )

    def display_predictions(data_point: dict, predicts: list[str], file=None):
        # multi-beam
        pid = data_point["pid"]
        assert len(predicts) > 1
        for j, predict in enumerate(predicts):
            print(f'## prediction {pid}.{j}:\n{predict}', file=file)
        return

    log_dir = os.path.join(
        config.results_dir,
        f'ckpt-{llm.model_basename}-{llm.cur_ckpt_count}/{config.generation_config.hash}'
    )

    generation_config = GenerationConfig(
        **config.generation_config.model_dump(),
        pad_token_id=llm.tokenizer.pad_token_id,
    )
    generate(
        llm,
        inference_dataset=dataset,
        generation_config=generation_config,
        log_dir=log_dir,
        batch_size=max(
            int(config.batch_size // config.generation_config.num_beams), 1
        ),
        display_fn=display_predictions,
        force_restart=False,
    )

    results = parse_raw_predict(
        os.path.join(log_dir, 'raw_predict.json'),
        n_beams=config.generation_config.num_beams,
        sep=config.sep,
    )
    results: dict[int, list[list[str]]]
    # pid: [tags@b for b in beams]
    with open(
        os.path.join(log_dir, 'raw_betags.json'), 'w', encoding='utf8'
    ) as fout:
        json.dump(results, fout)

    return


def generate(
    llm: LlamaInstruct,
    generation_config,
    inference_dataset: InferenceDataset,
    batch_size: int,
    log_dir: str,
    display_fn: Callable | None = None,
    force_restart: bool = False,
):

    def save(obj: dict[int, str], path: str):
        with open(path, 'w', encoding='utf8') as fout:
            json.dump(obj, fout, ensure_ascii=False, indent=4)
        return

    N_PER_SAVE = 10
    OUT_PREDICTION_TEMP = os.path.join(log_dir, 'raw_predict_temp.json')
    OUT_PREDICTION = os.path.join(log_dir, 'raw_predict.json')
    OUT_GENERATION_CONFIG = os.path.join(log_dir, 'generation_config.json')
    os.makedirs(log_dir, exist_ok=True)
    save(generation_config.to_dict(), OUT_GENERATION_CONFIG)
    assert not os.path.exists(OUT_PREDICTION), OUT_PREDICTION
    print(f'Generation will output in {log_dir}')
    if os.path.isfile(OUT_PREDICTION_TEMP) and not force_restart:
        with open(OUT_PREDICTION_TEMP, 'r', encoding='utf8') as fin:
            inference_results = {
                int(pid): res
                for pid, res in json.load(fin).items()
            }
    else:
        inference_results = {}

    inference_dataset = [
        d for d in inference_dataset if d['pid'] not in inference_results
    ]
    pbar = tqdm(
        enumerate(
            llm.batch_inference(
                inference_dataset,
                generation_config,
                batch_size,
                generation_sep=llama3_instruct_en_in_context.GENERATION_SEP,
            )
        ), total=len(inference_dataset)
    )
    for i, (d, predicts) in pbar:
        pbar.set_description(f'pid: {d["pid"]}')
        inference_results[d['pid']] = \
            predicts[0] if len(predicts) == 1 else predicts
        if i % N_PER_SAVE:
            save(inference_results, OUT_PREDICTION_TEMP)
        if i < 10:
            display_fn(d, predicts, file=pbar)

    pbar.close()
    save(inference_results, OUT_PREDICTION)
    if os.path.isfile(OUT_PREDICTION_TEMP):
        os.remove(OUT_PREDICTION_TEMP)
    return


if __name__ == '__main__':
    main()
