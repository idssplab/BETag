import os
import argparse
import naive_flow as nf
from betag.llm import LlamaInstruct
from betag.ft_utils import BEFTConfig
from betag.dataset import FTDataset
from betag.instructions import llama3_instruct_en_in_context


def filter_seqs(seqs: list[list[int]], min_seq_len: int = 2):
    org_len = len(seqs)
    seqs = [seq for seq in seqs if len(seq) >= min_seq_len]
    print(f'{org_len - len(seqs)}/{org_len} dropped due to {min_seq_len = }')
    print(f'{len(seqs)} sessions remaining')
    return seqs


def main():
    arg_parser = argparse.ArgumentParser('BE-Finetuning')
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

    ft_config = BEFTConfig(_env_file='envs/beft.amazon.scentific.env')
    print(nf.strfconfig(ft_config))

    if args.check:
        return

    train_dataset = FTDataset(
        seqs=filter_seqs(ft_config.inters, min_seq_len=ft_config.min_seq_len),
        base_tags=ft_config.base_tags,
        max_seq_len=ft_config.max_seq_len,
        n_tags_per_item=ft_config.n_tags_per_item,
    )
    val_dataset = None

    # Print several examples
    for i in [0, 1, 2]:
        d = train_dataset[i]
        print(f'-------------------- {i = } ------------------------')
        print(d['input'] + d['generation_prompt'] + d['output'])

    llm = LlamaInstruct(
        model_name=ft_config.llm_name,
        instruction=llama3_instruct_en_in_context,
        step_ckpt_dir='ckpts_sub',
        epoch_ckpt_dir=ft_config.ckpt_dir,
        inference_only=False,
    )

    for i in [0, 1, 2]:
        d = train_dataset[i]
        print(f'-------------------- {i = } ------------------------')
        print(llm.instruction(llm.tokenizer, d))

    llm.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epoch=ft_config.num_epoch,
        training_config=ft_config.training_config,
        lora_config=ft_config.lora_config,
        add_eos_token=False,
        save_checkpoint_callbacks=[ft_config.dump],
    )


if __name__ == '__main__':
    main()
