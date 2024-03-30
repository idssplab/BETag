import numpy as np
from torch.utils.data import Dataset as TorchDataset


class FTDataset(TorchDataset):

    def __init__(
        self,
        seqs: list[list[int]],
        base_tags: dict[int, list[str]],
        max_seq_len: int = 20,
        # max_n_tags_per_product: int | None = None,
        n_tags_per_item: tuple[int, int] | tuple[float, float]
        | None = None,
        # TODO: random  n_tags_per_product
        training: bool = True,
        sep: str = '| ',
    ):
        """_summary_

        Args:
            seqs (list[list[int]]): Sequence of sequences of item IDs
            base_tags (dict[int, list[str]]): A mapping from item ID to its base tags.
                Base tags that used as basic representation of items.
            max_seq_len (int, optional): Max length of sequences. Defaults to 20.
            n_tags_per_item (tuple[int, int] | tuple[float, float] | None, optional): 
                If (low, high): tuple[int, int]:
                    the number tags per items would sample a random int in [low, high).
                If (low, high): tuple[float, float]:
                    the number tags per items would be int(len(tags) * u), where
                        u sampled from a uniform distribution: u ~ U(low, high).
            training (bool, optional): Defaults to True.
        """
        super().__init__()
        self.training = training
        self.seqs = np.array(seqs, dtype=object)
        if training:
            self.counter = np.zeros((len(self.seqs), ), dtype=int)
            self.seqs_len = np.array(list(map(len, self.seqs)))
            self.sample_weights = self.seqs_len.astype(float)
        self.base_tags = base_tags
        self.max_seq_len = max_seq_len
        self.n_tags_per_item = n_tags_per_item
        if n_tags_per_item is not None:
            assert isinstance(n_tags_per_item[0], (int, float))
            assert isinstance(n_tags_per_item[1], (int, float))
        self.sep = sep
        return

    def __len__(self):
        return len(self.seqs)

    def _getitem_train(self, index):
        assert self.training is True
        if index == 0 and (self.sample_weights < self.seqs_len).all():
            # refresh
            self.sample_weights = self.seqs_len.astype(float)

        idx = np.random.choice(
            np.arange(len(self.seqs)), size=1,
            p=self.sample_weights / self.sample_weights.sum()
        )[0]
        self.sample_weights[idx] *= 0.5
        self.counter[idx] += 1
        seq = self.seqs[idx]
        start_idx = np.random.randint(
            0, 1 + max(0,
                       len(seq) - self.max_seq_len)
        )
        seq = seq[start_idx:start_idx + self.max_seq_len]
        return seq

    def __getitem__(self, index: int):

        if index >= len(self):
            raise IndexError

        def pid_to_input(i: int, pid: int):
            tags = self.base_tags[pid]
            if self.training:
                np.random.shuffle(tags)
                if self.n_tags_per_item is not None:
                    if isinstance(self.n_tags_per_item[0], float):
                        n_range = (
                            max(self.n_tags_per_item[0] * len(tags), 1),
                            self.n_tags_per_item[1] * len(tags) + 1,
                        )
                    else:
                        n_range = self.n_tags_per_item
                    n = np.random.randint(*n_range)
            else:
                if isinstance(self.n_tags_per_item[1], int):
                    n = self.n_tags_per_item[1] - 1
                else:
                    assert isinstance(self.n_tags_per_item[1], float)
                    n = int(self.n_tags_per_item[1] * len(tags))

            tags = tags[:n]
            return (f'{i+1}. ' + self.sep.join(tags))

        if self.training:
            seq = self._getitem_train(index)
        else:
            seq = self.seqs[index]
            seq = seq[-self.max_seq_len:]
        assert len(seq) >= 2, str(seq)
        article: str = '\n'.join(map(pid_to_input, range(len(seq)), seq))

        generation_prompt = '\n2.'
        # NOTE: '2.' instead of '2. '
        input_, output_ = article.split(generation_prompt, maxsplit=1)

        data = {
            'input': input_,
            'generation_prompt': generation_prompt,
            'output': output_
        }
        return data


class InferenceDataset(TorchDataset):

    def __init__(
        self,
        base_tags: dict[int, list[str]],
        max_n_tags_per_item: int | None = None,
        generation_prompt: str = '\n2.',
        sep: str = '| ',
    ):
        super().__init__()
        self.base_tags = base_tags
        self.pids = list(base_tags)
        self.max_n_tags_per_item = max_n_tags_per_item
        self.generation_prompt = generation_prompt
        self.sep = sep
        return

    def __len__(self):
        return len(self.base_tags)

    def __getitem__(self, index: int):

        pid = self.pids[index]
        input_ = (
            '1. ' +
            self.sep.join(self.base_tags[pid][:self.max_n_tags_per_item])
        )

        return {
            'input': input_,
            'generation_prompt': self.generation_prompt,
            'pid': pid
        }
