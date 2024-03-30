# BETag

This repository contains the code for the paper *BETag: Behavior-enhanced Item Tagging with Finetuned Large Language Models*.

## Repository Status  
We plan to make this code open source. However, as we are in the process of applying for a patent related to this work, the repository is temporarily unavailable.  

We are committed to completing the process as quickly as possible and will make the repository publicly accessible once the patent process is finalized. Thank you for your understanding and patience.  


## Installation for BETag Generation

1. Install PyTorch (version >= 2.0) with the appropriate CUDA version for your system. 
2. Install dependencies using the following command:
    - Alternatively, you can manually check and install dependencies listed in `pyproject.toml`.

```sh
pip install -e .
```

## Base Tag Generation

Base tags serve as the foundational representation of products and can be any relevant tags. We provide a script (`base_tags_generation.py`) for generating base tags using an LLM API.

The base tags must be organized in the following format for subsequent BE-finetuning and BETag Generation:

```python
Mapping[PID, list[str]]
```

## BE-Finetuning

To finetune the model on your own dataset, you need:

1. **Training interaction sequences**: A list of interaction sequences (e.g., `list[list[PID]]`).
2. **Base Tags**: A mapping of product IDs (PIDs) to lists of tags (`Mapping[PID, list[Tag]]`).

### Steps:

1. Prepare the dataset and specify paths to your data in a dotenv configuration file. For example:

```env
inters_path = dataset/amazon.scientific/inters.train.json
base_tags_path = dataset/amazon.scientific/base_tags.json
...
```

2. Run the finetuning script:

```sh
python beft.py --env path/to/the/.env
```

### Notes:

- Preprocessed datasets used in the paper are available [here](https://drive.google.com/drive/folders/1lInWdSQUyXEKRP-XY8QRv3LISzo6mNmV?usp=sharing).
- Default environment configurations can be found in the `envs.default` directory.
  - Finetuned checkpoints are available on [google drive](https://drive.google.com/drive/folders/1RzJYQTFFtvC7o8yMdVBxuotitBUYPtgg?usp=sharing).

---

## BETag Generation

For BETag Generation, interactions are not required for BETag Generation.

### Requirements:

1. **Base Tags**: Use the same base tags as in BE-finetuning.
2. **Checkpoint**: Path to the finetuned LLM checkpoint.

### Steps:

1. Configure the dotenv file with required paths.
2. Run the generation script:

```sh
python begen.py --env path/to/the/.env
```

### Output Files:

The output directory will contain the following files:

1. `generation_config.json`: Contains the generation configuration.
2. `raw_predict.json`: The raw output of the LLM.
3. `raw_betags.json`: Parsed BETags in the format `Mapping[PID, list[list[str]]]`.
    - For each product, the generated tags for each beam are stored separately.
    - You can select the top-M beams for each product:
      
      ```python
      betags = {pid: beams[:TOP_M+1] for pid, beams in raw_betags.items()}
      ```
      - Beams are sorted by score, from highest to lowest. The base tags are included as the first beam, resulting in `M+1` beams.
    - To use weighted tags or select top-K tags via:
      
      ```python
      from collections import Counter
      betags = {pid: Counter(sum(beams, [])).most_common(TOP_K) for pid, beams in betags.items()}
      ```

### Notes:

- Generated BETags are available [here](https://drive.google.com/drive/folders/1GozeWRkTJ4K3kpZpFmyBRf1zQaKFjG--?usp=sharing).

---

## Credits

The Amazon dataset used in this work was from [Recformer](#https://github.com/AaronHeee/RecFormer).


---

## Citation

```
TODO
```
