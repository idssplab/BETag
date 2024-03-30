from typing import Any
from transformers import PreTrainedTokenizer
# yapf: disable


_LLAMA3_EN_IN_CONTEXT = {
    'system':
    'You are a helpful recommendation assistant who performs next products prediction task.',
    'user': (
        'Given a sequence of browsed products, '
        'where each product is represented by a set of "|"-separated tags, '
        'predict the tags for the products most likely to be browsed next.'
        '\n'
        '## Example Browsed Products:'
        '\n'
        '1. Salt-Free Mackerel| Grilled Fish| Seafood Delight' '\n'
        '2. Salty Crispy Chicken| Air Fryer Cooking| Fried Chicken' '\n'
        '\n'
        '## Example Predictions:'
        '\n'
        '1. Salt-Free Mackerel| Grilled Fish| Seafood Delight' '\n'
        '2. Salty Crispy Chicken| Air Fryer Cooking| Fried Chicken' '\n'
        '3. Salmon fillet| Fish dishes| Healthy recipes' '\n'
        '4. Fish Nuggets| Fried Fish| Snack' '\n'
        '\n'
        '### Task:'
        '\n'
        'Given the following browsed product, '
        # '\n'
        # '{inputs}'
        # '\n'
        'Predict the tags for the product most likely to be browsed next.'
    ),
    'generation_prompt': (
        '### Predicted Next Products:'
        '\n'
        '{inputs}{prompt}' # prompt: e.g. "\n2. "
    )
}

# yapf: enable


def llama3_instruct_en_in_context(
    tokenizer: PreTrainedTokenizer,
    data_point: dict[str, Any],
    include_generation_prompt: bool = True,
):
    """
    Args:
        tokenizer (PreTrainedTokenizer): _description_
        data_point (dict[str, Any]): _description_
        include_generation_prompt (bool, optional): If true, the returned prompt
        would include the generation prompt. Otherwise, a tuple:
        (input_prompt, generation_prompt) would return.

    Returns:
        If include_generation_prompt==True: 
            str: prompt
        If include_generation_prompt==False: 
            tuple(str, str): input_prompt, generation_prompt
    """
    messages = [
        {
            'role': 'system',
            'content': _LLAMA3_EN_IN_CONTEXT['system']
        },
        {
            'role':
            'user',
            'content':
            _LLAMA3_EN_IN_CONTEXT['user'].format(inputs=data_point['input'])
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    generation_prompt = _LLAMA3_EN_IN_CONTEXT['generation_prompt'].format(
        inputs=data_point['input'], prompt=data_point['generation_prompt']
    )
    if include_generation_prompt:
        prompt += generation_prompt
        return prompt
    return prompt, generation_prompt


llama3_instruct_en_in_context.GENERATION_SEP = '<|start_header_id|>assistant<|end_header_id|>\n\n### Predicted Next Products:\n'


def base_llm(tokenizer: PreTrainedTokenizer, data_point: dict[str, Any]):

    assert 'input' in data_point
    assert 'generation_prompt' in data_point
    bos = tokenizer.bos_token or ''
    return (bos + data_point['input'] + data_point['generation_prompt'])
