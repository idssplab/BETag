# pip install --upgrade openai
import argparse
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()  # load environment variables


def generate_tags(row, dataset_name, openai_key, gen_feq, max_retries=3):

    client = OpenAI(api_key=openai_key)

    match dataset_name:
        case 'Scientific':
            prompt_text = (
                "Based on the product details provided below, generate several tags in English, formatted as a numbered list. "
                "These tags are intended for product recommendations on an online shopping website. "
                "Each tag should be concise, including key features or benefits of the product, and should avoid lengthy descriptions. "
                f"Name: {row['title']}, "
                f"Brand: {row['brand']}, "
                f"Category: {row['category']}. "
            )
        case 'Movielens-1M':
            prompt_text = (
                "Based on the movie title and plot provided below, generate several tags in English, formatted as a numbered list. "
                "These tags are intended for categorization and recommendation purposes on a streaming platform. "
                "Each tag should be concise, highlighting key themes, genres, or unique elements of the movie, and should avoid lengthy descriptions. "
                f"Title: {row['title']}, "
                f"Plot: {row['info_plot']}."
            )
        case _:
            prompt_text = "Invalid dataset choice. Please select a valid option."
            print(prompt_text)
            return

    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role": "user",
                        "content": prompt_text
                    }
                    ],
                n=gen_feq
            )
            return [response.choices[i].message.content.strip() for i in range(gen_feq)]

        except Exception as e:
            print(f"Retry {retries + 1}: An error occurred - {e}")
            retries += 1
            if retries == max_retries:
                print(
                    f"Failed to process row: {row['title']} - skipping"
                )
                return "Error generating tags; skipped entry."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="dataset name", default="")
    parser.add_argument("--data_path", type=str, help="data path", default="")
    parser.add_argument("--output_path", type=str, help="output path", default="./output/")
    parser.add_argument("--openai_key", type=str, help="OPENAI API KEY", default="")
    parser.add_argument("--gen_feq", type=int, help="gen_feq", default=1)
    parser.add_argument("--batch_size", type=int, help="batch size", default=100)
    
    paras = parser.parse_args()

    dataset_name = paras.dataset_name
    data_path = paras.data_path
    openai_key = paras.openai_key
    output_path = paras.output_path
    gen_feq = paras.gen_feq
    batch_size = paras.batch_size

    os.makedirs(output_path, exist_ok=True)
    data = pd.read_csv(data_path) #[0:5]
    num_batches = (len(data) + batch_size - 1) // batch_size
   
    batch_files = []
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, len(data))
        print(str(batch) + ': ' + str(start_idx) + ' ~ ' +str(end_idx))
        batch_data = data.iloc[start_idx:end_idx]
        tags = batch_data.apply(generate_tags, dataset_name=dataset_name, openai_key=openai_key, gen_feq=gen_feq, axis=1)
        pkl_name = f'tags_batch_{batch + 1}.pkl'
        batch_files.append(pkl_name)
        tags.to_pickle(os.path.join(output_path, pkl_name))

    all_tags = pd.Series()
    for batch_file in batch_files:
        batch_tags = pd.read_pickle(os.path.join(output_path, batch_file))
        batch_tags = batch_tags.apply(lambda texts: [text.replace('\r', '\n') for text in texts])
        all_tags = pd.concat([all_tags, batch_tags], ignore_index=True)

    all_tags.to_pickle(os.path.join(output_path, 'tags.pkl'))

    tags = pd.DataFrame(all_tags.tolist(), columns=[f'base_tags_{i+1}' for i in range(len(all_tags.iloc[0]))])
    tags.to_csv(os.path.join(output_path, "tags.csv"), index=False)

    for i in range(gen_feq):
        col_name = f'base_tags_{i+1}'
        tags[col_name] = tags[col_name].apply(lambda x: re.sub(r'(\d+)\. ', r'\n\1. ', x))
        tags[col_name] = tags[col_name].apply(lambda x: '\n'.join([line for line in x.splitlines() if line.strip() and line.strip()[0].isdigit()]))
        tags[col_name] = tags[col_name].str.replace('"', "")
        tags[col_name] = tags[col_name].str.replace('*', "")

    result = pd.concat([data, tags], axis=1)
    result.to_csv(os.path.join(output_path, "base_tags.csv"), index=False)
    print(f"Tags generation completed and data saved to base_tags.csv'.")



if __name__ == "__main__":
    main()
