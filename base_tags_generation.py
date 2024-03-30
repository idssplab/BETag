import pandas as pd
import openai
import os
from dotenv import load_dotenv

load_dotenv()  # load environment variables


def generate_tags(row, max_retries=3):
    prompt_text = (
        "Based on the product details provided below, generate up to 10 tags in Mandarin, formatted as a numbered list. "
        "These tags are for product recommendation on an online shopping website. "
        f"Name: {row['product_name']}, "
        f"Keywords: {row['keywords']}, "
        f"Descriptions: {row['product_info']}."
    )
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role": "user",
                        "content": prompt_text
                    }
                ]
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Retry {retries + 1}: An error occurred - {e}")
            retries += 1
            if retries == max_retries:
                print(
                    f"Failed to process row: {row['product_name']} - skipping"
                )
                return "Error generating tags; skipped entry."


def main():
    file_path = 'products_extracted_arranged.csv'
    data = pd.read_csv(file_path)
    data = data[~data['product_name'].str.
                contains("Product name not found", na=False)]

    openai.api_key = os.getenv('OPENAI_API_KEY')

    data['generated_tags'] = data.apply(generate_tags, axis=1)
    data.to_csv('ver4.csv', index=False)
    print("Tags generation completed and data saved to 'ver4.csv'.")


if __name__ == "__main__":
    main()
