import pandas as pd
import requests

def download_parquet(url, filename):
    # Download the file from the URL and save it locally
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print("Failed to download file, status code:", response.status_code)
        return False
    return True

def parquet_to_jsonl(parquet_filename, jsonl_filename):
    # Read the parquet file
    df = pd.read_parquet(parquet_filename)

    # Convert the dataframe to a JSON string and handle Unicode characters and forward slashes
    json_str = df.to_json(orient='records', lines=True, force_ascii=False)

    # Replace escaped forward slashes if needed
    json_str = json_str.replace('\\/', '/')

    # Write the modified JSON string to the JSONL file
    with open(jsonl_filename, 'w', encoding='utf-8') as file:
        file.write(json_str)

    print(f"Data saved to {jsonl_filename}")

# Correct URL for the test dataset
test_url = "https://huggingface.co/datasets/alvarobartt/dpo-mix-7k-simplified/resolve/main/data/train-00000-of-00001.parquet?download=true"

# Download and convert the test dataset
if download_parquet(test_url, "train.parquet"):
    parquet_to_jsonl("train.parquet", "mix_en.jsonl")
