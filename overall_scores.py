import json
from tabulate import tabulate
from huggingface_hub import hf_hub_download

# Hugging Face dataset repository details
dataset_repo = "cstr/Capybara-de-snippets"
dataset_files = [
    "Capybara_de_GPT4_scored.jsonl",
    "Capybara_de_Claude-3-Opus_scored.jsonl",
    "Capybara_de_GPT3.5_scored.jsonl",
    "Capybara_de_deepl_scored.jsonl",
    "Capybara_de_mixtral_scored.jsonl",
    "Capybara_de_occiglot_scored.jsonl",
    "Capybara_de_wmt19_scored.jsonl"
]

# Download the scored JSONL files from Hugging Face
file_paths = {}
for file_name in dataset_files:
    file_path = hf_hub_download(repo_id=dataset_repo, filename=file_name, repo_type="dataset")
    file_paths[file_name] = file_path
    print(f"Downloaded {file_name} to {file_path}")

# Create a dictionary to store the overall scores for each file
overall_scores = {}

# Process each scored JSONL file
for file_name in dataset_files:
    # Open the scored JSONL file
    with open(file_paths[file_name], "r", encoding="utf-8") as file:
        scored_data = [json.loads(line) for line in file]

    # Initialize variables to store the sum and count of scores
    input_score_sum = 0
    output_score_sum = 0
    total_turns = 0

    # Iterate over each conversation in the scored data
    for conv in scored_data:
        # Iterate over each turn in the conversation
        for turn in conv["conversation"]:
            input_score = turn["input_score"]
            output_score = turn["output_score"]

            # Check if the scores are lists and extract the first element
            if isinstance(input_score, list):
                input_score = input_score[0]
            if isinstance(output_score, list):
                output_score = output_score[0]

            input_score_sum += input_score
            output_score_sum += output_score
            total_turns += 1

    # Calculate the average input and output scores
    avg_input_score = input_score_sum / total_turns
    avg_output_score = output_score_sum / total_turns

    # Calculate the overall score as the average of input and output scores
    overall_score = (avg_input_score + avg_output_score) / 2

    # Store the overall score for the file
    overall_scores[file_name] = overall_score

# Create a table to display the overall scores
table_data = []
for file_name, score in overall_scores.items():
    table_data.append([file_name, score])

# Sort the table data based on the overall score in descending order
table_data.sort(key=lambda x: x[1], reverse=True)

# Print the table using the tabulate library
headers = ["File", "Overall Score"]
print(tabulate(table_data, headers, tablefmt="grid"))