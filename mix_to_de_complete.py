#python main.py
#this will work with hardcoded parameters, see below in function main(), it needs only one environment variable: HUGGINGFACE_TOKEN
#so it should run in a docker container like this: docker run --rm -e HUGGINGFACE_TOKEN=your_token_here crispstrobe/wmt21:latest

#python main.py
import os
import pandas as pd
import requests
from pathlib import Path
import ctranslate2
import time
import json
from tqdm import tqdm
import subprocess
import transformers
from huggingface_hub import snapshot_download, upload_file, HfApi, create_repo

# Function to download a Parquet file from a specified URL
def download_parquet(url, local_path):
	response = requests.get(url, stream=True)
	if response.status_code == 200:
		with open(local_path, 'wb') as file:
			for chunk in response.iter_content(chunk_size=1024):
				file.write(chunk)
		print("File downloaded successfully.")
	else:
		print(f"Failed to download file, status code: {response.status_code}")

# Corrected function to convert Parquet files to JSONL format
def convert_parquet_to_jsonl(parquet_filename, output_dir):
	try:
		# Read the parquet file
		df = pd.read_parquet(parquet_filename)
		print(f"Read Parquet file {parquet_filename} successfully.")

		# Convert the dataframe to a JSON string and handle Unicode characters and forward slashes
		json_str = df.to_json(orient='records', lines=True, force_ascii=False)
		print(f"Converted Parquet file to JSON string.")

		# Replace escaped forward slashes if needed
		json_str = json_str.replace('\\/', '/')

		# Write the modified JSON string to the JSONL file
		output_path = Path(output_dir) / 'train.jsonl'
		print(f"Attempting to save to {output_path}")
		with open(output_path, 'w', encoding='utf-8') as file:
			file.write(json_str)
		print(f"Data saved to {output_path}")
	except Exception as e:
		print(f"Failed to convert Parquet to JSONL: {e}")
		raise

# Function to count lines in a JSONL file
def count_lines_in_jsonl(file_path):
	with open(file_path, 'r', encoding='utf-8') as file:
		line_count = sum(1 for _ in file)
	return line_count

def parse_range_specification(range_specification, file_length):
	line_indices = []
	ranges = range_specification.split(',')
	for r in ranges:
		if '-' in r:
			parts = r.split('-')
			start = int(parts[0]) - 1 if parts[0] else 0
			end = int(parts[1]) - 1 if parts[1] else file_length - 1
			if start < 0 or end >= file_length:
				print(f"Range {r} is out of bounds.")
				continue  # Skip ranges that are out of bounds
			line_indices.extend(range(start, end + 1))
		else:
			single_line = int(r) - 1
			if single_line < 0 or single_line >= file_length:
				print(f"Line number {r} is out of bounds.")
				continue  # Skip line numbers that are out of bounds
			line_indices.append(single_line)
	return line_indices

def translate_text(text, translator, tokenizer, target_language="de"):
	try:
		segments = []
		newline_sequences = []  # To store sequences of newlines
		segment = ""

		i = 0
		while i < len(text):
			# Collect sequences of newlines
			if text[i] == '\n':
				newline_sequence = '\n'
				while i + 1 < len(text) and text[i + 1] == '\n':
					newline_sequence += '\n'
					i += 1
				if segment:
					segments.append(segment)  # Add the preceding text segment
					segment = ""
				newline_sequences.append(newline_sequence)  # Store the newline sequence
			else:
				segment += text[i]
				# If segment exceeds 500 characters, or if we reach the end of the text, process it
				if len(segment) >= 500 or i == len(text) - 1:
					end_index = max(segment.rfind('.', 0, 500), segment.rfind('?', 0, 500), segment.rfind('!', 0, 500))
					if end_index != -1 and len(segment) > 500:
						# Split at the last punctuation within the first 500 characters
						segments.append(segment[:end_index+1])
						segment = segment[end_index+1:].lstrip()
					else:
						# No suitable punctuation or end of text, add the whole segment
						segments.append(segment)
						segment = ""
			i += 1

		# Translate the collected text segments
		translated_segments = []
		for segment in segments:
			source = tokenizer.convert_ids_to_tokens(tokenizer.encode(segment))
			target_prefix = [tokenizer.lang_code_to_token[target_language]]
			results = translator.translate_batch([source], target_prefix=[target_prefix])
			target = results[0].hypotheses[0][1:]
			translated_segment = tokenizer.decode(tokenizer.convert_tokens_to_ids(target))
			translated_segments.append(translated_segment)

		# Reassemble the translated text with original newline sequences
		translated_text = ""
		for i, segment in enumerate(translated_segments):
			translated_text += segment
			if i < len(newline_sequences):
				translated_text += newline_sequences[i]  # Insert the newline sequence

		return translated_text.strip()

	except Exception as e:
		print(f"An error occurred during translation: {e}")
		return None

def translate_item_ufb(item, raw_file_path, translator, tokenizer, target_language="de"):
    try:
        # Übersetze den Prompt und speichere das Original
        original_prompt = item['prompt']
        translated_prompt = translate_text(original_prompt, translator, tokenizer, target_language)

        # Übersetze die gewählten und abgelehnten Inhalte
        original_chosen = item['chosen']
        translated_chosen = []
        for choice in original_chosen:
            translated_content = translate_text(choice['content'], translator, tokenizer, target_language)
            translated_chosen.append({'content': translated_content, 'role': choice['role']})

        original_rejected = item['rejected']
        translated_rejected = []
        for choice in original_rejected:
            translated_content = translate_text(choice['content'], translator, tokenizer, target_language)
            translated_rejected.append({'content': translated_content, 'role': choice['role']})

        # Schreibe die rohe Antwort in eine Sicherungsdatei
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write(f"Prompt: {translated_prompt}\n")
            raw_file.write(f"Chosen: {json.dumps(translated_chosen, ensure_ascii=False)}\n")
            raw_file.write(f"Rejected: {json.dumps(translated_rejected, ensure_ascii=False)}\n\n")

        print("Translation request successful.")
        # Aktualisiere das Originalelement mit den übersetzten Feldern, aber behalte das Original bei
        item['prompt_translated'] = translated_prompt
        item['chosen_translated'] = translated_chosen
        item['rejected_translated'] = translated_rejected
        return item

    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None
	    
def validate_item_ufb(item):
	# Check basic required fields including 'prompt' as a simple string
	required_fields = ['source', 'prompt', 'chosen', 'rejected']
	for field in required_fields:
		if field not in item:
			print(f"Missing required field: {field}")
			return False
		if field == 'prompt' and not isinstance(item['prompt'], str):
			print("Prompt must be a string.")
			return False

	# Check 'chosen' and 'rejected' which should be lists of dictionaries
	for field in ['chosen', 'rejected']:
		if not isinstance(item[field], list) or not item[field]:
			print(f"No entries or incorrect type for section: {field}")
			return False
		for idx, message in enumerate(item[field]):
			if 'content' not in message or 'role' not in message:
				print(f"Missing 'content' or 'role' field in {field} at index {idx}")
				return False
			if not isinstance(message['content'], str) or not isinstance(message['role'], str):
				print(f"Invalid type for 'content' or 'role' field in {field} at index {idx}")
				return False

	return True

def translate_item_sciriff(item, raw_file_path, translator, tokenizer, target_language="de"):
    try:
        # Übersetze jeden Teil der Nachrichten separat und behalte die Reihenfolge bei
        original_messages = item['messages']
        translated_messages = []
        for message in original_messages:
            translated_content = translate_text(message['content'], translator, tokenizer, target_language)
            translated_messages.append({'content': translated_content, 'role': message['role']})

        # Schreibe die rohe Antwort in eine Sicherungsdatei
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write("Messages content:\n")
            for translated_message in translated_messages:
                raw_file.write(f"{translated_message['role']}: {translated_message['content']}\n")
            raw_file.write("\n")

        print("Translation request successful.")
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None

    # Aktualisiere das Originalelement mit den übersetzten Feldern, aber behalte das Original bei
    item['messages_translated'] = translated_messages

    print("Translation processing successful.")
    return item

def validate_item_sciriff(item):
    required_fields = ['dataset', 'id', 'messages']
    for field in required_fields:
        if field not in item:
            print(f"Missing required field: {field}")
            return False

    # Prüfe, ob mindestens eine Nachricht vorhanden ist
    if not item['messages']:
        print("No messages found in the item.")
        return False

    # Validiere 'content' und 'role' Felder in allen Nachrichten
    for message in item['messages']:
        if 'content' not in message or 'role' not in message:
            print("Missing 'content' or 'role' field in message.")
            return False
        if not isinstance(message['content'], str) or not isinstance(message['role'], str):
            print("Invalid type for 'content' or 'role' field in message.")
            return False

    return True


def translate_item_mix(item, raw_file_path, translator, tokenizer, target_language="de"):
    try:
        # Übersetze jeden Teil des Prompts separat und behalte die Reihenfolge bei
        original_prompts = item['prompt']
        translated_prompts = []
        for message in original_prompts:
            translated_content = translate_text(message['content'], translator, tokenizer, target_language)
            translated_prompts.append({'content': translated_content, 'role': message['role']})

        # Übersetze die gewählten und abgelehnten Inhalte
        original_chosen = item['chosen']
        translated_chosen_content = translate_text(original_chosen[0]['content'], translator, tokenizer, target_language)
        translated_chosen = [{'content': translated_chosen_content, 'role': original_chosen[0]['role']}]

        original_rejected = item['rejected']
        translated_rejected_content = translate_text(original_rejected[0]['content'], translator, tokenizer, target_language)
        translated_rejected = [{'content': translated_rejected_content, 'role': original_rejected[0]['role']}]

        # Schreibe die rohe Antwort in eine Sicherungsdatei
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write("Prompt content:\n")
            for translated_prompt in translated_prompts:
                raw_file.write(f"{translated_prompt['role']}: {translated_prompt['content']}\n")
            raw_file.write(f"Chosen content: {translated_chosen_content}\n")
            raw_file.write(f"Rejected content: {translated_rejected_content}\n\n")

        print("Translation request successful.")
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None

    # Aktualisiere das Originalelement mit den übersetzten Feldern, aber behalte das Original bei
    item['prompt_translated'] = translated_prompts
    item['chosen_translated'] = translated_chosen
    item['rejected_translated'] = translated_rejected

    print("Translation processing successful.")
    return item

def validate_item_mix(item):
	required_fields = ['dataset', 'prompt', 'chosen', 'rejected']
	for field in required_fields:
		if field not in item:
			print(f"Missing required field: {field}")
			return False

	# Check for at least one element in 'prompt' and exactly one element in 'chosen' and 'rejected'
	if len(item['prompt']) < 1 or len(item['chosen']) != 1 or len(item['rejected']) != 1:
		print("Invalid number of elements in 'prompt', 'chosen', or 'rejected' field.")
		return False

	# Validate 'content' and 'role' fields in all messages of 'prompt', and single elements of 'chosen' and 'rejected'
	for choice in item['prompt'] + item['chosen'] + item['rejected']:
		if 'content' not in choice or 'role' not in choice:
			print("Missing 'content' or 'role' field in choice.")
			return False
		if not isinstance(choice['content'], str) or not isinstance(choice['role'], str):
			print("Invalid type for 'content' or 'role' field in choice.")
			return False

	return True


def translate_item_ufb_cached(item, raw_file_path, translator, tokenizer, target_language="de"):
    try:
        translated_texts = {}  # Cache zum Speichern übersetzter Texte

        # Übersetze den Prompt, falls nötig (da es eine Benutzereingabe ist und erneut erscheinen kann)
        original_prompt = item['prompt']
        if original_prompt not in translated_texts:
            translated_prompt = translate_text(original_prompt, translator, tokenizer, target_language)
            translated_texts[original_prompt] = translated_prompt
        else:
            translated_prompt = translated_texts[original_prompt]

        # Hilfsfunktion zum Umgang mit der Inhaltsübersetzung mit Caching
        def get_translated_content(content):
            if content not in translated_texts:
                translated_texts[content] = translate_text(content, translator, tokenizer, target_language)
            return translated_texts[content]

        # Übersetzungen für gewählte und abgelehnte Abschnitte verarbeiten
        original_chosen = item['chosen']
        translated_chosen = []
        for interaction in original_chosen:
            translated_content = get_translated_content(interaction['content'])
            translated_chosen.append({'content': translated_content, 'role': interaction['role']})

        original_rejected = item['rejected']
        translated_rejected = []
        for interaction in original_rejected:
            translated_content = get_translated_content(interaction['content'])
            translated_rejected.append({'content': translated_content, 'role': interaction['role']})

        # Schreibe die rohe Antwort in eine Sicherungsdatei
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write(f"Prompt: {translated_prompt}\n")
            raw_file.write(f"Chosen: {json.dumps(translated_chosen, ensure_ascii=False)}\n")
            raw_file.write(f"Rejected: {json.dumps(translated_rejected, ensure_ascii=False)}\n\n")

        print("Translation request successful.")
        # Aktualisiere das Originalelement mit den übersetzten Feldern, aber behalte das Original bei
        item['prompt_translated'] = translated_prompt
        item['chosen_translated'] = translated_chosen
        item['rejected_translated'] = translated_rejected
        return item

    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None
	    

def validate_item_ufb_cached(item):
	# Check basic required fields
	required_fields = ['source', 'prompt', 'chosen', 'rejected']
	for field in required_fields:
		if field not in item:
			print(f"Missing required field: {field}")
			return False

	# Ensure 'prompt' is a string
	if not isinstance(item['prompt'], str):
		print("Prompt must be a string.")
		return False

	# Check 'chosen' and 'rejected' which should be lists of dictionaries
	for field in ['chosen', 'rejected']:
		if not isinstance(item[field], list) or not item[field]:
			print(f"No entries or incorrect type for section: {field}")
			return False
		for idx, message in enumerate(item[field]):
			if 'content' not in message or 'role' not in message:
				print(f"Missing 'content' or 'role' field in {field} at index {idx}")
				return False
			if not isinstance(message['content'], str) or not isinstance(message['role'], str):
				print(f"Invalid type for 'content' or 'role' field in {field} at index {idx}")
				return False

	return True

def process_file(input_file_path, output_file_path, raw_file_path, line_indices, translator, tokenizer, model_type, target_language="de"):
    try:
        # Zuordnen von Validierungs- und Übersetzungsfunktionen basierend auf model_type
        if model_type == "mix":
            print("translating a mix-style model...")
            validate_item = validate_item_mix
            translate_item = translate_item_mix
        elif model_type == "ufb_cached":
            print("translating an ufb_cached-style model...")
            validate_item = validate_item_ufb_cached
            translate_item = translate_item_ufb_cached
        elif model_type == "ufb":
            print("translating an ultrafeedback-style model...")
            validate_item = validate_item_ufb
            translate_item = translate_item_ufb
        elif model_type == "sciriff":
            print("translating a SciRIFF-style model...")
            validate_item = validate_item_sciriff
            translate_item = translate_item_sciriff
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        with open(input_file_path, 'r', encoding='utf-8') as file:
            data_points = [json.loads(line) for line in file]

        failed_items = []
        failed_items_indices = []

        for index in tqdm(line_indices, desc="Processing lines", unit="item"):
            item = data_points[index]

            # Validieren der Elementstruktur
            if not validate_item(item):
                print("Skipping item due to invalid structure.")
                failed_items.append(item)
                continue

            # Übersetzen der relevanten Felder im Element
            translated_item = None
            retry_count = 0
            while translated_item is None and retry_count < 3:
                print("going to translate the item...")
                translated_item = translate_item(item, raw_file_path, translator, tokenizer, target_language)
                retry_count += 1
                if translated_item is None:
                    print(f"Translation failed for item. Retry attempt: {retry_count}")
                    time.sleep(1)
            
            if translated_item is not None:
                translated_item['index'] = index
                with open(output_file_path, 'a', encoding='utf-8') as file:
                    file.write(json.dumps(translated_item, ensure_ascii=False) + "\n")
            else:
                failed_items_indices.append(index)
                failed_items.append(item)
                print("Translation failed after multiple attempts. Skipping item.")

            # Validieren der übersetzten Elementstruktur
            if not validate_item(translated_item):
                print("Skipping translated item due to invalid structure.")
                failed_items.append(item)
                continue
        
        with open('failed_items.jsonl', 'w', encoding='utf-8') as file:
            for item in failed_items:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

        failed_items_str = generate_failed_items_str(failed_items_indices)
        with open('failed_items_index.txt', 'w', encoding='utf-8') as f:
            f.write(failed_items_str)
        
        print("Translation completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


def generate_failed_items_str(indices):
	if not indices:
		return ""

	indices.sort()
	range_start = indices[0]
	current = range_start
	ranges = []

	for i in indices[1:]:
		if i == current + 1:
			current = i
		else:
			if range_start == current:
				ranges.append(f"{range_start}")
			else:
				ranges.append(f"{range_start}-{current}")
			range_start = current = i

	if range_start == current:
		ranges.append(f"{range_start}")
	else:
		ranges.append(f"{range_start}-{current}")

	return ",".join(ranges)

# Function to upload the output file to Hugging Face
def upload_output_to_huggingface(output_file_path, repo_name, token):
	api = HfApi()
	
	# Check if the repository exists
	try:
		print("Checking repo:", repo_name)
		api.repo_info(repo_id=repo_name, repo_type="dataset", token=token)
	except Exception as e:
		if "404" in str(e):
			# Create the repository if it doesn't exist
			print("Creating it...")
			create_repo(repo_id=repo_name, repo_type="dataset", token=token)
			print(f"Created repository: {repo_name}")
		else:
			print(f"Failed to check repository existence: {e}")
			return

	# Upload the file to the repository
	try:
		print("Starting dataset upload from:", output_file_path)
		upload_file(
			path_or_fileobj=output_file_path,
			path_in_repo=output_file_path,
			repo_id=repo_name,
			repo_type="dataset",
			token=token,
			)
		print(f"Uploaded {output_file_path} to Hugging Face repository: {repo_name}")
	except Exception as e:
		print(f"Failed to upload {output_file_path} to Hugging Face: {e}")
		raise

def translate_dataset(train_url, local_parquet_path, input_file_path, output_file_path, raw_file_path, range_specification, model_type, output_dir, output_repo_name, token, translator, tokenizer, target_language="de"):
	try:
		# Download the Parquet file
		download_parquet(train_url, local_parquet_path)
	except Exception as e:
		print(f"Failed to download the Parquet file from {train_url}: {e}")
		return

	try:
		# Convert the downloaded Parquet file to JSONL
		convert_parquet_to_jsonl(local_parquet_path, output_dir)
	except Exception as e:
		print(f"Failed to convert Parquet to JSONL: {e}")
		return

	try:
		# Rename the JSONL file using subprocess to ensure correct handling
		subprocess.run(["mv", f"{output_dir}/train.jsonl", input_file_path], check=True)
	except subprocess.CalledProcessError as e:
		print(f"Failed to rename the file from 'train.jsonl' to {input_file_path}: {e}")
		return

	try:
		# Count lines in the JSONL file to validate contents
		line_count = count_lines_in_jsonl(input_file_path)
		print(f"Number of lines in the file: {line_count}")
	except Exception as e:
		print(f"Failed to count lines in {input_file_path}: {e}")
		return

	try:
		# Parse the range specification for processing specific lines
		line_indices = parse_range_specification(range_specification, file_length=line_count)
		if not line_indices:
			print("No valid line indices to process. Please check the range specifications.")
			return
	except Exception as e:
		print(f"Error parsing range specification '{range_specification}': {e}")
		return

	try:
		# Process the file with specified model type and line indices
		process_file(input_file_path, output_file_path, raw_file_path, line_indices, translator, tokenizer, model_type, target_language)
	except Exception as e:
		print(f"Failed to process the file {input_file_path}: {e}")
		return

	try:
		# Upload the output file to Hugging Face repository
		upload_output_to_huggingface(output_file_path, output_repo_name, token)
	except Exception as e:
		print(f"Failed to upload {output_file_path} to Hugging Face: {e}")  

# Integration der neuen Funktionen in die main-Funktion
def main():
    try:
        print("Initializing...")
        # Konfiguration und Pfade
        tokenizer_name = "facebook/wmt21-dense-24-wide-en-x"
        model_repo_name = "cstr/wmt21ct2_int8"  # Repository zum Herunterladen des Modells
        output_repo_name = "cstr/datasets_de_test"  # Repository zum Hochladen der Ausgabedatei
        token = os.getenv("HUGGINGFACE_TOKEN")

       # Herunterladen des Modell-Snapshots von Hugging Face
        model_path = snapshot_download(repo_id=model_repo_name, token=token)
        print(f"Model downloaded to: {model_path}")

        # Laden des CTranslate2-Modells
        translator = ctranslate2.Translator(model_path, device="auto")
        print("CTranslate2 model loaded successfully.")

        # Laden des Tokenizers
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.src_lang = "en"
        print("Tokenizer loaded successfully.")

        # Aufgaben definieren
        tasks = [
            {
                "url": "https://huggingface.co/datasets/alvarobartt/dpo-mix-7k-simplified/resolve/main/data/train-00000-of-00001.parquet?download=true",
                "local_path": "train.parquet",
                "input_file": "mix_en.jsonl",
                "output_file": "mix_de.jsonl",
                "raw_file": "mix_de_raw.jsonl",
                "range_spec": "1-5",
                "model_type": "mix"
            },
            {
                "url": "https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned/resolve/main/data/train-00000-of-00001.parquet?download=true",
                "local_path": "train.parquet",
                "input_file": "ufb_en.jsonl",
                "output_file": "ufb_de.jsonl",
                "raw_file": "ufb_de_raw.jsonl",
                "range_spec": "1-5",
                "model_type": "ufb"
            },
            {
                "url": "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet?download=true",
                "local_path": "train.parquet",
                "input_file": "ufb_cached_en.jsonl",
                "output_file": "ufb_cached_de.jsonl",
                "raw_file": "ufb_cached_de_raw.jsonl",
                "range_spec": "1-5",
                "model_type": "ufb_cached"
            },
            {
                "url": "https://huggingface.co/datasets/allenai/SciRIFF-train-mix/resolve/main/data/train-00000-of-00001.parquet?download=true",
                "local_path": "train.parquet",
                "input_file": "sciriff_en.jsonl",
                "output_file": "sciriff_de.jsonl",
                "raw_file": "sciriff_de_raw.jsonl",
                "range_spec": "1-5",
                "model_type": "sciriff"
            }
            ]

        for task in tasks:
            translate_dataset(
                train_url=task["url"],
                local_parquet_path=task["local_path"],
                input_file_path=task["input_file"],
                output_file_path=task["output_file"],
                output_dir=".",
                output_repo_name=output_repo_name,
                raw_file_path=task["raw_file"],
                token=token,
                range_specification=task["range_spec"],
                model_type=task["model_type"],
                translator=translator,
                tokenizer=tokenizer,
                target_language="de"  # Hardcoded target language
            )

    except Exception as e:
        print(f"An error occurred in the main function: {e}")
        raise

if __name__ == '__main__':
    main()
