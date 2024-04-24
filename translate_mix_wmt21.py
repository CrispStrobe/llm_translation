import json
from tqdm import tqdm
import ctranslate2
import time
import argparse
import logging
import transformers

translator = ctranslate2.Translator("wmt21ct2_int8") #from cstr/wmt21-dense-24-wide-en-x-stq8
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
tokenizer.src_lang = "en"   

logging.basicConfig(level=logging.INFO, filename='translation.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_range_specification(range_specification, file_length):
    line_indices = []
    ranges = range_specification.split(',')
    for r in ranges:
        if '-' in r:
            parts = r.split('-')
            start = int(parts[0]) - 1 if parts[0] else 0
            end = int(parts[1]) - 1 if parts[1] else file_length - 1
            if start < 0 or end >= file_length:
                logging.error(f"Range {r} is out of bounds.")
                continue  # Skip ranges that are out of bounds
            line_indices.extend(range(start, end + 1))
        else:
            single_line = int(r) - 1
            if single_line < 0 or single_line >= file_length:
                logging.error(f"Line number {r} is out of bounds.")
                continue  # Skip line numbers that are out of bounds
            line_indices.append(single_line)
    return line_indices

def translate_text(text, translator, tokenizer):
    """
    Translates the given text from English to German using CTranslate2 and the WMT21 model,
    with special handling for newlines and segmenting text longer than 500 characters.
    Ensures sequences of newlines (\n\n, \n\n\n, etc.) are accurately reproduced.
    """
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
            target_prefix = [tokenizer.lang_code_to_token["de"]]
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
        logging.error(f"An error occurred during translation: {e}")
        return None

def translate_item_singleturn(item, raw_file_path, translator):
    """
    Translates the relevant fields in the given item from English to German using CTranslate2 and the WMT21 model,
    and saves the raw response to a backup file.
    """
    try:
        # Extract the relevant fields for translation
        prompt_content = item['prompt'][0]['content']
        chosen_content = item['chosen'][0]['content']
        rejected_content = item['rejected'][0]['content']

        # Translate the fields
        translated_prompt_content = translate_text(prompt_content, translator, tokenizer)
        translated_chosen_content = translate_text(chosen_content, translator, tokenizer)
        translated_rejected_content = translate_text(rejected_content, translator, tokenizer)
        
        # Write the raw response to a backup file
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write(f"Prompt content: {translated_prompt_content}\n")
            raw_file.write(f"Chosen content: {translated_chosen_content}\n")
            raw_file.write(f"Rejected content: {translated_rejected_content}\n\n")
        
        logging.info("Translation request successful.")
    except Exception as e:
        logging.error(f"An error occurred during translation: {e}")
        return None
    
    # Update the original item with the translated fields
    item['prompt'][0]['content'] = translated_prompt_content
    item['chosen'][0]['content'] = translated_chosen_content
    item['rejected'][0]['content'] = translated_rejected_content
    
    logging.info("Translation processing successful.")
    return item

def translate_item(item, raw_file_path, translator):
    """
    Translates the relevant fields in the given item from English to German using CTranslate2 and the WMT21 model,
    and saves the raw response to a backup file.
    """
    try:
        # Translate each part of the prompt separately and preserve the order
        translated_prompts = []
        for message in item['prompt']:
            translated_content = translate_text(message['content'], translator, tokenizer)
            translated_prompts.append({'content': translated_content, 'role': message['role']})

        # Translate the chosen and rejected contents
        translated_chosen_content = translate_text(item['chosen'][0]['content'], translator, tokenizer)
        translated_rejected_content = translate_text(item['rejected'][0]['content'], translator, tokenizer)
        
        # Write the raw response to a backup file
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write("Prompt content:\n")
            for translated_prompt in translated_prompts:
                raw_file.write(f"{translated_prompt['role']}: {translated_prompt['content']}\n")
            raw_file.write(f"Chosen content: {translated_chosen_content}\n")
            raw_file.write(f"Rejected content: {translated_rejected_content}\n\n")
        
        logging.info("Translation request successful.")
    except Exception as e:
        logging.error(f"An error occurred during translation: {e}")
        return None
    
    # Update the original item with the translated fields
    item['prompt'] = translated_prompts
    item['chosen'][0]['content'] = translated_chosen_content
    item['rejected'][0]['content'] = translated_rejected_content
    
    logging.info("Translation processing successful.")
    return item

def validate_item(item):
    """
    Validates the structure, presence, and content of required fields in the given item,
    allowing for multiple elements in the 'prompt' field for multi-turn conversations.
    """
    required_fields = ['dataset', 'prompt', 'chosen', 'rejected']
    for field in required_fields:
        if field not in item:
            logging.warning(f"Missing required field: {field}")
            return False
    
    # Check for at least one element in 'prompt' and exactly one element in 'chosen' and 'rejected'
    if len(item['prompt']) < 1 or len(item['chosen']) != 1 or len(item['rejected']) != 1:
        logging.warning("Invalid number of elements in 'prompt', 'chosen', or 'rejected' field.")
        return False
    
    # Validate 'content' and 'role' fields in all messages of 'prompt', and single elements of 'chosen' and 'rejected'
    for choice in item['prompt'] + item['chosen'] + item['rejected']:
        if 'content' not in choice or 'role' not in choice:
            logging.warning("Missing 'content' or 'role' field in choice.")
            return False
        if not isinstance(choice['content'], str) or not isinstance(choice['role'], str):
            logging.warning("Invalid type for 'content' or 'role' field in choice.")
            return False
    
    return True
    
def process_file(input_file_path, output_file_path, raw_file_path, line_indices, translator):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data_points = [json.loads(line) for line in file]

        failed_items = []
        failed_items_indices = []  # To track failed item indices

        for index in tqdm(line_indices, desc="Processing lines", unit="item"):
            item = data_points[index]

            # Validate the item structure
            if not validate_item(item):
                logging.warning("Skipping item due to invalid structure.")
                failed_items.append(item)
                continue

            # Translate the relevant fields in the item
            translated_item = None
            retry_count = 0
            while translated_item is None and retry_count < 3:
                translated_item = translate_item(item, raw_file_path, translator)
                retry_count += 1
                if translated_item is None:
                    logging.warning(f"Translation failed for item. Retry attempt: {retry_count}")
                    time.sleep(1)  # Wait for a short time before retrying
            
            if translated_item is not None:
                translated_item['index'] = index  # Add the line number as an "index" field
                # Write the translated item to the output file immediately
                with open(output_file_path, 'a', encoding='utf-8') as file:
                    file.write(json.dumps(translated_item, ensure_ascii=False) + "\n")
            else:
                failed_items_indices.append(index)
                failed_items.append(item)
                logging.error("Translation failed after multiple attempts. Skipping item.")
            
            # Validate the translated item structure
            if not validate_item(translated_item):
                logging.warning("Skipping translated item due to invalid structure.")
                failed_items.append(item)
                continue
        
        # Write the failed items to a separate file
        with open('failed_items.jsonl', 'w', encoding='utf-8') as file:
            for item in failed_items:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

        # After processing all items, generate the failed items index string
        failed_items_str = generate_failed_items_str(failed_items_indices)
        with open('failed_items_index.txt', 'w', encoding='utf-8') as f:
            f.write(failed_items_str)
        
        logging.info("Translation completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def generate_failed_items_str(indices):
    """
    Converts a list of failed item indices into a string.
    """
    if not indices:
        return ""

    # Sort the list of indices and initialize the first range
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

    # Add the last range
    if range_start == current:
        ranges.append(f"{range_start}")
    else:
        ranges.append(f"{range_start}-{current}")

    return ",".join(ranges)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Process and translate lines or ranges of lines from a file.")
        parser.add_argument('range', nargs='?', default="1-", help='The range of lines to process (e.g., "1-456", "789", "10-20,22,24-"). Note: Lines are 1-indexed.')
        args = parser.parse_args()

        input_file_path = 'mix_en.jsonl' #from alvarobartt/dpo-mix-7k-simplified
        output_file_path = 'mix_de.jsonl'
        raw_file_path = 'mix_de_raw.jsonl'

        print("Processing file...")

        # Calculate the file length once to avoid doing it repeatedly inside the function
        with open(input_file_path, 'r', encoding='utf-8') as file:
            file_length = sum(1 for line in file)

        # Parse the range specification argument
        range_specification = args.range
        line_indices = parse_range_specification(range_specification, file_length=file_length)

        # Proceed with processing if there are valid line indices
        if line_indices:
            process_file(input_file_path, output_file_path, raw_file_path, line_indices, translator)
        else:
            logging.error("No valid line indices to process. Please check the range specifications.")
    
    except Exception as e:
        logging.error(f"Unhandled exception occurred: {e}")
