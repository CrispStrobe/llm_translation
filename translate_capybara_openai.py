
import json
import openai
from tqdm import tqdm
import time
import argparse
import logging

api_key = ""

logging.basicConfig(level=logging.INFO, filename='translation.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
try:
    openai_client = openai.OpenAI(api_key=api_key)
    #openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    print ("client initalised")
except Exception as e:
    print(f"An error occurred during API initialisation: {e}")

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

def translate_text_openai(text):
    """
    Translates the given text from English to German using OpenAI's translation capabilities.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that translates text from English to German. You only translate natural language parts and not e.g. code vocabulary. You MUST respond with only the translated text. NO explanation, NO comments, NO remarks about the translation itself, ONLY the translation."},
            {"role": "user", "content": f"Translate the following to German:\n\n{text}"}
        ]
        #print ("this is my message:\n",messages)

        response = openai_client.chat.completions.create(
            model="gpt-4-0125-preview",
            temperature=0.5,
            max_tokens=2000,
            messages=messages
        )

        translated_text = response.choices[0].message.content.strip()
        #print ("this is the response:\n",translated_text)
        return translated_text

    except Exception as e:
        logging.error(f"An error occurred during the translation API call: {e}")
        return None

def translate_item(item, raw_file_path):
    """
    Translates the relevant fields in the given item from English to German using the API for translation service,
    and saves the raw response to a backup file.
    """
    try:
        # Extract the relevant fields for translation
        conversation = item['conversation']
        for turn in conversation:
            turn['input'] = translate_text_openai(turn['input'])
            turn['output'] = translate_text_openai(turn['output'])
        
        # Write the raw response to a backup file
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        logging.info("Translation request successful.")
    
    except Exception as e:
        logging.error(f"An error occurred during API call: {e}")
        return None
    
    logging.info("Translation processing successful.")
    return item

def validate_item(item):
    """
    Validates the structure, presence, and content of required fields in the given item.
    """
    required_fields = ['source', 'conversation']
    for field in required_fields:
        if field not in item:
            logging.warning(f"Missing required field: {field}")
            return False
    
    if not isinstance(item['conversation'], list):
        logging.warning("'conversation' field is not a list.")
        return False
    
    for turn in item['conversation']:
        if 'input' not in turn or 'output' not in turn:
            logging.warning("Missing 'input' or 'output' field in conversation turn.")
            return False
        if not isinstance(turn['input'], str) or not isinstance(turn['output'], str):
            logging.warning("Invalid type for 'input' or 'output' field in conversation turn.")
            return False
    
    return True

def process_file(input_file_path, output_file_path, raw_file_path, line_indices):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data_points = [json.loads(line) for line in file]

        failed_items = []
        failed_items_indices = []  # To track failed item indices

        request_count = 0
        token_count = 0
        start_time = time.time()

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
                translated_item = translate_item(item, raw_file_path)
                retry_count += 1
                if translated_item is None:
                    logging.warning(f"Translation failed for item. Retry attempt: {retry_count}")
                    time.sleep(1)  # Wait for a short time before retrying
            
            if translated_item is not None:
                # Write the translated item to the output file immediately
                with open(output_file_path, 'a', encoding='utf-8') as file:
                    file.write(json.dumps(translated_item, ensure_ascii=False) + "\n")
            else:
                failed_items_indices.append(index)
                failed_items.append(item)
                logging.error("Translation failed after multiple attempts. Skipping item.")
                    
            request_count += 1
            # Estimate the token count conservatively as the script does not have access to the exact number
            token_count += len(json.dumps(item)) + (len(json.dumps(translated_item)) if translated_item else 0)
                        
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

        input_file_path = 'CapybaraPure_Decontaminated.jsonl'
        output_file_path = 'Capybara_de_openai.jsonl'
        raw_file_path = 'Capybara_de_openai_raw.jsonl'

        print ("Processing file...")

        # Calculate the file length once to avoid doing it repeatedly inside the function
        with open(input_file_path, 'r', encoding='utf-8') as file:
            file_length = sum(1 for line in file)

        # Parse the range specification argument
        range_specification = args.range
        line_indices = parse_range_specification(range_specification, file_length=file_length)

        # Proceed with processing if there are valid line indices
        if line_indices:
            process_file(input_file_path, output_file_path, raw_file_path, line_indices)
        else:
            logging.error("No valid line indices to process. Please check the range specifications.")
    
    except Exception as e:
        logging.error(f"Unhandled exception occurred: {e}")
