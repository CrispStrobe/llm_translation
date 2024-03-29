import json
from tqdm import tqdm
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import time
import argparse

# Initialize WMT tokenizer and model
mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

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

def translate_text_wmt(text):
    """
    Translates the given text from English to German using the WMT model.
    """
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    except Exception as e:
        print(f"An error occurred during the translation: {e}")
        return None

def translate_item(item):
    try:
        # Extract the relevant fields for translation
        conversation = item['conversation']
        translated_conversation = {}  # Store translated turns in a dictionary with index as key
        for index, turn in enumerate(conversation):
            translated_input = translate_text_wmt(turn['input'])
            translated_output = translate_text_wmt(turn['output'])
            
            translated_conversation[index] = {
                'input': translated_input,
                'output': translated_output
            }
        
        # Update the 'conversation' field with the translated turns while preserving the order
        item['conversation'] = [translated_conversation[index] for index in sorted(translated_conversation.keys())]
        
        return item
    
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None

    return item

def validate_item(item):
    """
    Validates the structure, presence, and content of required fields in the given item.
    """
    required_fields = ['source', 'conversation']
    for field in required_fields:
        if field not in item:
            print(f"Missing required field: {field}")
            return False
    
    if not isinstance(item['conversation'], list):
        print("'conversation' field is not a list.")
        return False
    
    for turn in item['conversation']:
        if 'input' not in turn or 'output' not in turn:
            print("Missing 'input' or 'output' field in conversation turn.")
            return False
        if not isinstance(turn['input'], str) or not isinstance(turn['output'], str):
            print("Invalid type for 'input' or 'output' field in conversation turn.")
            return False
    
    return True

def process_file(input_file_path, output_file_path, line_indices):
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
                print("Skipping item due to invalid structure.")
                failed_items.append(item)
                continue

            # Translate the relevant fields in the item
            translated_item = None
            retry_count = 0
            while translated_item is None and retry_count < 3:
                translated_item = translate_item(item)
                retry_count += 1
                if translated_item is None:
                    print(f"Translation failed for item. Retry attempt: {retry_count}")
                    time.sleep(1)  # Wait for a short time before retrying
            
            if translated_item is not None:
                # Write the translated item to the output file immediately
                with open(output_file_path, 'a', encoding='utf-8') as file:
                    file.write(json.dumps(translated_item, ensure_ascii=False) + '\n')

            request_count += 1
            token_count += len(tokenizer.encode(item['source'], return_tensors='pt').squeeze())

        elapsed_time = time.time() - start_time
        average_tokens_per_request = token_count / request_count if request_count > 0 else 0

        print(f"\nProcessed {request_count} items in {elapsed_time:.2f} seconds.")
        print(f"Average tokens per request: {average_tokens_per_request:.2f}")

        if failed_items:
            print(f"\nTranslation failed for {len(failed_items)} items.")

        return True

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Translate JSONL file using WMT model.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--lines", type=str, help="Comma-separated line indices or ranges to translate (e.g., '1,3-5,7').")
    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_path = args.output_file
    line_indices = []

    with open(input_file_path, 'r', encoding='utf-8') as file:
        file_length = sum(1 for _ in file)

    if args.lines:
        line_indices = parse_range_specification(args.lines, file_length)

    if process_file(input_file_path, output_file_path, line_indices):
        print("Translation completed successfully.")
    else:
        print("Translation encountered errors.")

if __name__ == "__main__":
    main()