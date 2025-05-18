import fire
import json
import pandas as pd
import re
import os
import torch
from itertools import islice
import transformers

with open("/yunity/sps58/huggingface_token.txt", "r") as file:
    ACCESS_TOKEN = file.read().strip()

model_path = "Llama-3-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def batch_iterator(iterator, batch_size):
    """Yield batches of specified size from an iterator."""
    iterator = iter(iterator)
    while True:
        batch = tuple(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def read_system_prompt(file_path):
    """Read system prompt from a file."""
    with open(file_path, "r") as file:
        return file.read()


def read_completed_ids(output_csv_path):
    """Read IDs of already processed rows from output CSV."""
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        df = pd.read_csv(output_csv_path)
        return set(df["AppointID"])
    else:
        return set()


def parse_result(result):
    # Remove control characters that are not allowed in JSON
    result = result.replace("\n", " ").replace("\r", " ")
    # Replace 'nan' with 'null' to make it valid JSON
    result = result.replace("nan", "null")
    match = re.search(r"\{[\s\S]*\}", result)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"Failed to parse JSON string: {json_str}")
            return None
    return None


def get_JSON(string):
    stack = []
    json_str = ""
    in_json = False

    for char in string:
        if char == "{":
            stack.append(char)
            in_json = True
        if in_json:
            json_str += char
        if char == "}":
            stack.pop()
            if not stack:
                break

    return json_str if json_str else string


def new_prompt(row, system_prompt):
    """Create a new prompt for each row of data."""
    user_prompt = f"""
        Note Requiring data extraction:
        
        {row['gen-anonymized']}
            
        JSON Output:
        """
    return system_prompt + "\n" + user_prompt


def generate_prompt(data, num_examples=7):
    """Generate a comprehensive prompt with examples for data extraction."""
    # Clean input data
    data["gen-anonymized"] = data["gen-anonymized"].replace("'", "")
    data["gen-anonymized"] = data["gen-anonymized"].replace('"', "")

    data_json = data.to_dict(orient="records")

    # Validate num_examples
    try:
        num_examples = int(num_examples)
    except ValueError as e:
        raise ValueError(
            f"num_examples should be an integer, got {num_examples}"
        ) from e

    # Create examples string
    examples = ""
    for num in range(num_examples):
        example_data = data_json[num].copy()
        gen_anonymized_text = example_data.pop("gen-anonymized")
        examples += f"""
        Example {num+1}:
        
        {gen_anonymized_text}
        
        JSON output:
                {example_data}
        
        """

    # Extract keys for extraction
    keys = data_json[0].keys()
    bulleted_list = "\n".join([f"- {key}" for key in keys if key != "gen-anonymized"])

    # Construct comprehensive prompt
    prompt = f"""
    Use the following examples of doctors notes to extract information about the client and their well-being.

        Make sure to only extract the following information:
        {bulleted_list}
        The rest of the categories are boolean values that are True if the client has that condition and False if they do not.
        Also, make sure to return your answer in proper JSON formatting, using only double quotes.
        
    """
    return prompt + "\n" + examples


def main(
    data_path,
    # tokenizer_path,
    num_examples,
    samples_path,
    temperature=0.6,
    top_p=0.9,
    max_seq_len=4960,
    max_gen_len=4960,
    max_batch_size=1,
):
    """
    Main function to process medical notes and extract structured JSON data.

    Args:
    - ckpt_dir: Directory for model checkpoints
    - tokenizer_path: Path to tokenizer
    - num_examples: Number of examples to use for few-shot learning
    - input_csv_path: Path to input CSV file
    - temperature: Sampling temperature for text generation
    - top_p: Nucleus sampling parameter
    - max_seq_len: Maximum sequence length
    - max_gen_len: Maximum generation length
    - max_batch_size: Maximum batch size for processing
    """
    # Convert num_examples to integer
    num_examples = int(num_examples)

    # Prepare output CSV path
    output_csv_path = f"outputs/extract_{num_examples}examples_share.csv"

    # Read input data
    samples = pd.read_csv(samples_path)
    data = pd.read_csv(data_path)
    # Decide how many examples to use for training, and how many rows to actually use the model on (assuming they are all in the same file)
    # input_df = data.iloc[num_examples + 1 : num_examples + 11,]
    input_df = data

    # Generate system prompt with examples
    system_prompt = generate_prompt(samples, num_examples)

    # Track completed IDs to avoid reprocessing
    completed_ids = read_completed_ids(output_csv_path)
    input_df = input_df[~input_df["AppointID"].isin(completed_ids)]

    print(f"Removed {len(completed_ids)} rows from input dataframe")
    print("Remaining rows: ", len(input_df))
    print("Going into the main loop now!")

    # Process data in batches
    for batch in batch_iterator(input_df.iterrows(), max_batch_size):
        batch_prompts = []
        batch_metadata = []

        # Prepare prompts for each row in batch
        for index, row in batch:
            full_prompt = new_prompt(row, system_prompt)
            print("This is the full prompt: ", full_prompt)
            batch_prompts.append(full_prompt)
            this_metadata = {
                "row": index,
                "ID": row["ID"],
                "AppointID": row["AppointID"],
                # 'de-anonymized': row['de-anonymized']
            }
            batch_metadata.append(this_metadata)

        # Generate text completions
        results = pipeline(
            batch_prompts,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,
        )

        # Process generated results
        new_rows = []
        for metadata, result in zip(batch_metadata, results):
            gen_text = get_JSON(result[0]["generated_text"])
            print(gen_text)
            gen_text = gen_text.replace("'", '"')
            parsed_json = parse_result(gen_text)
            if parsed_json is not None:
                metadata.update(parsed_json)
                new_rows.append(metadata)
            else:
                print("couldn't parse this: ", gen_text)

        # Write results to CSV
        if new_rows:
            print(f"Writing {len(new_rows)} rows to csv")
            print(f"New ids completed: {[row['AppointID'] for row in new_rows]}")
            new_df = pd.DataFrame(new_rows)
            new_df = new_df.drop_duplicates()

            if os.path.exists(output_csv_path):
                new_df.to_csv(output_csv_path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
