import fire
import json
import pandas as pd
import re
import os
import torch
from itertools import islice
import transformers

# Model path for the Llama-3-70B-Instruct model
model_path = "Llama-3-70B-Instruct"

# Initialize the HuggingFace transformers pipeline for text generation
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def batch_iterator(iterator, batch_size):
    """
    Yield batches of data from an iterator.
    """
    iterator = iter(iterator)
    while True:
        batch = tuple(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def read_system_prompt(file_path):
    """
    Read the system prompt from a file.
    """
    with open(file_path, "r") as file:
        return file.read()


def read_completed_ids(output_csv_path):
    """
    Read completed AppointIDs from the output CSV to avoid reprocessing.
    """
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        df = pd.read_csv(output_csv_path)
        return set(df["AppointID"])
    else:
        return set()


def parse_result(result):
    """
    Extract and parse the first JSON object found in the result string.
    Handles JSONDecodeError and replaces common invalid patterns.
    """
    result = result.replace("\n", " ").replace("\r", " ")
    result = result.replace("nan", "null")  # Convert invalid JSON 'nan' to 'null'
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
    """
    Extract the first JSON object from a string using stack-based parsing.
    """
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
    """
    Construct a prompt for the model using the system prompt and the anonymized note.
    """
    user_prompt = f"""
        Note Requiring data extraction:
        
        {row['gen-anonymized']}
            
        JSON Output:
        """
    return system_prompt + "\n" + user_prompt


def generate_prompt(data, num_examples=7):
    """
    Generate a system prompt with a specified number of extraction examples.
    """
    # Clean quote characters from input data
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

    # Build examples for few-shot prompting
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

    # List the fields to extract
    keys = data_json[0].keys()
    bulleted_list = "\n".join([f"- {key}" for key in keys if key != "gen-anonymized"])

    # Compose system prompt instructions
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
    num_examples,
    samples_path,
    temperature=0.6,
    top_p=0.9,
    max_seq_len=4960,
    max_gen_len=4960,
    max_batch_size=1,
):
    """
    Main function to extract structured JSON data from anonymized medical notes.
    Reads input data, generates prompts, runs the model, and writes extracted data.
    """
    num_examples = int(num_examples)

    output_csv_path = f"outputs/extract_{num_examples}examples_share.csv"

    # Read samples (for few-shot examples) and input data
    samples = pd.read_csv(samples_path)
    data = pd.read_csv(data_path)
    
    # Decide subset if needed (currently using all rows)
    input_df = data

    # Generate the shared system prompt
    system_prompt = generate_prompt(samples, num_examples)

    # Skip rows that have already been processed
    completed_ids = read_completed_ids(output_csv_path)
    input_df = input_df[~input_df["AppointID"].isin(completed_ids)]

    print(f"Removed {len(completed_ids)} rows from input dataframe")
    print("Remaining rows: ", len(input_df))
    print("Going into the main loop now!")

    for batch in batch_iterator(input_df.iterrows(), max_batch_size):
        batch_prompts = []
        batch_metadata = []

        # Prepare prompts and metadata for each row
        for index, row in batch:
            full_prompt = new_prompt(row, system_prompt)
            print("This is the full prompt: ", full_prompt)
            batch_prompts.append(full_prompt)
            this_metadata = {
                "row": index,
                "ID": row["ID"],
                "AppointID": row["AppointID"],
            }
            batch_metadata.append(this_metadata)

        # Generate structured data from the model
        results = pipeline(
            batch_prompts,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,
        )

        # Parse and store output
        new_rows = []
        for metadata, result in zip(batch_metadata, results):
            gen_text = get_JSON(result[0]["generated_text"])
            print(gen_text)
            gen_text = gen_text.replace("'", '"')  # Normalize quotes for JSON parsing
            parsed_json = parse_result(gen_text)
            if parsed_json is not None:
                metadata.update(parsed_json)
                new_rows.append(metadata)
            else:
                print("couldn't parse this: ", gen_text)

        # Write new rows to the output CSV
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
