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
    Removes control characters and handles JSONDecodeError.
    """
    result = result.replace("\n", " ").replace("\r", " ")
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
    Construct a prompt for the model using the system prompt and the narrative text.
    """
    user_prompt = f"""
            Note Requiring Anonymization:
            
            {row['NarrativeText']}
                
            JSON Output:
            """
    return system_prompt + "\n" + user_prompt


def generate_prompt(data, num_examples=7):
    """
    Generate a system prompt with a specified number of anonymization examples.
    """
    data["anonymized"] = data["anonymized"].str.replace("'", "")
    data["anonymized"] = data["anonymized"].str.replace('"', "")
    try:
        num_examples = int(num_examples)
    except ValueError as e:
        raise ValueError(
            f"num_examples should be an integer, got {num_examples}"
        ) from e
    examples = ""
    for num in range(num_examples):
        examples += f"""
        Example {num+1}:
        
        {data.iloc[num, 2]}
        
        JSON output:
                {{
                    "anon_text": "{data.iloc[num, 1]}"
                }}
        
        """

    prompt = f"""
    Use the following examples of doctors notes to remove the following personally identifying information from the final doctors note and return the anonymized note in JSON format:

        Identifiable Information:
        Name (Full name, initials, etc.)
        Dates (dates in any format, written or numeric)
        Addresses (addresses in any form, especially those including house or apartment numbers)
        
        Also, make sure to delete all single quotes and double quotes if they exist

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
    Main function to anonymize doctors' notes using a language model.
    Reads input data, generates prompts, runs the model, and writes anonymized output.
    """
    num_examples = int(num_examples)

    output_csv_path = f"outputs/anon_{num_examples}share.csv"
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    samples = pd.read_csv(samples_path)
    input_df = pd.read_csv(data_path)
    system_prompt = generate_prompt(samples, num_examples)
    completed_ids = read_completed_ids(output_csv_path)

    # Remove already processed rows
    input_df = input_df[~input_df["AppointID"].isin(completed_ids)]
    print(f"Removed {len(completed_ids)} rows from input dataframe")
    print("Remaining rows: ", len(input_df))

    print("Going into the main loop now!")

    for batch in batch_iterator(input_df.iterrows(), max_batch_size):
        batch_prompts = []
        batch_metadata = []

        for index, row in batch:
            full_prompt = new_prompt(row, system_prompt)
            batch_prompts.append(full_prompt)
            this_metadata = {
                "row": index,
                "ID": row["ID"],
                "AppointID": row["AppointID"],
            }
            batch_metadata.append(this_metadata)

        # Generate anonymized text using the model
        results = pipeline(
            batch_prompts,
            max_length=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,
        )

        new_rows = []
        for metadata, result in zip(batch_metadata, results):
            gen_text = get_JSON(result[0]["generated_text"])
            parsed_json = parse_result(gen_text)
            if parsed_json is not None:
                metadata["gen-anonymized"] = parsed_json["anon_text"]
                new_rows.append(metadata)
            else:
                print("couldn't parse this: ", gen_text)

        # Write new anonymized rows to the output CSV
        if new_rows:
            print(f"Writing {len(new_rows)} rows to csv")
            print(f"New ids completed: {[row['AppointID'] for row in new_rows]}")
            new_df = pd.DataFrame(new_rows)
            print("New df: ", new_df)
            if os.path.exists(output_csv_path):
                new_df.to_csv(output_csv_path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
