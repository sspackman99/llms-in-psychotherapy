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
    iterator = iter(iterator)
    while True:
        batch = tuple(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def read_system_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()


def read_completed_ids(output_csv_path):
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        df = pd.read_csv(output_csv_path)
        return set(df["AppointID"])
    else:
        return set()


def parse_result(result):
    # Remove control characters that are not allowed in JSON
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
    user_prompt = f"""
            Note Requiring Anonymization:
            
            {row['NarrativeText']}
                
            JSON Output:
            """
    return system_prompt + "\n" + user_prompt


def generate_prompt(data, num_examples=7):
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

    num_examples = int(num_examples)

    output_csv_path = f"outputs/anon_{num_examples}share.csv"
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    samples = pd.read_csv(samples_path)
    data = pd.read_csv(data_path)
    # input_df = data.iloc[num_examples + 1 : num_examples + 1001, :]
    # This is randomly sampling 3000 rows from the dataframe
    # input_df = data.sample(n=3000, random_state=42)
    input_df = data
    # print(num_examples)
    system_prompt = generate_prompt(samples, num_examples)
    # print(system_prompt)
    completed_ids = read_completed_ids(output_csv_path)

    input_df = input_df[~input_df["AppointID"].isin(completed_ids)]
    print(f"Removed {len(completed_ids)} rows from input dataframe")
    print("Remaining rows: ", len(input_df))

    print("Going into the main loop now!")

    for batch in batch_iterator(input_df.iterrows(), max_batch_size):
        batch_prompts = []
        batch_metadata = []
        batch_gen_lens = []
        # print("Batch: ", batch)

        for index, row in batch:
            full_prompt = new_prompt(row, system_prompt)
            # print("This is the full prompt: ", full_prompt)
            batch_prompts.append(full_prompt)
            # length = extract_prompt(full_prompt, "Selftext:", "JSON Output:")[1]
            # batch_gen_lens.append(length)
            this_metadata = {
                "row": index,
                "ID": row["ID"],
                "AppointID": row["AppointID"],
                # 'de-anonymized': row['de-anonymized']
            }
            batch_metadata.append(this_metadata)

        results = pipeline(
            batch_prompts,
            max_length=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,
        )

        new_rows = []
        for metadata, result in zip(batch_metadata, results):
            # print("result: ", result)
            gen_text = get_JSON(result[0]["generated_text"])
            parsed_json = parse_result(gen_text)
            if parsed_json is not None:
                # print("Parsed JSON: ", parsed_json)
                metadata["gen-anonymized"] = parsed_json["anon_text"]
                new_rows.append(metadata)
            else:
                print("couldn't parse this: ", gen_text)

        if new_rows:
            print(f"Writing {len(new_rows)} rows to csv")
            print(f"New ids completed: {[row['AppointID'] for row in new_rows]}")
            new_df = pd.DataFrame(new_rows)
            print("New df: ", new_df)
            # print("Does output csv exist? ", os.path.exists(output_csv_path))
            if os.path.exists(output_csv_path):
                new_df.to_csv(output_csv_path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
