# LLMs in Psychotheray Project README

This repository contains a collection of Python scripts and Jupyter notebooks I used for my master's project at BYU, focused on analyzing clinical notes to predict suicide ideation (SI) using machine learning techniques and natural language processing.

## Project Overview

Analyzing sensitive data, such as healthcare data, can be challenging due to access restrictions. For tabular data, workarounds can be used, such as excluding certain features, adding random noise, or applying a transformation to the data. Sensitive non-tabular data, such as text data, can be even more difficult to analyze because those classic workarounds are not applicable outside of the tabular setting.

This project generates value by: 1) providing access to sensitive data by processing it in a way that respects the privacy of the subject while still extracting meaningful information, and 2) combining that text data that was previously unavailable with existing quantitative data to improve model performance. In this case, I wanted to predict SI.

### The Data

I used data from BYU's on-campus counseling center called CAPS. Every time a client visits the center, they fill out a 45-question questionnaire, and the therapist records notes about the session. The questionnaire provides statements and asks on a scale of 0-4 how much the client agrees with them. One of the statements is "I have thoughts of ending my life." For the purposes of this project, any response above 0 was considered indicative of SI. The therapist notes themselves vary widely in length and topic.

### Methods

I chose Llama 3 70B for this project. I preferred an open-source model because I could keep all of the data and analysis on the university's own servers and avoid sending sensitive data elsewhere. I started by making one pass over the text data with Llama focused on anonymizing the text, then made a second pass with the model focused on extracting features from the text that were not already included in the questionnaire.

#### Anonymization

Few-shot prompting allowed me to guide Llama in how to anonymize the notes. To do this, I asked therapists from CAPS to provide several examples of notes that they had already de-identified. I then "re-identified" them by substituting names, dates, times, and places that I came up with myself, and used those data as examples of what a note should look like pre- and post-anonymization. I wrote the code so that whoever is running it can decide how many examples they would like to prompt the model with.

#### Data Extraction

Similar to the process for anonymization, I used few-shot prompting to coax the model into extracting the data I was interested in from the text. The therapists at CAPS again went through several case notes and identified the desired information. In terms of formatting, this looked like binary variables indicating whether a certain condition was present in the text (e.g., "Current Thoughts About Death"). These examples were then fed to the model as part of the prompt.

#### Predictive Modeling

I fit a logistic regression model to predict suicidal ideation (as indicated by the OQ survey) using the data. I also tried fitting a LightGBM model and an XGBoost model, but their performances were similar enough that I defaulted to the simplest and easiest to interpret model.

### Results

Using the text data in addition to the quantitative data alone makes a significant difference. The model achieves an F1-score of 0.647, a recall of 0.767, and precision of 0.559. Because this concerns identifying suicidal ideation, recall is the most important metric to watch for, as I aim to catch as many possible instances of suicidal ideation as I can. Additionally, the generative model's performance in anonymizing and extracting data from the text was excellent. Anonymizing the text alone replaced the work of 8 students totaling 1,250–2,100 hours, resulting in savings of $140k–$233k.


## File Overview

### Data Processing Scripts

#### `anonymize.py`
- A script that anonymizes clinical notes by removing personally identifiable information.
- Libraries used:
  - `fire`: Command-line interface library for creating parsable commands
  - `json`: For JSON data handling
  - `pandas`: For data manipulation and analysis
  - `torch`: Deep learning framework for tensor operations
  - `transformers`: Hugging Face library for transformer models
- Key functions:
  - `batch_iterator`: Creates batches from an iterator for efficient processing
  - `read_system_prompt`: Loads a system prompt from a file
  - `read_completed_ids`: Reads IDs of already processed records
  - `parse_result`, `get_JSON`: Parse and extract JSON from generated text
  - `new_prompt`, `generate_prompt`: Create prompts for the model
  - `main`: Orchestrates the anonymization process
- Output: CSV file containing anonymized clinical notes

#### `extract_data.py`
- Extracts structured information from clinical notes using LLM-based extraction.
- Libraries used: Same as anonymize.py
- Key functions: Similar structure to anonymize.py but focused on extracting structured data (e.g., suicide risk factors, symptoms) from the notes
- Output: CSV file containing extracted features from the notes (extract_{num_examples}examples_share.csv)

#### `run_bertopic.py`
- Performs topic modeling on the clinical notes using BERTopic.
- Libraries used:
  - `bertopic`: For topic modeling
  - `sklearn`: For feature extraction and vectorization
  - `torch`: For GPU memory management
  - `pandas`: For data handling
  - `fire`: For command-line interface
- Key functions:
  - `process_and_plot`: Filters data, applies topic modeling, and generates visualizations
  - `main`: Processes data with different filtering conditions
- Output:
  - CSV files with topic information
  - HTML visualizations of document clusters

### Jupyter Notebooks

#### `data_cleaning.ipynb`
- Cleans and prepares the raw clinical data for analysis.
- Key operations:
  - Removes short/useless entries
  - Adds unique identifiers
  - Prepares samples for extraction
  - Handles missing values
  - Creates visualizations of data distributions
  - Saves cleaned datasets
- Output: Various cleaned CSV files including cleaned_share.csv and cleaned_labled_samples.csv

#### `joining_tables.ipynb`
- Merges different data sources based on appointment IDs.
- Key operations:
  - Joins clinical note data with OQ questionnaire data
  - Joins extracted features with anonymized text
  - Saves merged datasets
- Output:
  - extract_merged_withOQ.csv
  - extract_merged_with_full_OQ.csv
  - extract_merged_withOQ_text.csv

#### `model_SI.ipynb`
- The main modeling notebook that builds predictive models for suicide ideation.
- Key operations:
  - Imports and prepares data
  - Creates abbreviations for column names
  - Performs exploratory data analysis (EDA)
  - Visualizes correlations and distributions
  - Implements and evaluates multiple machine learning models:
    - XGBoost
    - LightGBM
  - Cross-validates models with 5-fold validation
  - Uses SHAP for model explainability
  - Visualizes feature importance
- Output:
  - Model performance metrics (F1 score, recall, confusion matrix)
  - SHAP visualizations for feature importance
  - Word count distribution comparisons between SI and non-SI notes

#### `model_SI_with_full_OQ.ipynb`
- Similar to model_SI.ipynb but uses the complete OQ questionnaire data.

## Data Flow and Relationships
1. Raw clinical notes → anonymize.py → Anonymized notes
2. Anonymized notes → data_cleaning.ipynb → Cleaned notes
3. Cleaned notes → extract_data.py → Structured features
4. Structured features + OQ data → joining_tables.ipynb → Merged datasets
5. Merged datasets → model_SI.ipynb → Predictive models and analysis
6. Anonymized notes → run_bertopic.py → Topic analysis and visualizations

## Key Concepts
- SI (Suicide Ideation): The primary prediction target, represented by column OQ_8 in the questionnaire data
- Extraction: Using LLMs to convert unstructured clinical notes to structured data
- Features: Various indicators extracted from notes, including:
  - Past thoughts about death (PTAD)
  - Current thoughts about death (CAD)
  - Current intent to try to die (CITD)
  - Various psychological and social factors
- Model Evaluation: Uses F1 score and recall as primary metrics, with emphasis on recall for the positive class (detecting SI)

## Usage
Most scripts can be run with the fire CLI interface. For anonymization simply run

```bash
bash run_anonymize.sh
```

and for data extraction run

```bash
bash run_extract.sh
```
**IMPORTANT:**
Make sure you use a proper python environment with the correct packages installed.

The bertopic code can be run by just running the script itself, no need to use bash.


Notebooks should be run in the sequence: data_cleaning.ipynb → joining_tables.ipynb → model_SI.ipynb
