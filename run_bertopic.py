from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import torch
import pandas as pd
import sys

def process_and_plot(df, filter_condition, output_suffix):
    # Filter the dataframe based on the condition
    filtered_df = df[filter_condition]
    docs = filtered_df["gen-anonymized"].tolist()

    # Initialize the ClassTfidfTransformer with reduce_frequent_words
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Create your representation model
    representation_model = MaximalMarginalRelevance(diversity=0.2)
    
    vectorizer_model = CountVectorizer(stop_words="english")

    # Initialize and train the topic model with the custom ctfidf_model and representation_model
    topic_model = BERTopic(ctfidf_model=ctfidf_model, vectorizer_model=vectorizer_model, embedding_model="all-MiniLM-L6-v2", representation_model=representation_model)
    topics, probs = topic_model.fit_transform(docs)

    # Get topic information
    topic_info = topic_model.get_topic_info()
    print(topic_info)

    # Save topic info to CSV
    topic_info_df = pd.DataFrame(topic_info)
    topic_info_df.to_csv(f"../outputs/bertopic_output_{output_suffix}.csv", index=False)

    # Generate and save the visualize_documents plot
    fig = topic_model.visualize_documents(docs, topics=topics)
    fig.write_html(f"../outputs/visualize_documents_{output_suffix}.html")

    # Release GPU memory
    torch.cuda.empty_cache()
    del topic_model


def main():
    # Read the CSV file
    df = pd.read_csv("../Data/extract_merged_withOQ_text.csv")

    # Filter out rows where any of the three columns has a 1
    condition_with_1 = (df["current_thoughts_about_death"] == 1) | (df["current_thoughts_about_ways_to_die"] == 1) | (df["current_intent_to_try_to_die"] == 1)
    condition_with_0 = ~condition_with_1

    # Process and plot for rows where at least one of the three columns has a 1
    process_and_plot(df, condition_with_1, "at_least_one_1")

    # Process and plot for rows where all three columns have 0
    process_and_plot(df, condition_with_0, "all_0")


if __name__ == "__main__":
    main()
    sys.exit(0)