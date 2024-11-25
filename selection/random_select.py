import pandas as pd
import os
import random
import re
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize BERTScorer for Chinese language
scorer = BERTScorer(model_type='bert-base-chinese', lang='zh', rescale_with_baseline=True)


def random_sample(candidates, selection_size):
    return random.sample(candidates, min(selection_size, len(candidates)))

def calculate_total_diversity(selected_questions):
    """
    Calculate the total diversity of a set of selected questions.
    
    Parameters:
    selected_questions (list): A list of selected questions.
    
    Returns:
    float: The total diversity score based on pairwise distances.
    """
    total_diversity = 0.0  # Initialize total diversity score
    
    # Calculate the pairwise distance for each unique pair of questions
    for i in range(len(selected_questions)):
        for j in range(i + 1, len(selected_questions)):
            precison, recall, F1 = scorer.score([selected_questions[i]], [selected_questions[j]])
            total_diversity += recall.mean().item()
    
    return total_diversity

if __name__ == "__main__":
    file_path = './selection_eval_data.csv'
    output_path = './random_selected_questions_20_10.csv'
    
    df = pd.read_csv(file_path)
    
    results = []

    if not os.path.isfile(output_path):
        results_df = pd.DataFrame(columns=['question', 'selected', 'diversity'])
        results_df.to_csv(output_path, index=False)
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        candidates = re.split(r'\s*\d+\.\s*', row['generated similar question'])  # Adjust the column name as needed
        
        cleaned_questions = [q.strip() for q in candidates if q.strip()]
        
        budget = 10  # Total budget
        selection_size = 10  # Number of questions to select

        selected_questions = random_sample(cleaned_questions, selection_size)
        
        # Calculate total diversity by averaging pairwise F1 scores
        total_diversity = calculate_total_diversity(selected_questions)
        
        print(f"Randomly Selected Questions:")
        for question in selected_questions:
            print(question)
        
        result = {
            "question": row["question"],  # Original question from the row
            "selected": ", ".join(selected_questions),  # Join selected questions into a string
            "diversity": total_diversity  # Total diversity calculated from pairwise F1 scores
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    
    print(results_df["diversity"].mean())

    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}.")
