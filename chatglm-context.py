'''
Remark: This file includes a demonstration of running ChatGLM2-6B on the dataset without utilizing in-context learning or fine-tuning.
'''

import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # Import tqdm for progress bar
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help='data')
parser.add_argument("--k", type=int, help='k')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Path to the original CSV file
path = f'./data/{args.data}.csv'

# Number of similar question to be generated
k=args.k

df = pd.read_csv(path)

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval()


output_path = f'{args.data}_with_similar_question_intent_{k}.csv'

# Create an empty CSV file with headers if it doesn't exist
if not os.path.isfile(output_path):
    results_df = pd.DataFrame(columns=['question', 'answer', 'generated similar question'])
    results_df.to_csv(output_path, index=False)

# Use tqdm to create a progress bar for the DataFrame iteration
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
    # Generate the response using the model
    response, history = model.chat(
        tokenizer,
        f"Instruction: 帮我根据问题{row['title']}生成{k}个不同且意思相近的问题。\n 输出必须包括{k}个句子，不能少于{k}个句子，这是强制要求。",
        history=[],
        temperature=0.9,
        top_k=5
    )
    
    response = response.replace('\n', ' ').strip() 
    
    # Create a DataFrame for the current result
    current_result = pd.DataFrame([{
        'question': row['question'],
        'answer': row['answer'],
        'generated similar question': response
    }])
    
    # Append the current result to the CSV file
    current_result.to_csv(output_path, mode='a', header=False, index=False)

print(f"Results saved to {output_path}")
