# evaluate SIDE

import pandas as pd
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

DEVICE = "cpu"


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model and tokenizer
checkPointFolder = "/model/hard-negatives/141205"
tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
model = AutoModel.from_pretrained(checkPointFolder).to(DEVICE)


# Read CSV file
def process_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    similarities = []

    for index, row in df.iterrows():
        method = row[0]
        codeSummary = row[1]

        pair = [method, codeSummary]
        encoded_input = tokenizer(pair, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Compute similarity
        sim = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()

        # Store similarity in the third column
        df.at[index, 2] = sim

        print(f"Row {index}: Similarity = {sim}")

    # Save results back to CSV
    df.to_csv(file_path, index=False, header=False)
    return df


# Example usage
csv_file = "evaluation/train_base_function.csv"  #
results = process_csv(csv_file)
