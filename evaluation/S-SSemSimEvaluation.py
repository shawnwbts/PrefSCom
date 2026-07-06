# evaluate SBCS
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('dataProcess/output_comments.csv',encoding='ISO-8859-1')

# form https://github.com/antonio-mastropaolo/code-summarization-metric/tree/main
model = SentenceTransformer('xx/all-MiniLM-L6-v2/')


def compute_semantic_similarity(function_code, comment):
    function_vec = model.encode(function_code)
    comment_vec = model.encode(comment)
    similarity = cosine_similarity([function_vec], [comment_vec])
    return similarity[0][0]

#

threshold = 0.5
similarities = []

for index, row in df.iterrows():
    function_code = row['comment']
    comment = row['comment1']

    similarity = compute_semantic_similarity(function_code, comment)
    similarities.append(similarity)

    if similarity >= threshold:
        print(f"Row {index}: The comment is a good match for the function (Similarity: {similarity:.2f})")
    else:
        print(f"Row {index}: The comment does NOT match the function (Similarity: {similarity:.2f})")
df['similarity'] = similarities

df.to_csv('dataProcess/functions_and_comments_with_similarity.csv', index=False,encoding='ISO-8859-1')

print("done！")



