import os
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

texts = ["I love cats", "I adore felines"]

# Get embeddings
raw_responses = [client.embeddings.create(input=text, model="text-embedding-3-small") for text in texts]

embeddings = []

for res in raw_responses:
    emb = res.data[0].embedding
    embeddings.append(emb)

# Convert to numpy arrays
vecs = [np.array(e) for e in embeddings]

# Cosine similarity
cos_sim = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
print("Cosine similarity:", cos_sim)
