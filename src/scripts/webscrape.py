from sentence_transformers import SentenceTransformer
import numpy as np

RECIPE_INTENT_EXAMPLES = [
    "give me a recipe",
    "find me a recipe",
    "how do you make this",
    "how to cook this",
    "show me how to cook",
    "I want to cook something",
    "tell me the cooking instructions",
]

model = SentenceTransformer("all-MiniLM-L6-v2")

intent_embeddings = model.encode(RECIPE_INTENT_EXAMPLES, normalize_embeddings=True, convert_to_numpy=True)

def detect_recipe_intent(query, threshold=0.50):
    q_emb = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)

    sims = np.dot(q_emb, intent_embeddings.T)
    best = np.max(sims)

    return best >= threshold, best
