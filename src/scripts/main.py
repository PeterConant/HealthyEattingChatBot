import nutritiondb
import webscraping
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
import json

def ollama_generate(user_query, retrieved_context):
    url_generate = "http://localhost:11434/api/generate"
    prompt = f"""Use the following context to answer the question.

    Context:
    {retrieved_context}

    Question: {user_query}

    Answer:"""
    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    }
    print("\n\n---Sending Message to Model------------------\n\n")
    response = requests.post(url_generate, json=payload)
    print("Response: " + response.json()["response"])

def ollama_chat(user_query, retrieved_context):
    url_chat = "http://localhost:11434/api/chat"
    prompt = f"""Use the following context to answer the question.

    Context:
    {retrieved_context}

    Question: {user_query}

    Answer:"""
    payload = {
        "model": "phi3",
        "messages":[
            {"role": "system", "content": "You are a helpful healthy eatting chatbot. Answer the user's question using the provided context to inform you answer. Keep the conversation related to topics on food, cooking, and healthy eating."
            },
            {"role": "user", "content": f"""Context:
            {retrieved_context}
            User Question:
            {user_query}"""}
        ],
        "stream": False
    }

    response = requests.post(url_chat, json=payload)

    return response.json()["message"]["content"], retrieved_context

model = SentenceTransformer("all-MiniLM-L6-v2")
webscraping.set_model(model)
nutritiondb.set_model(model)

RECIPE_INTENT_EXAMPLES = [
    "give me a recipe for chicken parmesan",
    "how do you make chocolate chip cookies",
    "what's a good way to cook salmon",
    "I want to bake a cake",
    "show me instructions for pad thai",
    "how long do I cook a turkey",
    "what temperature for baking bread",
    "recipe ideas for dinner tonight",
]

FOODLIST_NUTRITION_INTENT_EXAMPLES = [
    "What are the nutrients in a bagel with cream cheese",
    "Is salmon low in sodium?",
    "Is avocado healthy?",
    "What are the macros for Greek yogurt?",
    "How many carbs in bread?",
    "How much protein is in chicken breast?",
    "Give me the nutritional information for almonds?",
    "Calories in a banana?",
    "Does spinach have iron?",
    "Compare nutrition of brown rice vs white rice?",
    "What vitamins are in oranges?",
    "How much vitamin A in apples?"
]

TOPNBOTTOM_FOODS_INTENT_EXAMPLES = [
    "What are some high protien foods?",
    "What are some low fat foods?",
    "What foods are high in protien?",
    "What are high in vitamins?",
    "Give me some foods that are high in carbohydrates."
    "Give me some foods that are low in fat.",
    "Give me some foods that are high in vitamines.",
    "What foods are very nutrient dense?"
]    

def detect_intent(query, threshold=0.50):
    foodlist_nutrition_intent_embeddings = model.encode(FOODLIST_NUTRITION_INTENT_EXAMPLES, normalize_embeddings=True, convert_to_numpy=True)
    topnbottom_foods_intent_embeddings = model.encode(TOPNBOTTOM_FOODS_INTENT_EXAMPLES, normalize_embeddings=True, convert_to_numpy=True)
    recipe_intent_embeddings = model.encode(RECIPE_INTENT_EXAMPLES, normalize_embeddings=True, convert_to_numpy=True)
    
    q_emb = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)

    foodlist_sims = np.dot(q_emb, foodlist_nutrition_intent_embeddings.T)
    foodlist_best = np.max(foodlist_sims)
    topnbottom_sims = np.dot(q_emb, topnbottom_foods_intent_embeddings.T)
    topnbottom_best = np.max(topnbottom_sims)
    recipe_sims = np.dot(q_emb, recipe_intent_embeddings.T)
    recipe_best = np.max(recipe_sims)

    scores = {
        'foodlist_best': foodlist_best,
        'topnbottom_best': topnbottom_best,
        'recipe_best': recipe_best
    }
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]

    return best_score >= threshold, best_score, best_name


def retrieve_context(query):
    #intent
    intent_bool, score, intent_type = detect_intent(query)
    if intent_bool:
        match intent_type:
            case 'foodlist_best':
                context = nutritiondb.get_foodnutrition_in_userquery(query)
            case 'topnbottom_best':
                context = nutritiondb.highlow_nutrients(query)
            case 'recipe_best':
                context = webscraping.recipe_request(query)
    return context
    
                

def prompt_user():
    user_prompt = "Ask about healthy eatting and recipes:\n"
    query = input(user_prompt)
    return query

def main(query="I need to eat more vitamin C, what should I eat."):
    #query = prompt_user() 
    context = retrieve_context(query)
    response, retrieved_context = ollama_chat(query, context)
    return response, retrieved_context

if __name__ == "__main__":
    main()

#"I have cream cheese, wine, chicken breast, pork, and mushrooms in my fridge"
#"what are some high protein foods?"
#"what foods are low in fat?"
#"I need more vitamin C, what foods should I eat?"
#"what should I eat if I want low fat"
#"Get me a recipe for lemon chicken"
#"How do I cook fried rice"
#"How do I make pizza"
#"How much fat does chicken breast have?"
#
#
#