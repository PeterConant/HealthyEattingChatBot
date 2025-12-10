import numpy as np
import spacy
import requests
from bs4 import BeautifulSoup
import time


#spacy.cli.download("en_core_web_sm")
spacy_nlp = spacy.load("en_core_web_sm")
_model = None

def set_model(model):
    global _model
    _model = model

generic_recipes = [
    "meatloaf","lasagna","chicken parmesan","kung pao chicken","beef and broccoli","Beef tacos","Mac and cheese","Pot roast",
    "Fried rice","Veggie stir fry","Lentil soup","Grilled cheese","Baked potatoes","Quesadillas","Pancakes","French toast",
    "Rice pilaf","Shepherd’s pie","Sloppy joes","Roasted vegetables","Burgers","Vegetable pasta","Chili","Baked salmon"
]

NON_FOOD_WORDS = set(word.lower() for word in ["recipe","ingredients"])

def remove_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in NON_FOOD_WORDS]
    return ' '.join(filtered_words)

def get_nounchunks(text):
    chunks = []
    doc = spacy_nlp(text)
    for chunk in doc.noun_chunks:
        chunks.append(chunk.text)
    return chunks

def extract_recipe_name(text):
    text_filtered = remove_words(text)
    noun_chunks = get_nounchunks(text_filtered)

    if not noun_chunks:
        return None, 0

    food_kind_emb = _model.encode(generic_recipes, normalize_embeddings=True, convert_to_numpy=True).mean(axis=0)
    ng_embs = _model.encode(noun_chunks, normalize_embeddings=True, convert_to_numpy=True)
    
    sims = np.dot(ng_embs, food_kind_emb.T)
    best_idx = sims.argmax()
    #print("Noun chunks:", noun_chunks)
    #print("Similarities:", sims)
    return noun_chunks[best_idx], sims[best_idx]
#extract_recipe_name("Get me a low calorie meatloaf recipe")


def getRecipeDetails(soup):
    detail_list = []
    l = soup.find('div', class_='mm-recipes-details__content')

    if not l:
        print("Could not find recipe details section")
        return []
    
    for item in l.find_all('div', class_='mm-recipes-details__item'):
        label = item.find('div', class_='mm-recipes-details__label')
        value = item.find('div', class_='mm-recipes-details__value')
        
        detail_dict = {
            'label': label.text.strip() if label else '',
            'value': value.text.strip() if value else '',
        }
        
        detail_list.append(detail_dict)
    
    #for detail in detail_list:
        #print(detail)
    return detail_list
#getRecipeDetails(soup)


def getIngredients(soup):
    ingredients_list = []
    l = soup.find('ul', class_='mm-recipes-structured-ingredients__list')

    if not l:
        print("Could not find ingredients section")
        return []
    
    for item in l.find_all('li'):
        quantity = item.find('span', attrs={'data-ingredient-quantity': 'true'})
        unit = item.find('span', attrs={'data-ingredient-unit': 'true'})
        name = item.find('span', attrs={'data-ingredient-name': 'true'})
        
        ingredient_dict = {
            'quantity': quantity.text.strip() if quantity else '',
            'unit': unit.text.strip() if unit else '',
            'name': name.text.strip() if name else '',
            'full_text': item.text.strip()  # Keep original text too
        }
        
        ingredients_list.append(ingredient_dict)
    
    #for ing in ingredients_list:
        #print(ing)
    return ingredients_list

#getIngredients(soup)


def getDirections(soup):
    dir_list = []
    l = soup.find('ol', class_='comp mntl-sc-block mntl-sc-block-startgroup mntl-sc-block-group--OL')

    if not l:
        print("Could not find recipe details section")
        return []
    
    for i, item in enumerate(l.find_all('li', class_='comp mntl-sc-block mntl-sc-block-startgroup mntl-sc-block-group--LI')):
        direction = item.find('p', class_='comp mntl-sc-block mntl-sc-block-html')
        
        detail_dict = {
            'step': "Step " + str(i),
            'direction': direction.text.strip() if direction else '',
        }
        
        dir_list.append(detail_dict)
    
    #for direc in dir_list:
        #print(direc)
    return dir_list
#getDirections(soup)


def getNutrition(soup):
    nutrition_list = []
    l = soup.find('table', class_='mm-recipes-nutrition-facts-summary__table')

    if not l:
        print("Could not find nutrition details section")
        return []
        
    for item in l.find_all('tr', class_='mm-recipes-nutrition-facts-summary__table-row'):
        label = item.find('td', class_='mm-recipes-nutrition-facts-summary__table-cell text-body-100')
        value = item.find('td', class_='mm-recipes-nutrition-facts-summary__table-cell text-body-100-prominent')
        
        nut_dict = {
            'label': label.text.strip() if label else '',
            'value': value.text.strip() if value else '',
        }
        
        nutrition_list.append(nut_dict)
    
    #for nut in nutrition_list:
        #print(nut)
    return nutrition_list
#getNutrition(soup)

def format_recipe(recipe_details, ingredients, directions, nutrition):
    
    # Format metadata
    details_text = "\n".join(f"{d['label']} {d['value']}" for d in recipe_details)
    
    # Format ingredients
    ingredients_text = "\n".join(f"- {d['full_text']}" for d in ingredients)
    
    # Format directions
    directions_text = "\n".join(f"{d['step']}: {d['direction']}" for d in directions)
    
    # Format nutrition
    nutrition_text = ", ".join(f"{d['label']}: {d['value']}" for d in nutrition)
    
    return f"""Recipe Information:
                {details_text}

                Ingredients:
                {ingredients_text}

                Directions:
                {directions_text}

                Nutrition (per serving):
                {nutrition_text}"""

def recipe_request(text):
    item, _ = extract_recipe_name(text)
    url_query = item.replace(" ", "+")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    search_url = f"https://www.allrecipes.com/search?q={url_query}"

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")

    results = soup.find('a', id="mntl-card-list-card--extendable_1-0",
                    class_="comp mntl-card-list-card--extendable mntl-universal-card mntl-document-card mntl-card card card--no-image")
    
    url = results['href']
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')
    
    recipe_details = getRecipeDetails(soup)
    ingredients = getIngredients(soup)
    directions = getDirections(soup)
    nutrition = getNutrition(soup)

    final_str = format_recipe(recipe_details, ingredients, directions, nutrition)
    return final_str

#user_query = "Get me a recipe for lemon chicken"
#text = recipe_request(user_query)
#print(text)