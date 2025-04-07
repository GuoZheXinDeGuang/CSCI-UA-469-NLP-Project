import json
import re
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

# ======================
# COMPREHENSIVE CLEANING RULES
# ======================

# 1. Measurement removal patterns
MEASUREMENT_PATTERNS = [
    r'^\d+\s*\/\s*\d+',    # Fractions (1/2)
    r'^\d+\.\d+',          # Decimals (1.5)
    r'^\d+\s*-\s*\d+',     # Ranges (1-2)
    r'^[\d½¼¾]+',          # Numbers and fractions
    r'^about\s+\d+',       # "about 2"
    r'^approx(?:imately)?\s+\d+' # "approximately 2"
]

# 2. Unit standardization mapping
UNIT_MAPPING = {
    r'tbsp(?:s?|oons?)': 'tablespoon',
    r'tsp(?:s?|oons?)': 'teaspoon',
    r'oz|ounces?': 'ounce',
    r'lb|lbs|pounds?': 'pound',
    r'c|cs|cups?': 'cup',
    r'pt|pints?': 'pint',
    r'qt|quarts?': 'quart',
    r'gal|gallons?': 'gallon',
    r'ml|milliliters?': 'milliliter',
    r'l|liters?': 'liter',
    r'g|grams?': 'gram',
    r'kg|kilograms?': 'kilogram',
    r'cloves?': 'clove',
    r'pinch(?:es)?': 'pinch',
    r'dash(?:es)?': 'dash'
}

# 3. Ingredient normalization rules
INGREDIENT_NORMALIZATION = {
    # Dairy
    r'whole\s*milk': 'milk',
    r'(?:\w+\s)?(?:low\s*fat|skim|nonfat)\s*milk': 'milk',
    r'heavy\s*cream': 'cream',
    r'(?:unsalted|salted)\s*butter': 'butter',
    
    # Flours/Starches
    r'all[\s-]*purpose\s*flour': 'flour',
    r'plain\s*flour': 'flour',
    r'(?:cake|pastry)\s*flour': 'flour',
    r'corn\s*starch': 'cornstarch',
    
    # Sweeteners
    r'(?:granulated|white)\s*sugar': 'sugar',
    r'confectioners\'\s*sugar|powdered\s*sugar': 'powdered sugar',
    r'(?:light|dark)\s*brown\s*sugar': 'brown sugar',
    
    # Oils/Fats
    r'extra\s*virgin\s*olive\s*oil': 'olive oil',
    r'vegetable\s*oil': 'oil',
    r'canola\s*oil': 'oil',
    
    # Proteins
    r'boneless\s*skinless\s*chicken\s*breasts?': 'chicken breast',
    r'ground\s*(?:beef|turkey|pork)': 'ground meat',
    r'(?:large|extra\s*large)\s*eggs?': 'egg',
    
    # Vegetables
    r'roma\s*tomatoes?': 'tomato',
    r'cherry\s*tomatoes?': 'tomato',
    r'yellow\s*onions?': 'onion',
    r'green\s*bell\s*peppers?': 'bell pepper',
    
    # Herbs/Spices
    r'fresh(?:ly)?\s*chopped\s*(.*)': r'\1',
    r'dried\s*(.*)': r'\1',
    r'minced\s*(.*)': r'\1',
    r'grated\s*(.*)': r'\1',
    
    # Canned/Jarred
    r'(?:canned|jarred)\s*(.*)': r'\1',
    r'in\s*(?:juice|water|oil|syrup)': '',
    r'low\s*sodium\s*(.*)': r'\1',
    
    # General cleaning
    r'\([^)]*\)': '',
    r'[\"\']': '',
    r'\s*-\s*': ' ',
    r'\s+': ' ',
    r'^\s+|\s+$': ''
}

# 4. Core ingredient mapping
CORE_INGREDIENTS = {
    r'chicken\s*curry': ['chicken', 'curry', 'onion', 'garlic', 'ginger'],
    r'chocolate\s*cake': ['flour', 'sugar', 'cocoa', 'egg', 'butter'],
    r'spaghetti\s*bolognese': ['pasta', 'ground meat', 'tomato', 'onion', 'garlic'],
    r'caesar\s*salad': ['lettuce', 'croutons', 'parmesan', 'dressing']
}

# 5. Dietary classification rules
NON_VEGETARIAN = {'meat', 'chicken', 'beef', 'pork', 'lamb', 'fish', 'seafood', 'bacon', 'sausage'}
NON_VEGAN = NON_VEGETARIAN.union({'egg', 'milk', 'cheese', 'butter', 'yogurt', 'cream', 'honey'})
GLUTEN_CONTAINING = {'flour', 'wheat', 'bread', 'pasta', 'barley', 'rye', 'couscous', 'farro', 'semolina'}

# ======================
# CLEANING FUNCTIONS
# ======================

def remove_measurements(text: str) -> str:
    """Remove measurement quantities from ingredient text"""
    for pattern in MEASUREMENT_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def standardize_units(text: str) -> str:
    """Standardize measurement units in ingredient text"""
    for pattern, replacement in UNIT_MAPPING.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_ingredient_name(text: str) -> str:
    """Normalize ingredient names using cleaning rules"""
    text = text.lower()
    text = remove_measurements(text)
    text = standardize_units(text)
    
    for pattern, replacement in INGREDIENT_NORMALIZATION.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text.strip()

def classify_ingredients(recipe_title: str, ingredients: List[str]) -> List[Dict[str, Any]]:
    """Classify ingredients as core or optional"""
    normalized_title = recipe_title.lower()
    core_ingredients = []
    
    # Find matching core ingredients pattern
    for pattern, core_items in CORE_INGREDIENTS.items():
        if re.search(pattern, normalized_title):
            core_ingredients = core_items
            break
    
    classified = []
    for ing in ingredients:
        original = ing
        normalized = normalize_ingredient_name(ing)
        
        # Determine if core ingredient
        is_core = any(
            re.search(r'\b' + re.escape(core) + r'\b', normalized)
            for core in core_ingredients
        )
        
        classified.append({
            'original': original,
            'normalized': normalized,
            'type': 'core' if is_core else 'optional'
        })
    
    return classified

def determine_dietary_tags(ingredients: List[Dict[str, Any]]) -> List[str]:
    """Determine dietary restrictions based on ingredients"""
    tags = set()
    vegetarian = True
    vegan = True
    gluten_free = True
    
    for ing in ingredients:
        normalized = ing['normalized']
        
        # Check for non-vegetarian ingredients
        if any(re.search(r'\b' + re.escape(item) + r'\b', normalized) for item in NON_VEGETARIAN):
            vegetarian = False
            vegan = False
        
        # Check for non-vegan ingredients
        if any(re.search(r'\b' + re.escape(item) + r'\b', normalized) for item in NON_VEGAN):
            vegan = False
        
        # Check for gluten-containing ingredients
        if any(re.search(r'\b' + re.escape(item) + r'\b', normalized) for item in GLUTEN_CONTAINING):
            gluten_free = False
    
    # Set appropriate tags
    if vegan:
        tags.add('vegan')
    elif vegetarian:
        tags.add('vegetarian')
    if gluten_free:
        tags.add('gluten-free')
    
    return sorted(tags)

def determine_course_type(recipe_title: str) -> str:
    """Determine course type based on recipe title"""
    title_lower = recipe_title.lower()
    
    dessert_keywords = {'dessert', 'cake', 'pie', 'cookie', 'brownie', 'ice cream', 'pudding'}
    side_keywords = {'salad', 'appetizer', 'starter', 'soup', 'bread', 'roll', 'dip'}
    
    if any(keyword in title_lower for keyword in dessert_keywords):
        return 'dessert'
    if any(keyword in title_lower for keyword in side_keywords):
        return 'side'
    return 'main'

def clean_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and process a single recipe record"""
    # Clean basic fields
    cleaned = {
        'id': int(recipe['id']),
        'title': recipe['title'].strip(),
        'link': recipe.get('link', '').strip(),
        'source': 'Gathered' if recipe.get('source') == 0 else 'Recipes1M'
    }
    
    # Clean ingredients
    if isinstance(recipe['ingredients'], str):
        try:
            ingredients = ast.literal_eval(recipe['ingredients'])
        except:
            ingredients = []
    else:
        ingredients = recipe['ingredients']
    
    cleaned['ingredients'] = classify_ingredients(recipe['title'], ingredients)
    
    # Clean directions
    if isinstance(recipe['directions'], str):
        try:
            directions = ast.literal_eval(recipe['directions'])
        except:
            directions = []
    else:
        directions = recipe['directions']
    cleaned['directions'] = [step.strip() for step in directions if step.strip()]
    
    # Clean NER tags
    if isinstance(recipe['NER'], str):
        try:
            ner_tags = ast.literal_eval(recipe['NER'])
        except:
            ner_tags = []
    else:
        ner_tags = recipe['NER']
    cleaned['NER'] = [tag.strip() for tag in ner_tags if tag.strip()]
    
    # Add derived fields
    cleaned['dietary_tags'] = determine_dietary_tags(cleaned['ingredients'])
    cleaned['course_type'] = determine_course_type(recipe['title'])
    
    return cleaned

def process_dataset(input_csv: str, output_files: Dict[str, str]):
    """
    Process the entire dataset from CSV to cleaned JSON files
    
    Args:
        input_csv: Path to input CSV file
        output_files: Dictionary with output file paths for train/dev/test
    """
    # Read CSV file
    df = pd.read_csv(input_csv)
    recipes = df.to_dict('records')
    
    # Clean all recipes
    cleaned_recipes = [clean_recipe(recipe) for recipe in recipes]
    
    # Split dataset (70% train, 15% dev, 15% test)
    train, temp = train_test_split(cleaned_recipes, test_size=0.3, random_state=42)
    dev, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # Save to JSON files
    for split, data in zip(['train', 'dev', 'test'], [train, dev, test]):
        with open(output_files[split], 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset processed. Files created: {list(output_files.values())}")

def process_first_10_rows(input_csv: str, output_file: str = 'first_10_recipes.json'):
    """
    Process and save only the first 10 rows for testing
    
    Args:
        input_csv: Path to input CSV file
        output_file: Output JSON file path
    """
    # Read first 10 rows
    df = pd.read_csv(input_csv, nrows=10)
    recipes = df.to_dict('records')
    
    # Clean recipes
    cleaned_recipes = [clean_recipe(recipe) for recipe in recipes]
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_recipes, f, indent=2, ensure_ascii=False)
    
    print(f"First 10 recipes saved to {output_file}")

# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    # Configuration
    config = {
        'input_csv': 'RecipeNLG_dataset.csv',
        'output_files': {
            'train': 'train_recipes.json',
            'dev': 'dev_recipes.json',
            'test': 'test_recipes.json'
        },
        'test_output': 'first_10_recipes.json'
    }
    
    # Uncomment to process full dataset
    process_dataset(config['input_csv'], config['output_files'])
    
    # Process first 10 rows only (for testing)
    # process_first_10_rows(config['input_csv'], config['test_output'])