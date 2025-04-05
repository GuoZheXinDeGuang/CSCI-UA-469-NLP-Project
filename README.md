# Recipe Retrieval System Based on Available Ingredients

## Overview
This project aims to develop a system that retrieves relevant recipes based on a user's available ingredients, reducing food waste and enhancing culinary creativity. The system leverages TF-IDF, semantic enhancements, and intelligent ingredient substitutions to provide practical recipe recommendations.

## Team Members
- **Zihan Liu**: System implementation, algorithm development  
- **Xulun Luo**: Evaluation, testing, error analysis  
- **Edward Lu**: Algorithm development, documentation, project management  

## Key Features
- **TF-IDF-based retrieval**: Matches recipes to ingredients efficiently.  
- **Query expansion**: Identifies synonyms and substitutes (e.g., "almond milk" for "dairy milk").  
- **Weighted importance**: Prioritizes core ingredients (e.g., "chicken" in "Chicken Curry").  
- **Substitution suggestions**: Recommends valid ingredient replacements.  

## Dataset
- **RecipeNLG**: Recipes with structured ingredients and steps.  
- **Cleaning steps**: Normalization, annotation (core/optional ingredients), and splitting (70/15/15 train/dev/test).  

## Evaluation Metrics
1. **Precision@5**: Relevance of top 5 recommendations.  
2. **Ingredient Utilization Score (IUS)**: Measures usage efficiency of core/optional ingredients.  
3. **Substitution Validity Rate (SVR)**: Expert-rated appropriateness of substitutions.  
