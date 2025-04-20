# Recipe Retrieval System Based on Available Ingredients

## Overview
This project aims to develop a system that retrieves relevant recipes based on a user's available ingredients, reducing food waste and enhancing culinary creativity. The system leverages TF-IDF, semantic enhancements, and intelligent ingredient substitutions to provide practical recipe recommendations.

## Team Members
- **Zihan Liu**: System implementation, algorithm development  
- **Xulun Luo**: Evaluation, testing, error analysis  
- **Edward Lu**: Algorithm development, documentation, project management  

## Progress Update

### April 8th
1. We cleaned our data and split our dataset into train, test and dev
2. We implemented the basic TF-IDF to calculate the values we need
3. We implemented the info-retreval based on the TF-IDF scores

### April 14th
1. Edward fine-tuned TF-IDF approach for recipe recommendation.

### April 18th
1. More functions of TF-IDF are added, including filtering the vegan, gluten-free and diabetic.
2. The function of handling misspell is achieved by calculating relative distances between words.
3. Evaluation is done; 13/15 of the test cases passed. 
4. Continue working on the BERT model on our project.
