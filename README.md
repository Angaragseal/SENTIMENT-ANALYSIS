# SENTIMENT-ANALYSIS

**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: ANGARAG SEAL  
**INTERN ID**: CT04DN666  
**DOMAIN**: DATA ANALYTICS  
**DURATION**: 4 WEEKS  
**MENTOR**: NEELA SANTOSH

 ---
# Sentiment Analysis on Tweets Dataset

## Project Overview

This project performs **sentiment analysis** on a large collection of tweets to extract and analyze public opinion expressed on social media. Sentiment analysis, or opinion mining, is a crucial technique in natural language processing (NLP) that classifies textual data based on emotional tone — typically categorized as positive, negative, or neutral.

With the exponential growth of social media platforms like Twitter, analyzing vast amounts of unstructured tweet data has become essential for businesses, governments, and researchers to understand public sentiment on products, services, events, and social issues in real time.

## Dataset Description

The dataset consists of approximately 3,500 tweets from users across various countries. Each record contains the following fields:

- `textID`: Unique identifier for the tweet  
- `text`: The content of the tweet  
- `sentiment`: Sentiment label (e.g., positive, negative, neutral)  
- `Time of Tweet`: Timestamp when the tweet was posted  
- `Age of User`: Age group of the user  
- `Country`: User’s country of origin  
- `Population - 2020`: Population of the user’s country (2020 estimate)  
- `Land Area (Km²)`: Land area of the country  
- `Density (P/Km²)`: Population density of the country  

This rich dataset allows not only sentiment classification but also demographic and geographic analysis, such as examining how sentiment varies by country, age group, or time.

## Objectives

The key objectives of this project are:

- To preprocess and clean tweet text data for accurate analysis  
- To perform exploratory data analysis (EDA) and visualize data distribution and sentiment trends  
- To analyze sentiment distribution across countries and age groups  
- To identify correlations between sentiment and demographic/geographic features  
- To build and evaluate machine learning models for sentiment classification (optional/future scope)  

## Methodology

1. **Data Loading and Cleaning:**  
   The dataset is loaded from a local CSV file (`test.csv`). Missing values and duplicates are handled. Text preprocessing includes tokenization, stopwords removal, and normalization.

2. **Exploratory Data Analysis (EDA):**  
   Various visualizations are created to understand the data better:  
   - Sentiment distribution using count plots and pie charts  
   - Top countries by tweet counts and their sentiment distribution  
   - Age group distribution  
   - Time-based trends of sentiment  
   - Demographic and geographic correlation plots  

3. **Data Visualization:**  
   The project uses matplotlib and seaborn libraries to create clear, side-by-side visualizations that provide insights into the data without overwhelming the viewer.

4. **Insights and Reporting:**  
   Based on the visualizations and statistics, key insights are drawn regarding sentiment trends and their possible drivers.

## Technologies Used

- **Python 3.x**  
- **Pandas** for data manipulation  
- **Matplotlib** and **Seaborn** for visualization  
- **NLTK** for text preprocessing (stopwords removal)  

## Usage

1. Ensure the dataset CSV file (`test.csv`) is in the project directory.  
2. Run the script to load data, perform analysis, and generate visualizations.  
3. Visualizations appear side by side for better comparison and insight extraction.

## Future Work

- Implement machine learning models for automated sentiment classification.  
- Perform time-series analysis to observe sentiment shifts over time.  
- Explore topic modeling to identify prevalent themes in tweets.  
- Integrate more advanced NLP techniques like word embeddings and transformers.

## Conclusion

This project demonstrates how social media data can be effectively analyzed to understand public sentiment on a large scale. By combining sentiment classification with demographic and geographic context, stakeholders can obtain actionable insights to make informed decisions, shape policies, or improve services.

---

##Output
