# Sentiment Analysis and Recommendation System

This project implements a sentiment analysis pipeline combined with a content-based recommendation system using Airbnb listing and review data. The objective is to demonstrate how unstructured text data (user reviews) can be leveraged alongside structured listing features to derive insights, predict outcomes, and generate recommendations.

The project was developed as an applied data science and machine learning exercise, integrating natural language processing, regression modeling, and exploratory time series analysis.

---

## Project Objectives

The key objectives of this project are:

- Analyze Airbnb listings to understand pricing and demand patterns
- Perform sentiment analysis on customer reviews to extract opinion polarity
- Build predictive models for price estimation
- Explore booking trends using time series analysis
- Develop a content-based recommendation system informed by sentiment and listing features

---

## Repository Structure

| File / Directory | Description |
|------------------|-------------|
| `Sentiment_Analysis.ipynb` | Main notebook containing data preprocessing, sentiment analysis, modeling, and recommendations |
| `listings.csv` | Airbnb listings dataset with pricing and property features |
| `reviews.csv` | Raw customer review text |
| `calendar.csv` | Booking calendar data used for time series analysis |
| `polarity_reviews.csv` | Reviews with computed sentiment polarity |
| `polarity_values_reviews.csv` | Aggregated sentiment scores |
| `comments.png`, `seattle.jpg` | Visual assets used in analysis |
| `.gitignore` | Git ignore configuration |

---

## Methodology

### 1. Data Preprocessing
- Cleaning missing and inconsistent values
- Normalizing numerical features
- Tokenizing and cleaning review text

### 2. Sentiment Analysis
- Natural Language Processing techniques applied to review text
- Sentiment polarity extraction to classify reviews as positive or negative
- Aggregation of sentiment scores at the listing level

### 3. Predictive Modeling
- Regression models to estimate listing prices
- Random Forest Regressor used to capture non-linear relationships
- Model evaluation using standard regression metrics

### 4. Time Series Analysis
- Analysis of booking trends over time
- Demand estimation based on historical availability data

### 5. Recommendation System
- Content-based recommendation approach
- Listings recommended based on similarity of features and sentiment profiles
- No collaborative filtering or user-user interaction modeling used

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- Jupyter Notebook
- Matplotlib and Seaborn for visualization

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook

### Installation

```bash
git clone https://github.com/paluuuk/Sentiment-Analysis-and-Recommendation-System.git
cd Sentiment-Analysis-and-Recommendation-System

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install pandas numpy scikit-learn matplotlib seaborn nltk
