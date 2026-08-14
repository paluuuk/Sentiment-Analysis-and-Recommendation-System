# Airbnb Sentiment & Recommendation Analysis

Applied NLP and recommendation-system project using Airbnb listing, review, and booking data to combine **sentiment analysis, content-based recommendation, price modeling, and demand exploration** in one analytical workflow.

The project is best viewed as an end-to-end data science case study: it starts from raw structured and unstructured data, engineers sentiment and listing features, evaluates predictive relationships, and builds similarity-based recommendations.

## Project Scope

The analysis covers four related problems:

1. **Review sentiment** — extract opinion polarity from guest reviews
2. **Content-based recommendations** — recommend similar listings using listing and sentiment features
3. **Price modeling** — explore non-linear relationships between listing attributes and price
4. **Demand analysis** — examine availability and booking patterns over time

## Analytical Pipeline

```text
Airbnb Listings + Reviews + Calendar Data
                 ↓
          Data Cleaning
                 ↓
     Text / Feature Processing
          ↙             ↘
 Sentiment Features   Listing Features
          ↘             ↙
       Combined Feature Space
          ↓             ↓
 Recommendation     Price / Demand
    Analysis           Analysis
```

## Methods

### Sentiment analysis

Review text is cleaned and processed using NLP techniques including token normalization / lemmatization and polarity scoring. Sentiment values are then aggregated so review information can be combined with structured listing attributes.

### Recommendation system

The project uses a **content-based recommendation approach** based on feature similarity. TF-IDF-style representations and cosine similarity are used to identify listings with similar characteristics rather than relying on collaborative user-user behavior.

### Predictive modeling

The notebook explores regression-based modeling for Airbnb pricing and related outcomes, including non-linear methods such as **Random Forest regression** and **Support Vector Machines** alongside exploratory regression techniques.

### Demand / temporal analysis

Calendar and availability data are used to inspect time-dependent booking patterns and demand behavior.

## Repository Structure

| File | Purpose |
|---|---|
| `Sentiment_Analysis.ipynb` | Main analysis notebook covering preprocessing, NLP, prediction, and recommendations |
| `listings.csv` | Structured Airbnb listing attributes |
| `reviews.csv` | Raw review text |
| `calendar.csv` | Availability / calendar information |
| `polarity_reviews.csv` | Intermediate sentiment-enriched review data |
| `polarity_values_reviews.csv` | Derived / aggregated sentiment outputs |
| `comments.png`, `seattle.jpg` | Analysis / presentation assets |

## Tech Stack

**Python · Pandas · NumPy · scikit-learn · NLP · TF-IDF · cosine similarity · Random Forest · SVM · Jupyter**

## Current Repository Status

This repository preserves the original exploratory analysis and therefore contains raw and generated datasets directly alongside the notebook. That makes it reproducible from the existing files, but it is not the structure I would use for a production data or ML project.

A cleaner engineering version would:

- keep source data outside Git or provide a download script
- separate generated features / outputs from raw inputs
- move reusable preprocessing and recommendation logic into `src/`
- pin dependencies in `requirements.txt` or `pyproject.toml`
- add tests for feature transformations and recommendation logic

## Why keep this project

This is a supporting portfolio project rather than a flagship engineering repository. It demonstrates breadth across **NLP, recommender systems, regression, feature engineering, and exploratory analytics**, complementing the deeper computer-vision and software projects elsewhere on this profile.
