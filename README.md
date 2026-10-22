# Airbnb Sentiment and Recommendation Analysis

An exploratory Python case study combining listing attributes, review text, and calendar records to study pricing, review sentiment, demand patterns, and content-based listing recommendations.

## Technical components

| Area | Implementation in the notebook |
| --- | --- |
| Review sentiment | NLTK VADER and TextBlob polarity/subjectivity |
| Text features | Word/ngram exploration and TF-IDF over listing names and descriptions |
| Recommendations | Linear kernel on normalized TF-IDF vectors to obtain cosine similarities |
| Price modeling | Linear/regression experiments, polynomial features, Random Forest, and SVM-based exploration |
| Temporal analysis | Calendar availability and price patterns |

These are related analytical experiments, not a deployed recommendation service or a demonstrated causal model of demand.

## Files

- [`Sentiment_Analysis.ipynb`](Sentiment_Analysis.ipynb): analysis source and explanatory notes.
- [`requirements.txt`](requirements.txt): packages imported by the notebook.
- `listings.csv`, `reviews.csv`, `calendar.csv`: input data used by the original experiment.
- `polarity_reviews.csv`, `polarity_values_reviews.csv`: intermediate sentiment artifacts.

Generated notebook outputs have been cleared so the analysis source is practical to inspect on GitHub. The original dataset files remain in the repository; the repository is still larger than a code-only package.

## Run locally

```bash
git clone https://github.com/paluuuk/Sentiment-Analysis-and-Recommendation-System.git
cd Sentiment-Analysis-and-Recommendation-System
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
jupyter notebook Sentiment_Analysis.ipynb
```

Run from the repository root so the relative CSV paths resolve. Download the NLTK resources requested by the notebook when prompted. Language-detection and text-analysis dependencies are included in the dependency list.

## Evaluation and limitations

The notebook explores multiple prediction tasks and data transformations. Random train/test splits and correlations from a single marketplace dataset should not be interpreted as a production forecast, a causal demand estimate, or a generalization guarantee.

The TF-IDF vectorizer uses a valid minimum document count of one. Dependencies are not version-locked, and a complete rerun is still required to verify compatibility and reproduce metrics in a new environment. No new model scores are claimed by this cleanup.

**Stack:** Python, pandas, NumPy, scikit-learn, NLTK, TextBlob, Plotly, Matplotlib, seaborn, wordcloud.

## Related engineering projects

- [Customer segmentation](https://github.com/paluuuk/Customer-segmentation)
- [Deal Preflight product engineering](https://github.com/paluuuk/deal-preflight)

[View all selected engineering work](https://github.com/paluuuk)

