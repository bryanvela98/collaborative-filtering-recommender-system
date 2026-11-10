# Collaborative Filtering Recommender System

This project implements and evaluates collaborative-filtering recommenders
using the MovieLens 100K dataset. It was developed for CSCI 6517 Recommender
Systems and includes both a guided tutorial and a complete analysis notebook.

## What is included

- User-user and item-item collaborative filtering with cosine similarity
- Popularity and user-average baselines
- Rating prediction with RMSE
- Top-k recommendation evaluation with precision and recall
- Five-fold cross-validation with confidence intervals
- A sparsity analysis comparing users above and below a 50-rating threshold
- Qualitative inspection of movie recommendations

## Key findings

| Evaluation | Best approach | Result |
| --- | --- | ---: |
| RMSE | Item-item cosine | 1.0201 |
| Precision@5 | User-user cosine | 0.5558 |
| Recall@5 | User-user cosine | 0.4863 |

The results show why rating accuracy and ranking quality should not be treated
as the same objective. Item-item filtering produced the lowest RMSE, while
user-user filtering produced the strongest top-5 ranking metrics. For users
with fewer than 50 ratings, item-item filtering was more robust
(RMSE 1.0388 versus 1.0510 for user-user filtering).

## Repository contents

- [`collaborative_filtering_analysis.ipynb`](./collaborative_filtering_analysis.ipynb):
  analysis, implementations, experiments, and discussion
- [`collaborative_filtering_tutorial.ipynb`](./collaborative_filtering_tutorial.ipynb):
  introductory notebook
  covering the recommender workflow
- [`CONCEPTS.md`](./CONCEPTS.md): formulas and implementation notes
- [`requirements.txt`](./requirements.txt): Python dependencies

## Running locally

1. Download the MovieLens 100K dataset from
   [GroupLens](https://grouplens.org/datasets/movielens/100k/).
2. Extract the archive so the project contains an `ml-100k/` directory.
3. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Open either notebook in Jupyter or VS Code and run the cells in order.

The dataset is intentionally ignored by Git because the MovieLens usage
license does not permit redistribution. Please review the dataset license
before using it outside this project.

## Dataset citation

Harper, F. Maxwell, and Joseph A. Konstan. “The MovieLens Datasets: History
and Context.” *ACM Transactions on Interactive Intelligent Systems*, 2015.
See the dataset's included `README` for the full citation and license.
