# Collaborative Filtering — Concept Cheat Sheet

Goal: get every recommender working end-to-end in
`collaborative_filtering_analysis.ipynb`, with
**P@k ≥ 0.1** for at least the similarity-based (cosine) recommenders. Below are
the concepts/formulas you need for each stub, plus the most common reasons P@k
comes out near 0 (which is almost always a bug, not a "this algorithm is just bad"
situation).

---

## Data Loading (from tutorial)

```python
fields = ['userID', 'itemID', 'rating', 'timestamp']
ratingDF = pd.read_csv('ml-100k/u.data', sep='\t', names=fields)

fieldsMovies = ['movieID', 'movieTitle', 'releaseDate', 'videoReleaseDate', 'IMDbURL',
                'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy',
                'crime', 'documentary', 'drama', 'fantasy', 'filmNoir', 'horror',
                'musical', 'mystery', 'romance', 'sciFi', 'thriller', 'war', 'western']
moviesDF = pd.read_csv('ml-100k/u.item', sep='|', names=fieldsMovies, encoding='latin-1')

numUsers = len(ratingDF.userID.unique())   # 943
numItems = len(ratingDF.itemID.unique())   # 1682
```

Train/test splits are pre-made files: `u1.base`/`u1.test` … `u5.base`/`u5.test`
(80K train / 20K test each), plus `ua`/`ub` variants.

---

## Q1(a) `dataPreprocessor`

Build a dense `num_users x num_items` matrix where `matrix[u, i]` = rating user
`u+1` gave item `i+1` (IDs are 1-indexed in the data, 0-indexed in the matrix),
and `0` everywhere there is no rating.

- Easiest vectorized approach:
  ```python
  matrix = np.zeros((num_users, num_items))
  matrix[df.userID.values - 1, df.itemID.values - 1] = df.rating.values
  ```
- Equivalent alternative: `df.pivot_table(index='userID', columns='itemID', values='rating').fillna(0)`,
  then reindex to guarantee the full `num_users x num_items` shape (pivot can
  produce a smaller matrix if some user/item ID never appears in a fold!).
  This is exactly the bug that breaks cross-validation — **always reindex/pad
  to the full (num_users, num_items) shape**, since train folds won't contain
  every user/item.

## Q1(b) `useraverage` and `popularity`

- **useraverage**: for each user, predicted rating for every item = that user's
  mean of their *observed* (nonzero) ratings. Watch out for users with zero
  ratings in a fold (mean of empty slice → NaN). Compute:
  ```python
  sums = train_matrix.sum(axis=1)
  counts = (train_matrix != 0).sum(axis=1)
  counts[counts == 0] = 1          # avoid div-by-zero
  user_avg = sums / counts          # shape (num_users,)
  predictionMatrix = np.tile(user_avg.reshape(-1, 1), (1, num_items))
  ```
- **popularity**: the tutorial defines popularity as **fraction of raters who
  liked** the item (`numLiked / numRated`, where liked = rating ≥ 4), broadcast
  as the same score for every user. Vectorized equivalent:
  ```python
  liked_mask = (train_matrix >= 4).astype(float)
  rated_mask = (train_matrix != 0).astype(float)
  rated_counts = rated_mask.sum(axis=0)                        # (num_items,)
  liked_counts = liked_mask.sum(axis=0)                        # (num_items,)
  item_popularity = np.where(rated_counts > 0, liked_counts / rated_counts, 0)
  predictionMatrix = np.tile(item_popularity.reshape(1, -1), (num_users, 1))
  ```
  Alternative simpler definition: raw count of raters `(train_matrix != 0).sum(axis=0)` — both preserve relative ranking for P@k/R@k.
  Note this is *not* a "rating" prediction (so RMSE on it is somewhat
  meaningless — that's the answer to Q4(b): popularity has no notion of a
  1–5 rating scale, so RMSE against true ratings isn't a fair comparison,
  though P@k/R@k still make sense since they only need a ranking).

---

## Q2(a) `euclidean` similarity

`pairwise_distances(matrix, metric='euclidean')` returns distances in
`[0, ∞)`. You must convert distance → similarity in `[0, 1]`. **Simple
negation does not work** (it gives values in `(-∞, 0]`). The standard
transform used in the lab/assignment is:

```python
distance_matrix = pairwise_distances(matrix, metric='euclidean')
similarity_matrix = 1 / (1 + distance_matrix)
```

This maps `distance=0 → similarity=1` and `distance→∞ → similarity→0`, and is
always non-negative — satisfying the FAQ's "similarity should never be
negative" rule.

### Cosine similarity (tutorial pattern)

The tutorial computes cosine similarity directly as:
```python
similarity = 1 - pairwise_distances(matrix, metric='cosine')
```
Because `pairwise_distances(..., metric='cosine')` returns *cosine distance* = `1 - cosine_similarity`, so `1 - distance = cosine_similarity` directly. Result is in `[-1, 1]`, but for rating vectors it's almost always non-negative.

### kNearestNeighbor filter (from tutorial)

Zero out all but the top-k neighbors per row (keeps only the k most similar):
```python
def kNearestNeighbor(sim, k):
    sim = sim.copy()
    m, n = sim.shape
    for i in range(m):
        sim[i, sim[i].argsort()[:-k]] = 0
    return sim
```
Apply before prediction: `similarity = kNearestNeighbor(similarity, k=50)`.

## Q2(b) `somethingelse`

Pick another metric from `pairwise_distances` (e.g. `'cityblock'`/Manhattan,
`'cosine'` variants, `'correlation'` for Pearson-like similarity, `'jaccard'`
for binarized data). Apply the **same kind of distance→similarity
transformation** appropriate to that metric's range:
- For unbounded distances (Manhattan, Euclidean-like): `1 / (1 + d)`.
- For metrics already bounded in `[0, 1]` or `[0, 2]` (cosine distance,
  correlation distance): `1 - d` (cosine distance is `1 - cosine similarity`,
  so `1 - d` recovers cosine similarity directly — that's literally what the
  given `cosine` function does).
Justify your choice in one sentence (e.g. "Pearson correlation distance
captures whether two users rate items in a *consistent relative order* even if
their rating scales differ, which raw cosine/Euclidean don't account for").

---

## Q3(a) `predict_all` — THE crucial step for getting P@k ≥ 0.1

This is where most low-precision results come from. You need a **vectorized
weighted-average** prediction:

### User-user CF
For each user `u` and item `i`:
```
pred[u, i] = (Σ_v sim[u, v] * train_matrix[v, i]) / (Σ_v |sim[u, v]| over v who rated i)
```
Vectorized form:
```python
sim_matrix = self.method(train_matrix)                 # (num_users, num_users)
rated_mask = (train_matrix != 0).astype(float)         # (num_users, num_items)
numerator = sim_matrix @ train_matrix                  # (num_users, num_items)
denominator = sim_matrix @ rated_mask                  # (num_users, num_items)  <-- only sum sims of users who rated item i
denominator[denominator == 0] = 1e-9                   # avoid divide-by-zero / NaN
prediction = numerator / denominator
```

### Item-item CF
Symmetric — similarity is between items (`num_items x num_items`), and you
weight across the items each user has rated:
```python
sim_matrix = self.method(train_matrix.T)               # (num_items, num_items)
rated_mask = (train_matrix != 0).astype(float)         # (num_users, num_items)
numerator = train_matrix @ sim_matrix                  # (num_users, num_items)
denominator = rated_mask @ sim_matrix                  # (num_users, num_items)
denominator[denominator == 0] = 1e-9
prediction = numerator / denominator
```

Then `self.__model = prediction` (store as the private model attribute the
rest of the class expects).

### Vectorized version with cold-start handling (from tutorial)

The tutorial's `vectorizedUserSimRecSys` adds a fallback for items that have
zero predicted scores (no neighbor overlap):
```python
useraverage = train_matrix.sum(axis=1) / (train_matrix != 0).sum(axis=1)
columns = prediction.sum(axis=0)
# For items with no predictions, fall back to user average
prediction[:, columns == 0] += np.expand_dims(useraverage, axis=1)
```
This prevents all-zero columns from collapsing ranking for those items.

### Why this matters for reaching P@k ≥ 0.1
`patk` ranks items by `nlargest(k, ...)` on your **raw prediction values**.
If your prediction matrix is:
- all zeros, or
- constant across items for a user (e.g., you forgot to divide / broadcast
  wrong), or
- full of `NaN` (division by zero when a user/item has no neighbors with
  overlapping ratings),

then `nlargest` returns an essentially arbitrary ordering (ties broken by
index order), and precision collapses toward random-guess level — typically
**far below 0.1** (random P@5 against ~16% "liked" items ≈ 0.05–0.08, and NaNs
make `nlargest` behave unpredictably/badly).

**Checklist to hit ≥ 0.1:**
1. Confirm `prediction` has *real, varying* floating point scores (not all
   identical, not NaN): `np.isnan(prediction).any()` should be `False`,
   and `prediction.std(axis=1)` should be clearly nonzero for most users.
2. Always guard the denominator (`denominator[denominator == 0] = small_eps`
   or use `np.divide(..., out=np.zeros_like(...), where=denom!=0)`).
3. Use `train_matrix != 0` as the "has this user rated this item" mask — do
   **not** use the full similarity-weighted sum over all items/users
   regardless of whether they were actually rated, or you'll dilute the signal
   toward zero.
4. Don't forget: `evaluate_test`/`patk`/`ratk` already handle masking out
   training entries — you don't need to zero those out yourself in
   `predict_all` (the assignment's NOTE explicitly says not to worry about
   it).
5. Sanity-check on a tiny slice before running full CV: pick one user, print
   their top-10 predicted items and check the scores are sensible/distinct.

---

## Q3(b) — User-user vs item-item RMSE

Think about **support**: average ratings-per-user vs. average ratings-per-item
(`100000/943 ≈ 106` per user vs. `100000/1682 ≈ 59` per item in the full
dataset — compute the *train-fold* numbers for your actual answer). More
support for a similarity computation ⇒ less noisy similarity estimates ⇒
better predictions. Whichever side (users or items) has denser overlap
typically yields the stronger CF variant — reason from your computed numbers,
not from the example numbers above.

---

## Q4 — Performance comparison

- **(b)** Baselines like `popularity` produce scores that aren't on the 1–5
  rating scale (they're counts), so **RMSE is not meaningful** for it (it'll
  be huge/uninterpretable) — but P@k/R@k are fine since ranking only needs
  relative order. `useraverage` *can* be evaluated on RMSE.
- **(c)/(d)** RMSE rewards getting the *exact rating value* close; P@k/R@k
  reward getting the *relative order of liked vs. unliked* items right. A
  method can nail one and miss the other (e.g., a method that always predicts
  close to the global mean can have low RMSE but terrible ranking because it
  never differentiates items).

---

## RMSE computation (tutorial pattern)

Only evaluate on items that appear in the test set (nonzero entries):
```python
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(pred, test):
    predItems = pred[test.nonzero()].flatten()
    testItems = test[test.nonzero()].flatten()
    return sqrt(mean_squared_error(predItems, testItems))
```
Tutorial results for reference: `useraverage` RMSE ≈ 1.06, user-similarity RMSE ≈ 1.01–1.02 across folds.

---

## P@k and R@k — "liked" threshold

The tutorial (and assignment) define **liked = rating ≥ 4**. The metric implementations:
```python
from heapq import nlargest

vf = np.vectorize(lambda x: 1 if x >= 4 else 0)

# Top-K items by predicted score
topK = nlargest(k, range(len(userVector)), userVector.take)

# Ground-truth liked items from test set
userTestVector = vf(testSet[userID, :]).nonzero()[0]

precision = len([item for item in topK if item in userTestVector]) / len(topK)
recall    = len([item for item in topK if item in userTestVector]) / len(userTestVector)
```
Skip users with no test ratings (`len(userTestVector) == 0`). Average over all remaining users.

---

## Cross-validation over 5 folds

```python
datasetsFileNames = [('u1.base','u1.test'), ('u2.base','u2.test'),
                     ('u3.base','u3.test'), ('u4.base','u4.test'), ('u5.base','u5.test')]

for trainFile, testFile in datasetsFileNames:
    trainDF = pd.read_csv(os.path.join(MOVIELENS_DIR, trainFile), sep='\t', names=fields)
    testDF  = pd.read_csv(os.path.join(MOVIELENS_DIR, testFile),  sep='\t', names=fields)
    trainMatrix = buildUserItemMatrix(trainDF, numUsers, numItems)
    testMatrix  = buildUserItemMatrix(testDF,  numUsers, numItems)
    # ... fit and evaluate
```
Average RMSE/P@k across folds for a stable estimate.

---

## Making recommendations for a specific user (tutorial)

```python
def userTopK(prediction, moviesDF, userID, k):
    userVector = prediction[userID + 1, :]           # +1 because 0-indexed matrix, 1-indexed IDs
    topK = nlargest(k, range(len(userVector)), userVector.take)
    return [moviesDF[moviesDF.movieID == x + 1]["movieTitle"].values[0] for x in topK]
```

---

## Q5 — Item-item similarity inspection

After you have `sim_matrix = SimBasedRecSys.cosine(train_matrix.T)` (items x
items), look up a movie's row, then:
```python
top5 = np.argsort(-sim_matrix[item_idx])[1:6]   # skip index 0 = the movie itself
```
Map indices back to titles via `u.item` (columns: item id | title | ...).
Remember cosine similarity here is driven purely by **co-rating patterns**
(which users rated both movies, and how), not by genre/plot — so "similar"
movies may just be ones popular with the same user community.

---

## Q6 — User segmentation by activity

- Build the histogram with `plt.hist(train_matrix.astype(bool).sum(axis=1))`
  (ratings per user) or `np.histogram(...)`.
- Pick `τ` by eyeballing where the distribution has a long tail vs. a dense
  cluster (commonly somewhere around the median, e.g., ~50–100 ratings for
  ML-100k).
- Filter the **test** dataframe to users above/below `τ` (based on counts
  computed from the **train** matrix — don't leak test info into the
  threshold split), run `cv.run(...)` separately on each subset.
- Expect: heavy users → denser neighborhoods → better similarity estimates →
  usually lower RMSE / higher P@k for both CF variants; light users → sparse
  neighborhoods → noisier similarity → worse performance, especially for
  user-user CF (fewer co-rated items to compute reliable similarity).

---

## General debugging tips

- Run the validation cells (`testDataPreprocessor`, `testPopularityRecSys`,
  etc. — note: these reference `validateXxx` functions but are *called* as
  `testXxx()`; you'll need to either rename the validators to `testXxx` or fix
  the call sites for them to run) after each stub to catch shape/NaN bugs
  early, before running the full 5-fold CV (which is slow).
- `np.any(sim_matrix < 0)` should be `False` for a correct similarity matrix.
- If P@k ever returns exactly the same number for very different algorithms,
  that's a strong signal your `prediction` matrices are degenerate (e.g., all
  zeros) and `nlargest` is just returning index order both times.
