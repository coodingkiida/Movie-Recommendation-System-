import argparse
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

COLUMNS = ["title","genres","keywords","tagline","cast","director"]

def _clean_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # keep alnum
    s = re.sub(r"\s+", " ", s).strip()
    return s

class Recommender:
    def __init__(self, csv_path: str = "data/movies_small.csv"):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        # ensure required columns
        missing = [c for c in COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        # fill NA and build combined text
        for c in COLUMNS[1:]:  # ignore title
            self.df[c] = self.df[c].fillna("")
        self.df["combined"] = (self.df["genres"]+" "+
                                 self.df["keywords"]+" "+
                                 self.df["tagline"]+" "+
                                 self.df["cast"]+" "+
                                 self.df["director"]).map(_clean_text)
        # vectorize
        self.vectorizer = CountVectorizer(stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.df["combined"])  # sparse matrix
        # build title index
        self.title_to_idx = {t.lower(): i for i, t in enumerate(self.df["title"])}
        # precompute cosine sim
        # For memory efficiency on large sets, compute on-the-fly; with small sample it's fine either way.
        self._cosine_cache = None

    def _cosine_row(self, idx: int) -> np.ndarray:
        if self._cosine_cache is None:
            # lazy compute full matrix if small set; else compute row-wise
            if self.matrix.shape[0] <= 6000:
                self._cosine_cache = cosine_similarity(self.matrix)
            else:
                self._cosine_cache = False  # flag for row-wise computation
        if isinstance(self._cosine_cache, np.ndarray):
            return self._cosine_cache[idx]
        # row-wise
        return cosine_similarity(self.matrix[idx], self.matrix).ravel()

    def recommend(self, title: str, top_k: int = 5):
        key = title.lower().strip()
        if key not in self.title_to_idx:
            # try fuzzy contains
            matches = [t for t in self.title_to_idx if key in t]
            if matches:
                raise KeyError(f"Title not found. Did you mean: {', '.join(sorted(set(m.title() for m in matches))[:5])}?" )
            raise KeyError(f"Title '{title}' not found in dataset.")
        idx = self.title_to_idx[key]
        sims = self._cosine_row(idx)
        # exclude itself
        sims[idx] = -1
        top_idx = np.argsort(-sims)[:top_k]
        results = (self.df.iloc[top_idx]["title"].tolist(),
                   sims[top_idx].round(3).tolist())
        return results

def cli():
    parser = argparse.ArgumentParser(description="Content-based Movie Recommender (cosine similarity)")
    parser.add_argument("--title", required=True, help="Movie title present in the dataset")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--data", default="data/movies_small.csv", help="Path to CSV dataset")
    args = parser.parse_args()

    rec = Recommender(args.data)
    try:
        titles, scores = rec.recommend(args.title, top_k=args.top_k)
    except KeyError as e:
        print(str(e))
        return
    print(f"\nTop {args.top_k} recommendations for: {args.title}\n")
    for rank, (t, s) in enumerate(zip(titles, scores), start=1):
        print(f"{rank:>2}. {t}  (similarity={s})")

if __name__ == "__main__":
    cli()
