# 🎬 Movie Recommendation System (Content-Based)

A **content-based movie recommendation system** built with **Python, Pandas, and scikit-learn**.  
It suggests similar movies using combined text features (**genres, keywords, tagline, cast, director**) and **cosine similarity**.

> This repo includes a small sample dataset (`data/movies_small.csv`) so the code runs out-of-the-box.  
> For better results, replace it with a larger dataset (e.g., TMDB movies metadata).

---

## 🚀 Features
- Text feature engineering with smart cleaning
- Bag-of-words vectorization (`CountVectorizer`)
- Fast cosine similarity lookups
- CLI usage: `python src/recommend.py --title "Inception" --top_k 5`

## 🧱 Project Structure
```
movie-recommendation-system/
│── data/
│   └── movies_small.csv
│── src/
│   └── recommend.py
│── notebook/
│   └── Movie_Recommendation_Quickstart.ipynb
│── requirements.txt
│── README.md
│── .gitignore
```

## 📦 Setup
```bash
# 1) Clone and enter
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system

# 2) (Optional) Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

## ▶️ Run (CLI)
```bash
python src/recommend.py --title "The Dark Knight" --top_k 5
```

## 🧪 Try in Python
```python
from src.recommend import Recommender
rec = Recommender("data/movies_small.csv")
rec.recommend("Inception", top_k=5)
```

## 🔄 Use a Bigger Dataset
Replace `data/movies_small.csv` with your own CSV that has these columns:
`title, genres, keywords, tagline, cast, director` (fill missing with empty strings).  
Then update the path when instantiating `Recommender`.

## 📝 License
MIT
