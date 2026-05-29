# 🎬 CineMatch - Movie Recommendation System

A modern, Netflix-inspired movie recommendation system built with Python and Streamlit.

## ✨ Features

- **Hybrid Recommendation Engine**: Combines content-based and collaborative filtering
- **Multi-Language Support**: English, Telugu, Hindi, Tamil, Korean, Japanese, Spanish, French
- **Beautiful UI**: Dark mode, glassmorphism cards, smooth animations
- **Smart Filters**: Language, genre, and rating filters
- **TMDB Integration**: Movie posters and details via TMDB API
- **Favorites**: Mark and track your favorite movies

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the Dataset
```bash
python download_data.py
```
Download the TMDB 5000 Movies dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place `tmdb_5000_movies.csv` as `data/movies.csv`.

### 3. (Optional) Get TMDB API Key
Get a free API key at [themoviedb.org](https://www.themoviedb.org/settings/api) for movie posters.

### 4. Run the App
```bash
streamlit run app.py
```

## 📁 Project Structure
```
movie-recommender/
├── app.py              # Main Streamlit app
├── model.py            # Recommendation engine
├── utils.py            # Helper functions & CSS
├── download_data.py    # Dataset download helper
├── data/               # Dataset folder
│   └── movies.csv      # TMDB movies dataset
├── assets/             # Custom assets
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🧠 How It Works

### Content-Based Filtering
Uses TF-IDF vectorization on movie genres, overview, and language, then computes cosine similarity to find similar movies.

### Collaborative Filtering
Weights recommendations by popularity using a Bayesian weighted rating formula (IMDB's approach).

### Hybrid Approach
Combines both methods: content similarity narrows candidates, then popularity weighting ranks them.

## 🌐 Deploy on Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy!

## 📄 License
MIT License
