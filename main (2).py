"""
Zomato Restaurant Data Analysis
---------------------------------
End-to-end exploratory data analysis of a Zomato restaurant dataset,
completed as part of the Cognifyz Technologies Data Analysis Internship.

This script performs:
  Level 1 - Data exploration & cleaning
  Level 2 - Cuisine & geographic analysis
  Level 3 - Ratings & correlation analysis

Usage:
    python main.py --data data/zomato.csv

Author: Lokavarapu Govindu
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUTPUT_DIR = "outputs"


# ------------------------------------------------------------------
# Level 1: Data Exploration & Cleaning
# ------------------------------------------------------------------
def load_and_clean_data(filepath):
    """Load the dataset and perform basic cleaning."""
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, encoding="latin-1")

    print(f"Initial shape: {df.shape}")
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Drop rows missing critical fields
    critical_cols = [c for c in ["Restaurant Name", "Cuisines", "Aggregate rating"] if c in df.columns]
    df = df.dropna(subset=critical_cols)

    # Standardize text columns
    for col in ["Cuisines", "City", "Rating text"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    print(f"\nShape after cleaning: {df.shape}")
    return df


def descriptive_summary(df):
    """Print descriptive statistics for numeric columns."""
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="number").T)


# ------------------------------------------------------------------
# Level 2: Cuisine & Geographic Analysis
# ------------------------------------------------------------------
def analyze_cuisines(df, top_n=10):
    """Identify the most common cuisines and their market share."""
    # Cuisines column often contains comma-separated multiple cuisines per restaurant
    cuisine_series = df["Cuisines"].str.split(",").explode().str.strip()
    cuisine_counts = cuisine_series.value_counts().head(top_n)
    cuisine_share = (cuisine_series.value_counts(normalize=True) * 100).head(top_n)

    print("\n--- Top Cuisines by Restaurant Count ---")
    print(cuisine_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index, palette="viridis")
    plt.title(f"Top {top_n} Most Common Cuisines")
    plt.xlabel("Number of Restaurants")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    _save_fig("top_cuisines.png")

    return cuisine_counts, cuisine_share


def analyze_city_distribution(df, top_n=10):
    """Determine which cities have the highest restaurant density."""
    city_counts = df["City"].value_counts().head(top_n)

    print("\n--- Top Cities by Restaurant Count ---")
    print(city_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=city_counts.values, y=city_counts.index, palette="mako")
    plt.title(f"Top {top_n} Cities by Restaurant Count")
    plt.xlabel("Number of Restaurants")
    plt.ylabel("City")
    plt.tight_layout()
    _save_fig("top_cities.png")

    return city_counts


def analyze_price_range(df):
    """Explore price range / cost-for-two distribution."""
    if "Price range" not in df.columns:
        return None

    plt.figure(figsize=(8, 5))
    sns.countplot(x="Price range", data=df, palette="crest")
    plt.title("Restaurant Distribution by Price Range")
    plt.xlabel("Price Range (1 = Cheapest, 4 = Most Expensive)")
    plt.ylabel("Number of Restaurants")
    plt.tight_layout()
    _save_fig("price_range_distribution.png")


# ------------------------------------------------------------------
# Level 3: Ratings & Correlation Analysis
# ------------------------------------------------------------------
def analyze_votes_vs_ratings(df):
    """Analyze the relationship between votes and ratings."""
    if "Votes" not in df.columns or "Aggregate rating" not in df.columns:
        return None

    correlation = df["Votes"].corr(df["Aggregate rating"])
    print(f"\n--- Correlation between Votes and Ratings: {correlation:.3f} ---")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Votes", y="Aggregate rating", data=df, alpha=0.4)
    plt.title(f"Votes vs. Aggregate Rating (Correlation = {correlation:.2f})")
    plt.xlabel("Number of Votes")
    plt.ylabel("Aggregate Rating")
    plt.tight_layout()
    _save_fig("votes_vs_ratings.png")

    return correlation


def analyze_rating_text(df):
    """Breakdown of rating text categories (proxy for review sentiment)."""
    if "Rating text" not in df.columns:
        return None

    rating_text_counts = df["Rating text"].value_counts()
    print("\n--- Rating Text Category Breakdown ---")
    print(rating_text_counts)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=rating_text_counts.values, y=rating_text_counts.index, palette="flare")
    plt.title("Restaurant Count by Rating Text Category")
    plt.xlabel("Number of Restaurants")
    plt.ylabel("Rating Text")
    plt.tight_layout()
    _save_fig("rating_text_breakdown.png")

    return rating_text_counts


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def _save_fig(filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Zomato Restaurant Data Analysis")
    parser.add_argument(
        "--data", type=str, default="data/zomato.csv",
        help="Path to the Zomato dataset CSV file (default: data/zomato.csv)"
    )
    args = parser.parse_args()

    # Level 1
    df = load_and_clean_data(args.data)
    descriptive_summary(df)

    # Level 2
    analyze_cuisines(df)
    analyze_city_distribution(df)
    analyze_price_range(df)

    # Level 3
    analyze_votes_vs_ratings(df)
    analyze_rating_text(df)

    print(f"\nAnalysis complete. Charts saved to the '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    main()
