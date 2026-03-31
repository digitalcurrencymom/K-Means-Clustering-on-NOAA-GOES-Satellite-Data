# kmeans_goes.py
# Author: Kendra
# Purpose: Apply K-Means clustering to NOAA GOES satellite data and save CSV results
# Dataset: NOAA GOES-16, 17, 18 & 19 (AWS Registry of Open Data)

import argparse
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV {path}: {exc}")

    required = ["radiance", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")

    return df


def cluster_goes(data: pd.DataFrame, n_clusters: int, random_state: int = 42) -> pd.DataFrame:
    df = data.copy()
    features = ["radiance", "latitude", "longitude"]

    X = df[features].dropna()
    if X.empty:
        raise ValueError("No rows available after dropping NaN values from radiance/lat/lon columns")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    inertia = float(kmeans.inertia_)
    silhouette = float(silhouette_score(X_scaled, labels)) if n_clusters > 1 and len(X) > n_clusters else float('nan')

    df.loc[X.index, "Cluster"] = labels

    stats = {
        "n_clusters": n_clusters,
        "n_rows": len(X),
        "inertia": inertia,
        "silhouette_score": silhouette,
        "cluster_counts": df["Cluster"].value_counts().to_dict(),
    }

    return df, stats


def plot_clusters(df: pd.DataFrame, show_plot: bool = True):
    if "Cluster" not in df.columns:
        raise ValueError("Dataframe must contain 'Cluster' column to plot")

    plt.figure(figsize=(10, 7))
    plt.scatter(df["longitude"], df["latitude"], c=df["Cluster"], cmap="plasma", s=12, alpha=0.7)
    plt.colorbar(label="Cluster")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("GOES Satellite Data Clustering")
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


def main(input_path: str, output_path: str, n_clusters: int, no_plot: bool):
    data = load_data(input_path)
    clustered, stats = cluster_goes(data, n_clusters=n_clusters)

    print("\nCluster analysis statistics:")
    print(f"- Input rows: {stats['n_rows']}")
    print(f"- Cluster count: {stats['n_clusters']}")
    print(f"- Inertia: {stats['inertia']:.4f}")
    print(f"- Silhouette score: {stats['silhouette_score']:.4f}")
    print("- Rows per cluster:")
    for cluster, count in sorted(stats["cluster_counts"].items()):
        print(f"  Cluster {cluster}: {count}")

    clustered.to_csv(output_path, index=False)
    print(f"\nSaved clustered CSV to: {output_path}")

    if not no_plot:
        plot_clusters(clustered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-means clustering for NOAA GOES CSV dataset")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="goes_clustered.csv", help="Path to output CSV file")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters")
    parser.add_argument("--no-plot", action="store_true", help="Do not display the scatter plot")

    args = parser.parse_args()

    try:
        main(args.input, args.output, args.clusters, args.no_plot)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
