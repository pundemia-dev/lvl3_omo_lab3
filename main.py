import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    MeanShift,
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    SpectralClustering
)
import hdbscan
from sklearn.metrics import adjusted_rand_score

def load_data():
    files = list(Path("QCM Sensor Alcohol Dataset").glob("*.csv"))
    if not files:
        raise FileNotFoundError("Dataset files not found in 'QCM Sensor Alcohol Dataset' directory.")
    
    df_list = []
    for f in files:
        df = pd.read_csv(f, sep=';')
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    X = full_df.iloc[:, :10].values
    Y_onehot = full_df.iloc[:, 10:15].values
    y = np.argmax(Y_onehot, axis=1)
    return X, y

def main():
    X_raw, y = load_data()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca_2d = PCA(n_components=2)
    X_pca2 = pca_2d.fit_transform(X_scaled)
    
    pca_3d = PCA(n_components=3)
    X_pca3 = pca_3d.fit_transform(X_scaled)
    
    pca_opt = PCA(n_components=0.95)
    X_pca_opt = pca_opt.fit_transform(X_scaled)
    
    datasets = {
        f"Original ({X_scaled.shape[1]}D)": X_scaled,
        "PCA 2D": X_pca2,
        "PCA 3D": X_pca3,
        f"PCA Opt ({X_pca_opt.shape[1]}D)": X_pca_opt
    }
    
    models = {
        "k-Means (k=5)": KMeans(n_clusters=5, random_state=42, n_init='auto'),
        "Agglomerative (k=5)": AgglomerativeClustering(n_clusters=5),
        "Mean Shift": MeanShift(),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
        "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=5),
        "OPTICS": OPTICS(min_samples=5),
        "Affinity Propagation": AffinityPropagation(random_state=42),
        "Spectral Clustering (k=5)": SpectralClustering(n_clusters=5, random_state=42, assign_labels='discretize')
    }
    
    results = []
    
    for data_name, X in datasets.items():
        for model_name, model in models.items():
            start_time = time.time()
            try:
                labels = model.fit_predict(X)
                exec_time = time.time() - start_time
                
                ari = adjusted_rand_score(y, labels)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                results.append({
                    "Dataset": data_name,
                    "Algorithm": model_name,
                    "Clusters Found": n_clusters,
                    "ARI": round(ari, 4),
                    "Time (s)": round(exec_time, 4)
                })
            except Exception as e:
                results.append({
                    "Dataset": data_name,
                    "Algorithm": model_name,
                    "Clusters Found": "Error",
                    "ARI": "Error",
                    "Time (s)": "Error"
                })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv("clustering_results.csv", index=False)

if __name__ == "__main__":
    main()