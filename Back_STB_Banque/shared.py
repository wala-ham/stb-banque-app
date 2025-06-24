import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.gridspec as gridspec
import joblib # For saving models
from mpl_toolkits.mplot3d import Axes3D # For 3D plots

# --- Data Loading and Preprocessing for AI & Relationship Analysis ---
def load_and_preprocess_data_for_ai_and_relations(client_file='Client - Sheet1.csv', supplier_file='Fournisseur - Sheet1.csv'):
    """
    Charge les données brutes, effectue le prétraitement, agrège les caractéristiques
    (y compris le score de fragmentation), et calcule les composants PCA pour les visualisations d'IA.
    Retourne également les DataFrames bruts traités pour l'analyse des relations.
    """
    df_clients_raw_orig = None
    df_suppliers_raw_orig = None

    try:
        df_clients_raw_orig = pd.read_csv(client_file)
        df_clients_raw_orig['Source'] = 'Client'
        df_suppliers_raw_orig = pd.read_csv(supplier_file)
        df_suppliers_raw_orig['Source'] = 'Fournisseur'
    except FileNotFoundError as e:
        raise RuntimeError(f"Erreur de chargement des données : {e}")

    # 1. Prétraitement de base des DataFrames
    def preprocess_base(df):
        df['Date_Operation_dt_object'] = pd.to_datetime(df['Date_Operation'], errors='coerce')
        if 'Date_Naissance' in df.columns:
            df['Date_Naissance_dt_object'] = pd.to_datetime(df['Date_Naissance'], errors='coerce')
        else:
            df['Date_Naissance_dt_object'] = pd.NaT # S'assurer que la colonne existe même si non présente

        categorical_cols = ['Tranche_Age', 'Civilite', 'Sexe', 'Statut_Civil', 'Segment', 'Situation_Contractuelle', 'Activite_Economique']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Inconnu')

        if 'Somme_Impaye' in df.columns:
            df = df.drop(columns=['Somme_Impaye', 'Nombre_Impaye'], errors='ignore')

        for col in ['Salaire', 'Total montant cheque', 'Montant_cheque', 'Nombre', 'Montant']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        df['is_split_record'] = (df['Nombre'] > 1).astype(int)
        return df

    df_clients_processed = preprocess_base(df_clients_raw_orig.copy())
    df_suppliers_processed = preprocess_base(df_suppliers_raw_orig.copy())

    # 2. Agrégation des caractéristiques (pour le score de fragmentation et autres)
    def aggregate_features(df, id_col, prefix):
        df_valid_dates = df.dropna(subset=['Date_Operation_dt_object'])
        max_date_overall = df_valid_dates['Date_Operation_dt_object'].max()

        features = df.groupby(id_col).agg(
            total_montant=(f'Total montant cheque', 'sum'),
            moy_montant_cheque=(f'Montant_cheque', 'mean'),
            max_montant_cheque=(f'Montant_cheque', 'max'),
            min_montant_cheque=(f'Montant_cheque', 'min'),
            freq_transactions=(f'Nombre', 'sum'),
            nb_operations_distinctes=('Date_Operation_dt_object', 'nunique'),
            recence=('Date_Operation_dt_object', lambda x: (max_date_overall - x.max()).days if pd.notna(x.max()) else np.nan),
            nb_paiements_fractionnes_records=('is_split_record', 'sum')
        ).reset_index()

        features['score_fragmentation'] = features.apply(
            lambda row: row['freq_transactions'] / row['nb_operations_distinctes'] if row['nb_operations_distinctes'] > 0 else 0,
            axis=1
        )
        features['score_fragmentation'] = features['score_fragmentation'].replace([np.inf, -np.inf], np.nan).fillna(0)
        features.columns = [id_col] + [f'{prefix}_{col}' for col in features.columns if col != id_col]
        return features

    client_features_agg = aggregate_features(df_clients_processed, 'Compte_key_Payeur', 'client')
    supplier_features_agg = aggregate_features(df_suppliers_processed, 'Compte_Key', 'supplier')

    # Fusionner des colonnes originales pour le contexte ou le plotting
    client_features_extended = pd.merge(client_features_agg,
                                         df_clients_processed[['Compte_key_Payeur', 'Segment']].drop_duplicates(),
                                         on='Compte_key_Payeur', how='left')
    supplier_features_extended = pd.merge(supplier_features_agg,
                                          df_suppliers_processed[['Compte_Key', 'Activite_Economique']].drop_duplicates(),
                                          on='Compte_Key', how='left')

    # 3. Préparer les données pour la mise à l'échelle, PCA et Isolation Forest
    numerical_cols_clients_for_ai = [col for col in client_features_extended.columns if col.startswith('client_') and col not in ['client_Compte_key_Payeur', 'client_recence']]
    numerical_cols_suppliers_for_ai = [col for col in supplier_features_extended.columns if col.startswith('supplier_') and col not in ['supplier_Compte_Key', 'supplier_recence']]

    for df in [client_features_extended, supplier_features_extended]:
        for col_list in [numerical_cols_clients_for_ai, numerical_cols_suppliers_for_ai]:
            for col in col_list:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

    # 4. Mettre à l'échelle et appliquer la PCA
    scaler_clients = StandardScaler()
    X_clients_scaled = scaler_clients.fit_transform(client_features_extended[numerical_cols_clients_for_ai])
    pca_clients = PCA(n_components=2)
    client_features_extended['pca1'] = pca_clients.fit_transform(X_clients_scaled)[:, 0]
    client_features_extended['pca2'] = pca_clients.transform(X_clients_scaled)[:, 1]
    client_features_extended_data_for_clustering = X_clients_scaled # Conserver les données scalées pour le clustering

    scaler_suppliers = StandardScaler()
    X_suppliers_scaled = scaler_suppliers.fit_transform(supplier_features_extended[numerical_cols_suppliers_for_ai])
    pca_suppliers = PCA(n_components=2)
    supplier_features_extended['pca1'] = pca_suppliers.fit_transform(X_suppliers_scaled)[:, 0]
    supplier_features_extended['pca2'] = pca_suppliers.transform(X_suppliers_scaled)[:, 1]
    supplier_features_extended_data_for_clustering = X_suppliers_scaled # Conserver les données scalées pour le clustering

    # Return all necessary DataFrames and scaled data for clustering
    return df_clients_processed, df_suppliers_processed, client_features_extended, supplier_features_extended, \
           numerical_cols_clients_for_ai, numerical_cols_suppliers_for_ai, \
           client_features_extended_data_for_clustering, supplier_features_extended_data_for_clustering

# --- Clustering Analysis Function ---
def perform_clustering_analysis(X_data, data_for_labels, id_col, prefix):
    """
    Performs Hierarchical and K-Means clustering, evaluates metrics,
    and visualizes results.
    Args:
        X_data (np.array): Scaled numerical data for clustering.
        data_for_labels (pd.DataFrame): DataFrame to add cluster labels to (e.g., client_features_extended).
        id_col (str): The unique identifier column (e.g., 'Compte_key_Payeur' or 'Compte_Key').
        prefix (str): 'client' or 'supplier' for naming conventions.
    Returns:
        pd.DataFrame: The data_for_labels DataFrame with cluster labels added.
    """
    print(f"\n--- Performing Clustering for {prefix.capitalize()}s ---")

    # Ensure X_data is a numpy array for scikit-learn
    X = np.array(X_data)

    if X.shape[0] < 2:
        print(f"Not enough data points for clustering for {prefix}s. Skipping clustering.")
        return data_for_labels

    # 1. PCA for visualization
    pca_viz = PCA(n_components=3)
    X_pca_3d = pca_viz.fit_transform(X)
    X_pca_2d = X_pca_3d[:, :2] # Take first two components for 2D plot

    print(f"Variance expliquée par chaque composante pour {prefix}:", pca_viz.explained_variance_ratio_)

    # --- Hierarchical Clustering ---
    print("\n--- Hierarchical Clustering ---")
    linked = linkage(X, method='ward')

    plt.figure(figsize=(12, 6))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title(f"Dendrogramme - Clustering Hiérarchique ({prefix.capitalize()})")
    plt.xlabel("Observations")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

    # Try different numbers of clusters with silhouette score
    hc_silhouette_scores = []
    print(f"\nSilhouette Scores for HC ({prefix.capitalize()}):")
    for k in range(2, 7):
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X)
        if len(np.unique(labels)) > 1: # Silhouette score requires at least 2 clusters
            score = silhouette_score(X, labels)
            hc_silhouette_scores.append(score)
            print(f"Nombre de clusters = {k} → Silhouette Score = {score:.4f}")
        else:
            print(f"Nombre de clusters = {k} → Not enough distinct clusters for Silhouette Score.")

    # Apply HC with an optimal k (you might choose based on dendrogram/silhouette)
    # For demonstration, let's pick k=3 if available, otherwise the best from range(2,7)
    optimal_k_hc = 3
    if len(hc_silhouette_scores) > 0:
        best_k_idx_hc = np.argmax(hc_silhouette_scores)
        optimal_k_hc = range(2, 7)[best_k_idx_hc]
        print(f"\nOptimal K for HC (based on Silhouette) for {prefix.capitalize()}: {optimal_k_hc}")
    else:
        print(f"\nCould not determine optimal K for HC for {prefix.capitalize()}. Defaulting to {optimal_k_hc}.")


    hc_model = AgglomerativeClustering(n_clusters=optimal_k_hc, linkage='ward')
    hc_labels = hc_model.fit_predict(X)

    # Add labels to data_for_labels
    cluster_names_hc = {i: f"{prefix.capitalize()} Cluster HC {i+1}" for i in range(optimal_k_hc)}
    # Example specific names - you'll refine these after analysis
    if prefix == 'client' and optimal_k_hc == 3:
        cluster_names_hc = {0: "Client Stable", 1: "Client à Risque", 2: "Client Intermédiaire"}
    elif prefix == 'supplier' and optimal_k_hc == 3:
        cluster_names_hc = {0: "Fournisseur Prioritaire", 1: "Fournisseur Moyen", 2: "Fournisseur Marginal"}


    data_for_labels[f'Cluster_Hc_{prefix}'] = hc_labels
    data_for_labels[f'Niveau_Risque_hc_{prefix}'] = data_for_labels[f'Cluster_Hc_{prefix}'].map(cluster_names_hc)

    print(f"\n{prefix.capitalize()} Cluster H-C Distribution:\n", data_for_labels[f'Niveau_Risque_hc_{prefix}'].value_counts())

    # Metrics for HC
    if len(np.unique(hc_labels)) > 1:
        db_score_hc = davies_bouldin_score(X, hc_labels)
        ch_score_hc = calinski_harabasz_score(X, hc_labels)
        print(f"Davies-Bouldin Index (HC) : {db_score_hc:.4f}")
        print(f"Calinski-Harabasz Index (HC) : {ch_score_hc:.4f}")
    else:
        print("Cannot compute Davies-Bouldin or Calinski-Harabasz for HC: Not enough distinct clusters.")


    # Visualisation des Clusters HC (PCA 2D)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=hc_labels, cmap='viridis')
    plt.title(f"Visualisation des Clusters H-C (PCA 2D) - {prefix.capitalize()}")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.colorbar(scatter, label="Cluster")
    plt.show()

    # --- K-Means Clustering ---
    print("\n--- K-Means Clustering ---")
    k_range = range(2, min(11, X.shape[0])) # Ensure k does not exceed number of samples

    if len(k_range) < 2:
        print(f"Not enough data points to test multiple K values for K-Means for {prefix}s.")
        return data_for_labels

    inertia = []
    silhouette_scores_kmeans = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10) # Added n_init
        kmeans.fit(X)
        clusters = kmeans.predict(X)

        inertia.append(kmeans.inertia_)
        if len(np.unique(clusters)) > 1:
            sil_score = silhouette_score(X, clusters)
            silhouette_scores_kmeans.append(sil_score)
        else:
            silhouette_scores_kmeans.append(0) # Or np.nan, handle as appropriate

    # Create a figure with two subplots for metrics
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # 1. Elbow Method Plot
    ax0 = plt.subplot(gs[0])
    ax0.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
    ax0.set_xlabel('Nombre de clusters (k)', fontsize=12)
    ax0.set_ylabel('Inertie', fontsize=12)
    ax0.set_title(f'Méthode du coude ({prefix.capitalize()})', fontsize=14)
    ax0.grid(True)
    for i, val in enumerate(inertia):
        ax0.annotate(f"{val:.0f}", (k_range[i], inertia[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # 2. Silhouette Score Plot
    ax1 = plt.subplot(gs[1])
    ax1.plot(k_range, silhouette_scores_kmeans, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Nombre de clusters (k)', fontsize=12)
    ax1.set_ylabel('Score de silhouette', fontsize=12)
    ax1.set_title(f'Score de silhouette ({prefix.capitalize()})', fontsize=14)
    ax1.grid(True)
    for i, val in enumerate(silhouette_scores_kmeans):
        ax1.annotate(f"{val:.3f}", (k_range[i], silhouette_scores_kmeans[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{prefix}_kmeans_elbow_silhouette.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Determine optimal K for K-Means (example: choose based on your analysis of the plots)
    # For demonstration, let's assume optimal k=3 based on typical elbow/silhouette patterns
    optimal_k_kmeans = 3
    if len(silhouette_scores_kmeans) > 0:
        best_k_idx_kmeans = np.argmax(silhouette_scores_kmeans)
        optimal_k_kmeans = k_range[best_k_idx_kmeans]
        print(f"\nOptimal K for K-Means (based on Silhouette) for {prefix.capitalize()}: {optimal_k_kmeans}")
    else:
        print(f"\nCould not determine optimal K for K-Means for {prefix.capitalize()}. Defaulting to {optimal_k_kmeans}.")


    print(f"\nNombre de clusters choisi pour K-Means: k={optimal_k_kmeans}")
    if optimal_k_kmeans - k_range[0] < len(silhouette_scores_kmeans):
        print(f"Score de silhouette correspondant: {silhouette_scores_kmeans[optimal_k_kmeans - k_range[0]]:.4f}")
    else:
        print("Silhouette score for chosen optimal K not available in calculated range.")


    # Apply K-means with optimal k
    kmeans_model = KMeans(n_clusters=optimal_k_kmeans, random_state=42, n_init=10)
    kmeans_labels = kmeans_model.fit_predict(X) # Fit on original scaled data X for better clustering

    # Add labels to data_for_labels
    cluster_names_kmeans = {i: f"{prefix.capitalize()} Cluster KMeans {i+1}" for i in range(optimal_k_kmeans)}
    # Example specific names - you'll refine these after analysis
    if prefix == 'client' and optimal_k_kmeans == 3:
        cluster_names_kmeans = {0: "Client à Risque", 1: "Client Stable", 2: "Client Intermédiaire"}
    elif prefix == 'supplier' and optimal_k_kmeans == 3:
        cluster_names_kmeans = {0: "Fournisseur Critique", 1: "Fournisseur Opérationnel", 2: "Fournisseur Occasionnel"}

    data_for_labels[f'Cluster_Kmeans_{prefix}'] = kmeans_labels
    data_for_labels[f'Niveau_Risque_Kmeans_{prefix}'] = data_for_labels[f'Cluster_Kmeans_{prefix}'].map(cluster_names_kmeans)

    print(f"\n{prefix.capitalize()} Cluster K-Means Distribution:\n", data_for_labels[f'Niveau_Risque_Kmeans_{prefix}'].value_counts())

    # Save the K-Means model
    joblib.dump(kmeans_model, f'models/kmeans_model_{prefix}.pkl')

    # Metrics for K-Means
    if len(np.unique(kmeans_labels)) > 1:
        dbi_score_kmeans = davies_bouldin_score(X, kmeans_labels)
        kmeans_chi_score = calinski_harabasz_score(X, kmeans_labels)
        print(f"Indice de Davies-Bouldin pour K-Means (k={optimal_k_kmeans}) : {dbi_score_kmeans:.4f}")
        print(f"Indice CHI - K-Means: {kmeans_chi_score:.2f}")
    else:
        print("Cannot compute Davies-Bouldin or Calinski-Harabasz for K-Means: Not enough distinct clusters.")


    # Visualisation des Clusters K-Means (PCA 2D)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title(f"Visualisation des Clusters K-Means (PCA 2D) - {prefix.capitalize()}")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.colorbar(scatter, label="Cluster")
    plt.show()

    # Visualisation des Clusters K-Means (PCA 3D)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=kmeans_labels , cmap='Set2', s=60, alpha=0.7)
    ax.set_xlabel('Composante principale 1')
    ax.set_ylabel('Composante principale 2')
    ax.set_zlabel('Composante principale 3')
    plt.title(f'Visualisation PCA 3D des clusters K-Means (k = {optimal_k_kmeans}) - {prefix.capitalize()}', fontsize=14)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    ax.add_artist(legend1)
    plt.show()

    return data_for_labels

# --- Main execution flow ---
if __name__ == "__main__":
    # Load and preprocess data
    df_clients_processed, df_suppliers_processed, client_features_extended, supplier_features_extended, \
    numerical_cols_clients_for_ai, numerical_cols_suppliers_for_ai, \
    client_features_extended_data_for_clustering, supplier_features_extended_data_for_clustering = \
        load_and_preprocess_data_for_ai_and_relations(client_file='Client - Sheet1.csv', supplier_file='Fournisseur - Sheet1.csv')

    # Perform clustering for clients
    client_features_clustered = perform_clustering_analysis(
        X_data=client_features_extended_data_for_clustering,
        data_for_labels=client_features_extended.copy(), # Pass a copy to avoid modifying original during clustering
        id_col='Compte_key_Payeur',
        prefix='client'
    )
    print("\nClient Features with Clusters:\n", client_features_clustered.head())

    # Perform clustering for suppliers
    supplier_features_clustered = perform_clustering_analysis(
        X_data=supplier_features_extended_data_for_clustering,
        data_for_labels=supplier_features_extended.copy(), # Pass a copy to avoid modifying original during clustering
        id_col='Compte_Key',
        prefix='supplier'
    )
    print("\nSupplier Features with Clusters:\n", supplier_features_clustered.head())

    # You can now use client_features_clustered and supplier_features_clustered
    # for further analysis, reporting, or saving the results.
    # For example, saving to CSV:
    client_features_clustered.to_csv('client_features_with_clusters.csv', index=False)
    supplier_features_clustered.to_csv('supplier_features_with_clusters.csv', index=False)
    print("\nClustering results saved to CSV files.")