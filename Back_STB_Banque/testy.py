# =================================================================================
# SCRIPT FINAL DE SEGMENTATION CLIENTÈLE - VERSION AVEC NOMS MÉTIERS FIXES
#
# Ce script utilise une liste de noms de segments prédéfinis par l'utilisateur
# et les affiche correctement sur le graphique.
# =================================================================================

# --- Section 1: Import des librairies nécessaires ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


# --- Section 2: Fonction de chargement et de préparation des données ---
def load_and_preprocess_data(client_file='Client - Sheet1.csv'):
    try:
        df_raw = pd.read_csv(client_file)
    except FileNotFoundError:
        raise SystemExit(f"ERREUR CRITIQUE : Le fichier '{client_file}' est introuvable.")

    df_raw['Date_Operation_dt_object'] = pd.to_datetime(df_raw['Date_Operation'], errors='coerce')

    numeric_cols = ['Salaire', 'Total montant cheque', 'Montant_cheque', 'Nombre', 'Somme_Impaye', 'Nombre_Impaye']
    for col in numeric_cols:
        if col in df_raw.columns:
            median_val = df_raw[col].median()
            df_raw[col].fillna(median_val, inplace=True)
        else:
            df_raw[col] = 0

    max_date = df_raw['Date_Operation_dt_object'].max() if not df_raw['Date_Operation_dt_object'].dropna().empty else pd.Timestamp.now()

    agg_dict = {
        'total_montant_depense': ('Total montant cheque', 'sum'),
        'freq_transactions': ('Compte_key_Payeur', 'count'),
        'recence_jours': ('Date_Operation_dt_object', lambda x: (max_date - x.max()).days if pd.notna(x.max()) else -1),
        'total_impaye': ('Somme_Impaye', 'sum'),
    }

    client_features = df_raw.groupby('Compte_key_Payeur').agg(**agg_dict).reset_index()

    demo_cols = ['Salaire']
    demo_data = df_raw[['Compte_key_Payeur'] + demo_cols].groupby('Compte_key_Payeur').first().reset_index()
    client_features = pd.merge(client_features, demo_data, on='Compte_key_Payeur', how='left')
    
    features_for_ml = client_features.drop(columns=['Compte_key_Payeur'])
    features_for_ml = pd.get_dummies(features_for_ml, drop_first=True)
    features_for_ml.fillna(features_for_ml.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_for_ml)
    
    return client_features, X_scaled


# --- Section 3: Exécution du pipeline principal ---
if __name__ == "__main__":
    
    df_clients_final, X_scaled = load_and_preprocess_data()
    print("✅ Données chargées et prêtes.\n")

    N_CLUSTERS = 3
    hc = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    hc_labels = hc.fit_predict(X_scaled)
    df_clients_final['Cluster_ID'] = hc_labels

    # --- ÉVALUATION TECHNIQUE DU CLUSTERING FINAL ---
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    print("\n--- Évaluation Technique du Clustering (K=3) ---")
# Le Score de Silhouette mesure à quel point les clusters sont denses et bien séparés.
    sil_score = silhouette_score(X_scaled, hc_labels)
    print(f"Score de Silhouette : {sil_score:.4f} (Plus c'est proche de 1, mieux c'est)")

# L'indice de Davies-Bouldin mesure la similarité moyenne entre chaque cluster et son plus proche voisin.
    db_score = davies_bouldin_score(X_scaled, hc_labels)
    print(f"Indice de Davies-Bouldin : {db_score:.4f} (Plus c'est proche de 0, mieux c'est)")

# L'indice de Calinski-Harabasz est le ratio de la dispersion entre les clusters sur la dispersion à l'intérieur des clusters.
    ch_score = calinski_harabasz_score(X_scaled, hc_labels)
    print(f"Indice de Calinski-Harabasz : {ch_score:.2f} (Plus c'est élevé, mieux c'est)")


# La suite du script continue ici...
# --- ÉTAPE 4: DÉFINITION DYNAMIQUE ET FORMATAGE DES TITRES ---
    print("\n--- Analyse des caractéristiques et identification des segments... ---")

    # --- ÉTAPE 4: DÉFINITION FIXE ET FORMATAGE DES TITRES ---
    print("--- Analyse des caractéristiques des clusters... ---")
    analysis_cols = ['total_montant_depense', 'freq_transactions', 'recence_jours', 'total_impaye', 'Salaire']
    cluster_analysis_median = df_clients_final.groupby('Cluster_ID')[analysis_cols].median()

    # --- MODIFICATION ---
    # La logique "intelligente" a été retirée.
    # Nous utilisons maintenant VOTRE dictionnaire de noms fixes.
    cluster_names_map = {
        0: "Client Stable",
        1: "Client à Risque",
        2: "Client Intermédiaire"
    }

    print("\nUtilisation du mapping de noms prédéfini :")
    print(cluster_names_map)
    print("\nIMPORTANT : Vérifiez que les chiffres dans le tableau ci-dessous correspondent bien à ces noms.")
    print("------------------------------------------------------------------------------------")
    print(cluster_analysis_median) # Affiche le tableau avec les numéros pour vérification
    print("------------------------------------------------------------------------------------\n")
    
    # Création des étiquettes composées pour le graphique
    plot_labels_map = {}
    for cluster_id, name in cluster_names_map.items():
        plot_labels_map[cluster_id] = f"Cluster {cluster_id}\n{name}"

    # On renomme les colonnes du tableau d'analyse avec ces nouvelles étiquettes
    cluster_analysis_renamed_for_plot = cluster_analysis_median.rename(columns=plot_labels_map)

    # --- VISUALISATION FINALE ---
    print("\n--- Génération du graphique final avec les titres exacts... ---")
    plt.figure(figsize=(16, 8))
    
    sns.heatmap(
        cluster_analysis_renamed_for_plot.T,
        cmap='BuPu',
        annot=True,
        fmt='.0f',
        linewidths=1,
        linecolor='white'
    )
    plt.title('Portrait-Robot des Segments de Clientèle', fontsize=18, weight='bold')
    plt.ylabel('Caractéristiques Clés', fontsize=14)
    plt.xlabel('Segments de Clientèle Identifiés', fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=12, weight='bold')
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- ÉTAPE 5: APPLICATION FINALE ET EXPLOITATION ---
    df_clients_final['Nom_Segment'] = df_clients_final['Cluster_ID'].map(cluster_names_map)
    segment_counts = df_clients_final['Nom_Segment'].value_counts()
    
    print("\n--- Exploitation Stratégique ---")
    print("\nRépartition finale des clients par segment :")
    print(segment_counts)

    print("\n--- FIN DU SCRIPT ---")