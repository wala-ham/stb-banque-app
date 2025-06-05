import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io
from flask import Flask, jsonify, send_file, Blueprint, request

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

    # Retourner tous les DataFrames nécessaires
    return df_clients_processed, df_suppliers_processed, client_features_extended, supplier_features_extended, \
           numerical_cols_clients_for_ai, numerical_cols_suppliers_for_ai, \
           client_features_extended_data_for_clustering, supplier_features_extended_data_for_clustering

# --- Initialisation Globale des Données et Modèles AI ---
# Ces variables globales contiendront les DataFrames traités, les composants PCA et les scores/prédictions.
df_clients_global_for_relations = None # Raw processed clients for common entity check
df_suppliers_global_for_relations = None # Raw processed suppliers for common entity check
client_data_for_ai_global = None # Client data with aggregated features, PCA, and anomaly prediction
supplier_data_for_ai_global = None # Supplier data with aggregated features, PCA, and anomaly prediction
client_data_scaled_for_clustering_global = None # Scaled client data for KMeans input
supplier_data_scaled_for_clustering_global = None # Scaled supplier data for KMeans input
_client_features_for_if_names_global = None # Features used for Isolation Forest (clients)
_supplier_features_for_if_names_global = None # Features used for Isolation Forest (suppliers)


ai_bp = Blueprint('ai_analysis', __name__, url_prefix='/ai')

try:
    df_clients_global_for_relations, df_suppliers_global_for_relations, \
    client_data_for_ai_global, supplier_data_for_ai_global, \
    _client_features_for_if_names_global, _supplier_features_for_if_names_global, \
    client_data_scaled_for_clustering_global, supplier_data_scaled_for_clustering_global = load_and_preprocess_data_for_ai_and_relations()

    # Entraîner les modèles Isolation Forest et ajouter la prédiction d'anomalie
    iforest_clients = IsolationForest(random_state=42, contamination=0.05)
    client_data_for_ai_global['is_anomaly'] = iforest_clients.fit_predict(
        client_data_for_ai_global[_client_features_for_if_names_global]
    )

    iforest_suppliers = IsolationForest(random_state=42, contamination=0.05)
    supplier_data_for_ai_global['is_anomaly'] = iforest_suppliers.fit_predict(
        supplier_data_for_ai_global[_supplier_features_for_if_names_global]
    )

    print("Modèles d'IA (Détection d'Anomalies) entraînés avec succès.")

except RuntimeError as e:
    print(f"Erreur lors du chargement des données ou de l'entraînement des modèles d'IA : {e}")
    # Réinitialisation des variables globales en cas d'erreur
    df_clients_global_for_relations = None
    df_suppliers_global_for_relations = None
    client_data_for_ai_global = None
    supplier_data_for_ai_global = None
    client_data_scaled_for_clustering_global = None
    supplier_data_scaled_for_clustering_global = None
    _client_features_for_if_names_global = None
    _supplier_features_for_if_names_global = None

# --- API Endpoints pour la Détection d'Anomalies (Images) ---

@ai_bp.route('/anomalies/clients/pca-image', methods=['GET'])
def get_client_anomalies_pca_image():
    """Génère et renvoie une image PNG des anomalies clients dans l'espace PCA 2D."""
    if client_data_for_ai_global is None: return jsonify({"error": "Données IA clients non disponibles."}), 500
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='pca1', y='pca2', hue='is_anomaly', data=client_data_for_ai_global,
                        palette={1: 'blue', -1: 'red'}, alpha=0.7, ax=ax)
        ax.set_title('Détection d\'Anomalies chez les Clients (PCA 2D)')
        ax.legend(title='Est Anomalie', labels=['Normal', 'Anomalie'])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image PCA des anomalies clients : {str(e)}"}), 500


@ai_bp.route('/anomalies/fournisseurs/pca-image', methods=['GET'])
def get_supplier_anomalies_pca_image():
    """Génère et renvoie une image PNG des anomalies fournisseurs dans l'espace PCA 2D."""
    if supplier_data_for_ai_global is None: return jsonify({"error": "Données IA fournisseurs non disponibles."}), 500
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='pca1', y='pca2', hue='is_anomaly', data=supplier_data_for_ai_global,
                        palette={1: 'blue', -1: 'red'}, alpha=0.7, ax=ax)
        ax.set_title('Détection d\'Anomalies chez les Fournisseurs (PCA 2D)')
        ax.legend(title='Est Anomalie', labels=['Normal', 'Anomalie'])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image PCA des anomalies fournisseurs : {str(e)}"}), 500


@ai_bp.route('/anomalies/clients/fragmentation-anomaly-image', methods=['GET'])
def get_client_fragmentation_anomaly_image():
    """Génère et renvoie une image PNG du Total Montant vs Score de Fragmentation client, avec anomalies."""
    if client_data_for_ai_global is None: return jsonify({"error": "Données IA clients non disponibles."}), 500
    try:
        plot_df = client_data_for_ai_global.dropna(subset=['client_total_montant', 'client_score_fragmentation', 'is_anomaly'])
        if plot_df.empty: return jsonify({"error": "Pas de données valides pour le plot de fragmentation vs anomalie client."}), 404
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='client_total_montant', y='client_score_fragmentation', hue='is_anomaly',
                        sizes=(20, 400), data=plot_df, palette={1: 'blue', -1: 'red'}, alpha=0.7, ax=ax)
        ax.set_title('Clients : Total Montant vs Score de Fragmentation (Anomalies en Rouge)')
        ax.set_xlabel('Total Montant Cheque Client (échelle log)')
        ax.set_ylabel('Score de Fragmentation (échelle log)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(title='Est Anomalie', labels=['Normal', 'Anomalie'], bbox_to_anchor=(1.05, 1), loc='upper left')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image de fragmentation vs anomalie client : {str(e)}"}), 500

# --- Endpoints Clustering K-Means (Images) ---

@ai_bp.route('/clustering/clients-image', methods=['GET'])
def get_client_clustering_image():
    """Génère et renvoie une image PNG du clustering K-Means des clients dans l'espace PCA 2D."""
    if client_data_for_ai_global is None or client_data_scaled_for_clustering_global is None:
        return jsonify({"error": "Données clients pour le clustering non disponibles."}), 500
    n_clusters = request.args.get('k', default=3, type=int)
    if not (2 <= n_clusters <= 8):
        return jsonify({"error": "Le nombre de clusters (k) doit être entre 2 et 8."}), 400
    try:
        kmeans_clients = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        client_data_for_ai_global['cluster'] = kmeans_clients.fit_predict(client_data_scaled_for_clustering_global)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=client_data_for_ai_global,
                        palette='viridis', alpha=0.8, ax=ax, legend='full')
        ax.set_title(f'Clusters de Clients (K-Means sur PCA 2D, k={n_clusters})')
        ax.legend(title='Cluster')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image de clustering client : {str(e)}"}), 500


@ai_bp.route('/clustering/fournisseurs-image', methods=['GET'])
def get_supplier_clustering_image():
    """Génère et renvoie une image PNG du clustering K-Means des fournisseurs dans l'espace PCA 2D."""
    if supplier_data_for_ai_global is None or supplier_data_scaled_for_clustering_global is None:
        return jsonify({"error": "Données fournisseurs pour le clustering non disponibles."}), 500
    n_clusters = request.args.get('k', default=3, type=int)
    if not (2 <= n_clusters <= 8):
        return jsonify({"error": "Le nombre de clusters (k) doit être entre 2 et 8."}), 400
    try:
        kmeans_suppliers = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        supplier_data_for_ai_global['cluster'] = kmeans_suppliers.fit_predict(supplier_data_scaled_for_clustering_global)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=supplier_data_for_ai_global,
                        palette='viridis', alpha=0.8, ax=ax, legend='full')
        ax.set_title(f'Clusters de Fournisseurs (K-Means sur PCA 2D, k={n_clusters})')
        ax.legend(title='Cluster')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image de clustering fournisseur : {str(e)}"}), 500

# --- Endpoints Analyse de la Relation Clients-Fournisseurs (Images) ---

@ai_bp.route('/relations/correlation-matrix-image', methods=['GET'])
def get_common_entities_correlation_matrix_image():
    """
    Génère et renvoie une image PNG de la matrice de corrélation
    des comportements de paiement pour les entités communes.
    """
    if df_clients_global_for_relations is None or df_suppliers_global_for_relations is None or \
       client_data_for_ai_global is None or supplier_data_for_ai_global is None:
        return jsonify({"error": "Données nécessaires pour l'analyse des relations non disponibles."}), 500

    try:
        clients_payers_ids = df_clients_global_for_relations['Compte_key_Payeur'].unique()
        suppliers_payers_ids = df_suppliers_global_for_relations['Compte_key_Payeur'].unique()

        common_payers_ids = np.intersect1d(clients_payers_ids, suppliers_payers_ids)

        if len(common_payers_ids) < 2: # Nécessite au moins 2 entités communes pour calculer la corrélation
            return jsonify({"error": "Pas assez d'entités communes pour calculer une matrice de corrélation significative (moins de 2)."}), 404

        # Filtrer les caractéristiques clients pour les entités communes
        common_client_features = client_data_for_ai_global[
            client_data_for_ai_global['Compte_key_Payeur'].isin(common_payers_ids)
        ].copy()

        # Agréger les caractéristiques des fournisseurs lorsqu'ils agissent comme payeurs
        supplier_payer_features = df_suppliers_global_for_relations.groupby('Compte_key_Payeur').agg(
            total_montant_fournisseur_payeur=('Total montant cheque', 'sum'),
            moy_montant_cheque_fournisseur_payeur=('Montant_cheque', 'mean'),
            nb_operations_distinctes_fournisseur_payeur=('Date_Operation_dt_object', 'nunique'),
            freq_transactions_fournisseur_payeur=('Nombre', 'sum'),
            nb_paiements_fractionnes_records_fournisseur_payeur=('is_split_record', 'sum')
        ).reset_index()

        supplier_payer_features['score_fragmentation_fournisseur_payeur'] = \
            supplier_payer_features.apply(
                lambda row: row['freq_transactions_fournisseur_payeur'] / row['nb_operations_distinctes_fournisseur_payeur'] if row['nb_operations_distinctes_fournisseur_payeur'] > 0 else 0,
                axis=1
            )
        supplier_payer_features['score_fragmentation_fournisseur_payeur'] = \
            supplier_payer_features['score_fragmentation_fournisseur_payeur'].replace([np.inf, -np.inf], np.nan).fillna(0)


        merged_common_entities = pd.merge(
            common_client_features[['Compte_key_Payeur', 'client_total_montant', 'client_moy_montant_cheque', 'client_score_fragmentation']],
            supplier_payer_features,
            left_on='Compte_key_Payeur',
            right_on='Compte_key_Payeur',
            how='inner'
        )

        if merged_common_entities.empty:
            return jsonify({"error": "Aucune donnée fusionnée pour les entités communes."}), 404

        correlation_cols = [
            'client_total_montant', 'client_moy_montant_cheque', 'client_score_fragmentation',
            'total_montant_fournisseur_payeur', 'moy_montant_cheque_fournisseur_payeur',
            'score_fragmentation_fournisseur_payeur'
        ]
        df_corr = merged_common_entities[correlation_cols].copy()
        df_corr = df_corr.fillna(df_corr.median()) # Remplir les NaN dans les données de corrélation

        if df_corr.empty or len(df_corr) < 2:
            return jsonify({"error": "Pas assez de données valides dans les entités communes pour calculer la matrice de corrélation."}), 404

        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title('Matrice de Corrélation des Comportements de Paiement (Entités Communes)')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig_corr)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image de la matrice de corrélation : {str(e)}"}), 500


@ai_bp.route('/relations/total-amount-scatter-image', methods=['GET'])
def get_common_entities_total_amount_scatter_image():
    """
    Génère et renvoie une image PNG d'un graphique de dispersion
    comparant le 'Total montant cheque' Client vs Fournisseur pour les entités communes.
    """
    if df_clients_global_for_relations is None or df_suppliers_global_for_relations is None or \
       client_data_for_ai_global is None or supplier_data_for_ai_global is None:
        return jsonify({"error": "Données nécessaires pour l'analyse des relations non disponibles."}), 500

    try:
        clients_payers_ids = df_clients_global_for_relations['Compte_key_Payeur'].unique()
        suppliers_payers_ids = df_suppliers_global_for_relations['Compte_key_Payeur'].unique()
        common_payers_ids = np.intersect1d(clients_payers_ids, suppliers_payers_ids)

        if len(common_payers_ids) < 1:
            return jsonify({"error": "Aucune entité commune trouvée pour le graphique de dispersion des montants."}), 404

        common_client_features = client_data_for_ai_global[
            client_data_for_ai_global['Compte_key_Payeur'].isin(common_payers_ids)
        ].copy()

        supplier_payer_features = df_suppliers_global_for_relations.groupby('Compte_key_Payeur').agg(
            total_montant_fournisseur_payeur=('Total montant cheque', 'sum')
        ).reset_index()

        merged_common_entities = pd.merge(
            common_client_features[['Compte_key_Payeur', 'client_total_montant']],
            supplier_payer_features,
            left_on='Compte_key_Payeur',
            right_on='Compte_key_Payeur',
            how='inner'
        )

        if merged_common_entities.empty:
            return jsonify({"error": "Aucune donnée fusionnée pour les entités communes pour le graphique des montants."}), 404

        fig_scatter1, ax_scatter1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='client_total_montant', y='total_montant_fournisseur_payeur', data=merged_common_entities, alpha=0.7, ax=ax_scatter1)
        ax_scatter1.set_title('Total Montant Client vs Total Montant Fournisseur (Payeur)')
        ax_scatter1.set_xlabel('Total Montant Cheque Client')
        ax_scatter1.set_ylabel('Total Montant Cheque Fournisseur (Payeur)')
        ax_scatter1.set_xscale('log')
        ax_scatter1.set_yscale('log')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig_scatter1)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image de dispersion des montants : {str(e)}"}), 500


@ai_bp.route('/relations/fragmentation-scatter-image', methods=['GET'])
def get_common_entities_fragmentation_scatter_image():
    """
    Génère et renvoie une image PNG d'un graphique de dispersion
    comparant le 'Score de Fragmentation' Client vs Fournisseur pour les entités communes.
    """
    if df_clients_global_for_relations is None or df_suppliers_global_for_relations is None or \
       client_data_for_ai_global is None or supplier_data_for_ai_global is None:
        return jsonify({"error": "Données nécessaires pour l'analyse des relations non disponibles."}), 500

    try:
        clients_payers_ids = df_clients_global_for_relations['Compte_key_Payeur'].unique()
        suppliers_payers_ids = df_suppliers_global_for_relations['Compte_key_Payeur'].unique()
        common_payers_ids = np.intersect1d(clients_payers_ids, suppliers_payers_ids)

        if len(common_payers_ids) < 1:
            return jsonify({"error": "Aucune entité commune trouvée pour le graphique de dispersion de fragmentation."}), 404

        common_client_features = client_data_for_ai_global[
            client_data_for_ai_global['Compte_key_Payeur'].isin(common_payers_ids)
        ].copy()

        supplier_payer_features = df_suppliers_global_for_relations.groupby('Compte_key_Payeur').agg(
            freq_transactions_fournisseur_payeur=('Nombre', 'sum'),
            nb_operations_distinctes_fournisseur_payeur=('Date_Operation_dt_object', 'nunique')
        ).reset_index()

        supplier_payer_features['score_fragmentation_fournisseur_payeur'] = \
            supplier_payer_features.apply(
                lambda row: row['freq_transactions_fournisseur_payeur'] / row['nb_operations_distinctes_fournisseur_payeur'] if row['nb_operations_distinctes_fournisseur_payeur'] > 0 else 0,
                axis=1
            )
        supplier_payer_features['score_fragmentation_fournisseur_payeur'] = \
            supplier_payer_features['score_fragmentation_fournisseur_payeur'].replace([np.inf, -np.inf], np.nan).fillna(0)


        merged_common_entities = pd.merge(
            common_client_features[['Compte_key_Payeur', 'client_score_fragmentation']],
            supplier_payer_features,
            left_on='Compte_key_Payeur',
            right_on='Compte_key_Payeur',
            how='inner'
        )

        if merged_common_entities.empty:
            return jsonify({"error": "Aucune donnée fusionnée pour les entités communes pour le graphique de fragmentation."}), 404

        fig_scatter2, ax_scatter2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='client_score_fragmentation', y='score_fragmentation_fournisseur_payeur', data=merged_common_entities, alpha=0.7, ax=ax_scatter2)
        ax_scatter2.set_title('Score de Fragmentation Client vs Fournisseur (Payeur) pour Entités Communes')
        ax_scatter2.set_xlabel('Score de Fragmentation Client')
        ax_scatter2.set_ylabel('Score de Fragmentation Fournisseur (en tant que Payeur)')
        ax_scatter2.set_xscale('log')
        ax_scatter2.set_yscale('log')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig_scatter2)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image de dispersion de fragmentation : {str(e)}"}), 500


# --- Pour exécuter ce fichier indépendamment (pour le test) ---
if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(ai_bp) # Enregistrer le blueprint
    app.run(debug=True, port=5001) # Exécuter sur un port différent (ex: 5001)