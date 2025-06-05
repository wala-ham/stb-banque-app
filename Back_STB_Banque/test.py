import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import google.generativeai as genai
import os

# --- Configuration Streamlit ---
st.set_page_config(layout="wide", page_title="Analyse PFE - Clients & Fournisseurs")

# Initialize chat history for LLM (still relevant for the LLM chat feature)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Gemini API Configuration (TOP LEVEL) ---
try:
    genai.configure(api_key="AIzaSyBoCEyoVBzlT-Toc7j8I52qa9mjTM38UYY")
except KeyError:
    st.error("Google API Key not found in Streamlit Secrets or environment variables. "
             "Please add it to your `.streamlit/secrets.toml` file "
             "as `GOOGLE_API_KEY=\"YOUR_API_KEY\"` or set it as an environment variable.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# Initialize the generative model (TOP LEVEL)
MODEL_NAME = "gemini-2.0-flash"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"Error initializing Gemini model '{MODEL_NAME}': {e}. "
             "Please check your API key, model name, and internet connectivity.")
    st.stop()

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    """Charge les données et effectue le prétraitement initial."""

    # Chargement des données clients
    try:
        df_clients = pd.read_csv('Client - Sheet1.csv')
        df_clients['Source'] = 'Client'
    except FileNotFoundError:
        st.error("Erreur: Le fichier 'Client - Sheet1.csv' n'a pas été trouvé. Assurez-vous qu'il est dans le même dossier que l'application.")
        st.stop()

    # Chargement des données fournisseurs
    try:
        df_suppliers = pd.read_csv('Fournisseur - Sheet1.csv')
        df_suppliers['Source'] = 'Fournisseur'
    except FileNotFoundError:
        st.error("Erreur: Le fichier 'Fournisseur - Sheet1.csv' n'a pas été trouvé. Assurez-vous qu'il est dans le même dossier que l'application.")
        st.stop()

    # Prétraitement commun
    def preprocess(df):
        df['Date_Operation'] = pd.to_datetime(df['Date_Operation'], errors='coerce')
        if 'Date_Naissance' in df.columns:
            df['Date_Naissance'] = pd.to_datetime(df['Date_Naissance'], errors='coerce')

        # Supprimer les colonnes vides (détectées lors de l'exploration initiale)
        if 'Somme_Impaye' in df.columns:
            df = df.drop(columns=['Somme_Impaye', 'Nombre_Impaye'], errors='ignore')

        # Remplir les NaN pour les catégorielles
        categorical_cols = ['Tranche_Age', 'Civilite', 'Sexe', 'Statut_Civil', 'Segment', 'Situation_Contractuelle', 'Activite_Economique']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Inconnu')

        # Remplir les NaN pour les numériques (Salaire)
        for col in ['Salaire', 'Total montant cheque', 'Montant_cheque', 'Nombre']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Créer la métrique de paiement fractionné par opération
        df['is_split_record'] = (df['Nombre'] > 1).astype(int)
        return df

    df_clients = preprocess(df_clients)
    df_suppliers = preprocess(df_suppliers)

    # Agrégation des caractéristiques pour l'IA et le focus sur le fractionnement
    def aggregate_features(df, id_col, prefix):
        features = df.groupby(id_col).agg(
            total_montant=(f'Total montant cheque', 'sum'),
            moy_montant_cheque=(f'Montant_cheque', 'mean'),
            max_montant_cheque=(f'Montant_cheque', 'max'),
            min_montant_cheque=(f'Montant_cheque', 'min'),
            freq_transactions=(f'Nombre', 'sum'),
            nb_operations_distinctes=('Date_Operation', 'nunique'),
            recence=('Date_Operation', lambda x: (df['Date_Operation'].max() - x.max()).days),
            nb_paiements_fractionnes_records=('is_split_record', 'sum')
        ).reset_index()

        # Score de fragmentation : somme des 'Nombre' par opération distincte
        # Évite la division par zéro si nb_operations_distinctes est 0
        features['score_fragmentation'] = features.apply(
            lambda row: row['freq_transactions'] / row['nb_operations_distinctes'] if row['nb_operations_distinctes'] > 0 else 0,
            axis=1
        )

        # Renommer les colonnes avec le préfixe, en conservant l'ID original sans préfixe pour la fusion future
        features.columns = [id_col] + [f'{prefix}_{col}' for col in features.columns if col != id_col]
        return features

    client_features_extended = aggregate_features(df_clients, 'Compte_key_Payeur', 'client')
    supplier_features_extended = aggregate_features(df_suppliers, 'Compte_Key', 'supplier')

    # --- Calculer la PCA ici pour qu'elle soit toujours disponible pour l'affichage ---
    numerical_cols_clients_for_pca = [col for col in client_features_extended.columns if col.startswith('client_') and col not in ['client_Compte_key_Payeur', 'client_recence']] # recence can sometimes be tricky with PCA if not handled carefully
    X_clients_pca_data = client_features_extended[numerical_cols_clients_for_pca].fillna(client_features_extended[numerical_cols_clients_for_pca].median())
    scaler_clients_pca = StandardScaler()
    X_clients_scaled_pca = scaler_clients_pca.fit_transform(X_clients_pca_data)
    pca_clients = PCA(n_components=2)
    client_features_extended['pca1'] = pca_clients.fit_transform(X_clients_scaled_pca)[:, 0]
    client_features_extended['pca2'] = pca_clients.transform(X_clients_scaled_pca)[:, 1]

    numerical_cols_suppliers_for_pca = [col for col in supplier_features_extended.columns if col.startswith('supplier_') and col not in ['supplier_Compte_Key', 'supplier_recence']]
    X_suppliers_pca_data = supplier_features_extended[numerical_cols_suppliers_for_pca].fillna(supplier_features_extended[numerical_cols_suppliers_for_pca].median())
    scaler_suppliers_pca = StandardScaler()
    X_suppliers_scaled_pca = scaler_suppliers_pca.fit_transform(X_suppliers_pca_data)
    pca_suppliers = PCA(n_components=2)
    supplier_features_extended['pca1'] = pca_suppliers.fit_transform(X_suppliers_scaled_pca)[:, 0]
    supplier_features_extended['pca2'] = pca_suppliers.transform(X_suppliers_scaled_pca)[:, 1]

    return df_clients, df_suppliers, client_features_extended, supplier_features_extended

# Load data once and cache it
df_clients, df_suppliers, client_features_extended, supplier_features_extended = load_and_preprocess_data()

# --- Content Functions for Each Section ---

def show_data_overview():
    st.header("Aperçu des Données Brutes et Préparées")
    st.subheader("Données Clients (5 premières lignes)")
    st.dataframe(df_clients.head())
    st.subheader("Informations sur les Clients")
    buffer_client = io.StringIO()
    df_clients.info(buf=buffer_client, verbose=True, show_counts=True)
    st.text(buffer_client.getvalue())

    st.subheader("Données Fournisseurs (5 premières lignes)")
    st.dataframe(df_suppliers.head())
    st.subheader("Informations sur les Fournisseurs")
    buffer_supplier = io.StringIO()
    df_suppliers.info(buf=buffer_supplier, verbose=True, show_counts=True)
    st.text(buffer_supplier.getvalue())

def show_client_exploration():
    st.header("Exploration Détaillée des Données Clients")

    st.subheader("Distributions des Montants et Fréquences")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df_clients['Total montant cheque'], bins=50, kde=True, ax=ax1)
        ax1.set_title('Distribution du "Total montant cheque"')
        ax1.set_xlabel('Total montant cheque')
        ax1.set_ylabel('Nombre de transactions')
        ax1.set_xlim(0, df_clients['Total montant cheque'].quantile(0.99))
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.histplot(df_clients['Montant_cheque'], bins=50, kde=True, ax=ax2)
        ax2.set_title('Distribution du "Montant_cheque"')
        ax2.set_xlabel('Montant cheque')
        ax2.set_ylabel('Nombre de transactions')
        ax2.set_xlim(0, df_clients['Montant_cheque'].quantile(0.99))
        st.pyplot(fig2)
        plt.close(fig2)

    col3, col4 = st.columns(2)
    with col3:
        fig3, ax3 = plt.subplots()
        sns.histplot(df_clients['Nombre'], bins=30, kde=True, ax=ax3)
        ax3.set_title('Distribution du "Nombre" de chèques/transactions')
        ax3.set_xlabel('Nombre de chèques/transactions')
        ax3.set_ylabel('Fréquence')
        ax3.set_xlim(0, df_clients['Nombre'].quantile(0.99))
        st.pyplot(fig3)
        plt.close(fig3)

    st.subheader("Répartition par Catégories")
    selected_categorical_client = st.selectbox("Sélectionnez une colonne catégorielle (Clients)",
                                                    ['Segment', 'Tranche_Age', 'Sexe', 'Statut_Civil', 'Situation_Contractuelle'])
    fig_cat_client, ax_cat_client = plt.subplots(figsize=(10, 6))
    sns.countplot(y=selected_categorical_client, data=df_clients,
                    order=df_clients[selected_categorical_client].value_counts().index, ax=ax_cat_client, palette='viridis')
    ax_cat_client.set_title(f'Répartition des clients par "{selected_categorical_client}"')
    ax_cat_client.set_xlabel('Nombre de clients')
    ax_cat_client.set_ylabel(selected_categorical_client)
    st.pyplot(fig_cat_client)
    plt.close(fig_cat_client)

    st.subheader("Analyse Temporelle des Clients")
    df_clients['Annee_Mois'] = df_clients['Date_Operation'].dt.to_period('M')
    monthly_total_clients = df_clients.groupby('Annee_Mois')['Total montant cheque'].sum().reset_index()
    monthly_total_clients['Annee_Mois'] = monthly_total_clients['Annee_Mois'].astype(str)

    fig_time_client, ax_time_client = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Annee_Mois', y='Total montant cheque', data=monthly_total_clients, ax=ax_time_client)
    ax_time_client.set_title('Évolution mensuelle du "Total montant cheque" (Clients)')
    ax_time_client.set_xlabel('Mois')
    ax_time_client.set_ylabel('Total montant cheque')
    ax_time_client.tick_params(axis='x', rotation=45)
    st.pyplot(fig_time_client)
    plt.close(fig_time_client)

def show_supplier_exploration():
    st.header("Exploration Détaillée des Données Fournisseurs")

    st.subheader("Distributions des Montants et Fréquences")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df_suppliers['Total montant cheque'], bins=50, kde=True, ax=ax1)
        ax1.set_title('Distribution du "Total montant cheque"')
        ax1.set_xlabel('Total montant cheque')
        ax1.set_ylabel('Nombre de transactions')
        ax1.set_xlim(0, df_suppliers['Total montant cheque'].quantile(0.99))
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.histplot(df_suppliers['Montant_cheque'], bins=50, kde=True, ax=ax2)
        ax2.set_title('Distribution du "Montant_cheque"')
        ax2.set_xlabel('Montant cheque')
        ax2.set_ylabel('Nombre de transactions')
        ax2.set_xlim(0, df_suppliers['Montant_cheque'].quantile(0.99))
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Répartition par Catégories (Top 10)")
    selected_categorical_supplier = st.selectbox("Sélectionnez une colonne catégorielle (Fournisseurs)",
                                                    ['Activite_Economique', 'Segment', 'Sexe', 'Statut_Civil'])

    if selected_categorical_supplier == 'Activite_Economique':
        top_items = df_suppliers['Activite_Economique'].value_counts().head(10).index
        df_filtered_cat = df_suppliers[df_suppliers['Activite_Economique'].isin(top_items)]
    else:
        df_filtered_cat = df_suppliers

    fig_cat_supplier, ax_cat_supplier = plt.subplots(figsize=(10, 6))
    sns.countplot(y=selected_categorical_supplier, data=df_filtered_cat,
                    order=df_filtered_cat[selected_categorical_supplier].value_counts().index, ax=ax_cat_supplier, palette='rocket')
    ax_cat_supplier.set_title(f'Répartition des fournisseurs par "{selected_categorical_supplier}"')
    ax_cat_supplier.set_xlabel('Nombre de fournisseurs')
    ax_cat_supplier.set_ylabel(selected_categorical_supplier)
    st.pyplot(fig_cat_supplier)
    plt.close(fig_cat_supplier)

    st.subheader("Analyse Temporelle des Fournisseurs")
    df_suppliers['Annee_Mois'] = df_suppliers['Date_Operation'].dt.to_period('M')
    monthly_total_suppliers = df_suppliers.groupby('Annee_Mois')['Total montant cheque'].sum().reset_index()
    monthly_total_suppliers['Annee_Mois'] = monthly_total_suppliers['Annee_Mois'].astype(str)

    fig_time_supplier, ax_time_supplier = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Annee_Mois', y='Total montant cheque', data=monthly_total_suppliers, ax=ax_time_supplier, color='orange')
    ax_time_supplier.set_title('Évolution mensuelle du "Total montant cheque" (Fournisseurs)')
    ax_time_supplier.set_xlabel('Mois')
    ax_time_supplier.set_ylabel('Total montant cheque')
    ax_time_supplier.tick_params(axis='x', rotation=45)
    st.pyplot(fig_time_supplier)
    plt.close(fig_time_supplier)

def show_split_payments_analysis():
    st.header("Analyse Approfondie des Paiements Fractionnés")
    st.write("Le **score de fragmentation** est défini comme la somme du 'Nombre' de chèques par client/fournisseur divisée par le nombre d'opérations distinctes. Un score élevé indique un fractionnement plus important.")
    st.write("`nb_paiements_fractionnes_records` est le nombre d'enregistrements où 'Nombre' > 1 pour un client/fournisseur.")

    st.subheader("Distribution du Score de Fragmentation")
    col1, col2 = st.columns(2)
    with col1:
        fig_frag_client, ax_frag_client = plt.subplots()
        sns.histplot(client_features_extended['client_score_fragmentation'], bins=50, kde=True, ax=ax_frag_client)
        ax_frag_client.set_title('Distribution du Score de Fragmentation (Clients)')
        ax_frag_client.set_xlabel('Score de Fragmentation Client')
        st.pyplot(fig_frag_client)
        plt.close(fig_frag_client)
    with col2:
        fig_frag_supplier, ax_frag_supplier = plt.subplots()
        sns.histplot(supplier_features_extended['supplier_score_fragmentation'], bins=50, kde=True, ax=ax_frag_supplier)
        ax_frag_supplier.set_title('Distribution du Score de Fragmentation (Fournisseurs)')
        ax_frag_supplier.set_xlabel('Score de Fragmentation Fournisseur')
        st.pyplot(fig_frag_supplier)
        plt.close(fig_frag_supplier)

    st.subheader("Paiements Fractionnés par Segment/Activité")
    col3, col4 = st.columns(2)
    with col3:
        fig_frag_seg_client, ax_frag_seg_client = plt.subplots(figsize=(10, 6))
        df_clients_with_frag_segment = df_clients.merge(client_features_extended[['Compte_key_Payeur', 'client_score_fragmentation']], on='Compte_key_Payeur', how='left')
        sns.boxplot(x='client_score_fragmentation', y='Segment', data=df_clients_with_frag_segment,
                                 order=df_clients_with_frag_segment['Segment'].value_counts().index, ax=ax_frag_seg_client, palette='viridis', showfliers=False)
        ax_frag_seg_client.set_title('Score de Fragmentation par Segment Client')
        ax_frag_seg_client.set_xlabel('Score de Fragmentation')
        ax_frag_seg_client.set_ylabel('Segment')
        st.pyplot(fig_frag_seg_client)
        plt.close(fig_frag_seg_client)
    with col4:
        fig_frag_act_supplier, ax_frag_act_supplier = plt.subplots(figsize=(10, 6))
        df_suppliers_with_frag_activity = df_suppliers.merge(supplier_features_extended[['Compte_Key', 'supplier_score_fragmentation']], on='Compte_Key', how='left')
        top_activities = df_suppliers_with_frag_activity['Activite_Economique'].value_counts().head(10).index
        sns.boxplot(x='supplier_score_fragmentation', y='Activite_Economique',
                                 data=df_suppliers_with_frag_activity[df_suppliers_with_frag_activity['Activite_Economique'].isin(top_activities)],
                                 order=top_activities, ax=ax_frag_act_supplier, palette='rocket', showfliers=False)
        ax_frag_act_supplier.set_title('Score de Fragmentation par Activité Économique (Top 10 Fournisseurs)')
        ax_frag_act_supplier.set_xlabel('Score de Fragmentation')
        ax_frag_act_supplier.set_ylabel('Activité Économique')
        st.pyplot(fig_frag_act_supplier)
        plt.close(fig_frag_act_supplier)

    st.subheader("Définition du Seuil de Risque de Fractionnement")
    st.markdown("Basé sur la distribution et les anomalies (voir section IA), nous pouvons définir un seuil. Par exemple, le 95ème percentile.")

    q_client = client_features_extended['client_score_fragmentation'].quantile(0.95)
    q_supplier = supplier_features_extended['supplier_score_fragmentation'].quantile(0.95)
    st.write(f"Seuil 95ème percentile pour le Score de Fragmentation Clients : **{q_client:.2f}**")
    st.write(f"Seuil 95ème percentile pour le Score de Fragmentation Fournisseurs : **{q_supplier:.2f}**")

    st.markdown("Les clients/fournisseurs avec un score de fragmentation supérieur à ce seuil pourraient être considérés à risque ou méritant une attention particulière.")

def show_anomaly_detection():
    st.header("Détection d'Anomalies avec Isolation Forest")
    st.write("L'Isolation Forest identifie les points de données qui sont 'anormaux' par rapport à la majorité. Les anomalies sont marquées en rouge.")

    # Paramètres d'anomalie
    contamination_clients = st.sidebar.slider("Contamination (Clients - % d'anomalies attendues)", 0.01, 0.10, 0.05, 0.01)
    contamination_suppliers = st.sidebar.slider("Contamination (Fournisseurs - % d'anomalies attendues)", 0.01, 0.10, 0.05, 0.01)

    # Préparation des données pour l'Isolation Forest
    numerical_cols_clients_for_anomaly = [col for col in client_features_extended.columns if col.startswith('client_') and col not in ['client_Compte_key_Payeur', 'client_recence']]
    X_clients_scaled = StandardScaler().fit_transform(client_features_extended[numerical_cols_clients_for_anomaly].fillna(client_features_extended[numerical_cols_clients_for_anomaly].median()))

    numerical_cols_suppliers_for_anomaly = [col for col in supplier_features_extended.columns if col.startswith('supplier_') and col not in ['supplier_Compte_Key', 'supplier_recence']]
    X_suppliers_scaled = StandardScaler().fit_transform(supplier_features_extended[numerical_cols_suppliers_for_anomaly].fillna(supplier_features_extended[numerical_cols_suppliers_for_anomaly].median()))

    # Modèles Isolation Forest
    model_clients = IsolationForest(random_state=42, contamination=contamination_clients)
    client_features_extended['is_anomaly'] = model_clients.fit_predict(X_clients_scaled)

    model_suppliers = IsolationForest(random_state=42, contamination=contamination_suppliers)
    supplier_features_extended['is_anomaly'] = model_suppliers.fit_predict(X_suppliers_scaled)

    st.subheader("Anomalies Clients")
    st.write(f"Nombre de clients identifiés comme anomalies : {sum(client_features_extended['is_anomaly'] == -1)}")
    st.dataframe(client_features_extended[client_features_extended['is_anomaly'] == -1].head())

    # Utilisation des PCA déjà calculées
    fig_anom_client, ax_anom_client = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='pca1', y='pca2', hue='is_anomaly', data=client_features_extended,
                    palette={1: 'blue', -1: 'red'}, alpha=0.7, ax=ax_anom_client)
    ax_anom_client.set_title('Détection d\'Anomalies chez les Clients (PCA 2D)')
    ax_anom_client.legend(title='Est Anomalie', labels=['Normal', 'Anomalie'])
    st.pyplot(fig_anom_client)
    plt.close(fig_anom_client)

    st.subheader("Anomalies Fournisseurs")
    st.write(f"Nombre de fournisseurs identifiés comme anomalies : {sum(supplier_features_extended['is_anomaly'] == -1)}")
    st.dataframe(supplier_features_extended[supplier_features_extended['is_anomaly'] == -1].head())

    # Utilisation des PCA déjà calculées
    fig_anom_supplier, ax_anom_supplier = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='pca1', y='pca2', hue='is_anomaly', data=supplier_features_extended,
                    palette={1: 'blue', -1: 'red'}, alpha=0.7, ax=ax_anom_supplier)
    ax_anom_supplier.set_title('Détection d\'Anomalies chez les Fournisseurs (PCA 2D)')
    ax_anom_supplier.legend(title='Est Anomalie', labels=['Normal', 'Anomalie'])
    st.pyplot(fig_anom_supplier)
    plt.close(fig_anom_supplier)

    st.markdown("**Lien avec les paiements fractionnés :** Les anomalies peuvent inclure des clients/fournisseurs avec des scores de fragmentation inhabituellement élevés (ou bas), indiquant des comportements de paiement à examiner.")
    st.subheader("Anomalies Clients vs Fragmentation")
    fig_anom_frag_client, ax_anom_frag_client = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='client_total_montant', y='client_score_fragmentation', hue='is_anomaly', size='client_nb_paiements_fractionnes_records',
                    sizes=(20, 400), data=client_features_extended, palette={1: 'blue', -1: 'red'}, alpha=0.7, ax=ax_anom_frag_client)
    ax_anom_frag_client.set_title('Clients : Total Montant vs Score de Fragmentation (Anomalies en Rouge)')
    ax_anom_frag_client.set_xlabel('Total Montant Cheque Client')
    ax_anom_frag_client.set_ylabel('Score de Fragmentation')
    ax_anom_frag_client.set_xscale('log')
    ax_anom_frag_client.set_yscale('log')
    ax_anom_frag_client.legend(title='Est Anomalie', labels=['Normal', 'Anomalie'], bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_anom_frag_client)
    plt.close(fig_anom_frag_client)

def show_clustering_analysis():
    st.header("Clustering et Segmentation avec K-Means")
    st.write("Le clustering regroupe les clients/fournisseurs en segments basés sur leurs caractéristiques de paiement.")

    # Slider pour choisir le nombre de clusters
    n_clusters_clients = st.sidebar.slider("Nombre de Clusters (Clients)", 2, 8, 3)
    n_clusters_suppliers = st.sidebar.slider("Nombre de Clusters (Fournisseurs)", 2, 8, 3)

    numerical_cols_clients_for_clustering = [col for col in client_features_extended.columns if col.startswith('client_') and col not in ['client_Compte_key_Payeur', 'client_recence']]
    X_clients_scaled_clustering = StandardScaler().fit_transform(client_features_extended[numerical_cols_clients_for_clustering].fillna(client_features_extended[numerical_cols_clients_for_clustering].median()))

    numerical_cols_suppliers_for_clustering = [col for col in supplier_features_extended.columns if col.startswith('supplier_') and col not in ['supplier_Compte_Key', 'supplier_recence']]
    X_suppliers_scaled_clustering = StandardScaler().fit_transform(supplier_features_extended[numerical_cols_suppliers_for_clustering].fillna(supplier_features_extended[numerical_cols_suppliers_for_clustering].median()))

    # Clustering Clients
    kmeans_clients = KMeans(n_clusters=n_clusters_clients, random_state=42, n_init=10)
    client_features_extended['cluster'] = kmeans_clients.fit_predict(X_clients_scaled_clustering)

    st.subheader("Clusters Clients")
    fig_cluster_client, ax_cluster_client = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=client_features_extended, palette='viridis', alpha=0.8, ax=ax_cluster_client)
    ax_cluster_client.set_title(f'Clusters de Clients (K-Means sur PCA 2D, k={n_clusters_clients})')
    ax_cluster_client.legend(title='Cluster')
    st.pyplot(fig_cluster_client)
    plt.close(fig_cluster_client)

    st.write("Analyse des caractéristiques moyennes par cluster client (standardisées puis inversées pour lisibilité):")
    # Use the scaler fitted specifically for clustering data
    scaler_for_inverse_transform_clients = StandardScaler().fit(client_features_extended[numerical_cols_clients_for_clustering].fillna(client_features_extended[numerical_cols_clients_for_clustering].median()))
    cluster_centers_clients = pd.DataFrame(scaler_for_inverse_transform_clients.inverse_transform(kmeans_clients.cluster_centers_), columns=numerical_cols_clients_for_clustering)
    st.dataframe(cluster_centers_clients)
    st.markdown("**Interprétation :** Examinez les valeurs de `client_score_fragmentation` dans chaque cluster pour identifier les segments avec des comportements de fractionnement élevés.")

    # Clustering Fournisseurs
    kmeans_suppliers = KMeans(n_clusters=n_clusters_suppliers, random_state=42, n_init=10)
    supplier_features_extended['cluster'] = kmeans_suppliers.fit_predict(X_suppliers_scaled_clustering)

    st.subheader("Clusters Fournisseurs")
    fig_cluster_supplier, ax_cluster_supplier = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=supplier_features_extended, palette='viridis', alpha=0.8, ax=ax_cluster_supplier)
    ax_cluster_supplier.set_title(f'Clusters de Fournisseurs (K-Means sur PCA 2D, k={n_clusters_suppliers})')
    ax_cluster_supplier.legend(title='Cluster')
    st.pyplot(fig_cluster_supplier)
    plt.close(fig_cluster_supplier)

    st.write("Analyse des caractéristiques moyennes par cluster fournisseur (standardisées puis inversées pour lisibilité):")
    scaler_for_inverse_transform_suppliers = StandardScaler().fit(supplier_features_extended[numerical_cols_suppliers_for_clustering].fillna(supplier_features_extended[numerical_cols_suppliers_for_clustering].median()))
    cluster_centers_suppliers = pd.DataFrame(scaler_for_inverse_transform_suppliers.inverse_transform(kmeans_suppliers.cluster_centers_), columns=numerical_cols_suppliers_for_clustering)
    st.dataframe(cluster_centers_suppliers)
    st.markdown("**Interprétation :** Les clusters révèlent des segments de fournisseurs avec des comportements de transaction similaires, y compris le fractionnement.")

def show_client_supplier_relation():
    st.header("Analyse de la Relation entre Clients et Fournisseurs")

    clients_payers_ids = df_clients['Compte_key_Payeur'].unique()
    suppliers_payers_ids = df_suppliers['Compte_key_Payeur'].unique()

    common_payers_ids = np.intersect1d(clients_payers_ids, suppliers_payers_ids)

    st.write(f"Nombre de Compte_key_Payeur uniques chez les clients: {len(clients_payers_ids)}")
    st.write(f"Nombre de Compte_key_Payeur uniques chez les fournisseurs (en tant que payeurs): {len(suppliers_payers_ids)}")
    st.write(f"Nombre de Compte_key_Payeur communs aux deux fichiers (entités agissant comme payeurs dans les deux contextes): {len(common_payers_ids)}")

    if len(common_payers_ids) > 0:
        st.markdown("Ces entités agissent à la fois comme 'clients' (payeurs dans le fichier clients) et comme 'fournisseurs' (payeurs dans le fichier fournisseurs).")

        common_client_features = client_features_extended[client_features_extended['Compte_key_Payeur'].isin(common_payers_ids)].copy()

        # Agrégation spécifique pour les fournisseurs lorsqu'ils agissent comme payeurs (Compte_key_Payeur)
        supplier_payer_features = df_suppliers.groupby('Compte_key_Payeur').agg(
            total_montant_fournisseur_payeur=('Total montant cheque', 'sum'),
            moy_montant_cheque_fournisseur_payeur=('Montant_cheque', 'mean'),
            nb_operations_distinctes_fournisseur_payeur=('Date_Operation', 'nunique'),
            freq_transactions_fournisseur_payeur=('Nombre', 'sum'),
            nb_paiements_fractionnes_records_fournisseur_payeur=('is_split_record', 'sum')
        ).reset_index()

        supplier_payer_features['score_fragmentation_fournisseur_payeur'] = \
            supplier_payer_features.apply(
                lambda row: row['freq_transactions_fournisseur_payeur'] / row['nb_operations_distinctes_fournisseur_payeur'] if row['nb_operations_distinctes_fournisseur_payeur'] > 0 else 0,
                axis=1
            )

        merged_common_entities = pd.merge(
            common_client_features,
            supplier_payer_features,
            left_on='Compte_key_Payeur',
            right_on='Compte_key_Payeur',
            how='inner'
        )
        st.subheader("Caractéristiques des Entités Communes")
        st.dataframe(merged_common_entities.head())
        st.write(f"Nombre d'entités communes dans le DataFrame fusionné: {len(merged_common_entities)}")

        st.subheader("Matrice de Corrélation des Comportements de Paiement")
        correlation_cols = [
            'client_total_montant', 'client_moy_montant_cheque', 'client_score_fragmentation',
            'total_montant_fournisseur_payeur', 'moy_montant_cheque_fournisseur_payeur',
            'score_fragmentation_fournisseur_payeur'
        ]
        df_corr = merged_common_entities[correlation_cols].copy()
        df_corr = df_corr.fillna(df_corr.median())

        if not df_corr.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            ax_corr.set_title('Matrice de Corrélation des Comportements de Paiement (Entités Communes)')
            st.pyplot(fig_corr)
            plt.close(fig_corr)
            st.markdown("Interprétation : Des valeurs proches de 1 ou -1 indiquent une forte corrélation positive ou négative. Une corrélation élevée entre les scores de fragmentation côté client et fournisseur pourrait indiquer des comportements de gestion de trésorerie similaires pour ces entités.")
        else:
            st.warning("Pas assez de données dans les entités communes pour calculer la matrice de corrélation.")

        st.subheader("Graphiques de Dispersion pour les Entités Communes")
        col_scatter1, col_scatter2 = st.columns(2)
        with col_scatter1:
            if 'client_total_montant' in merged_common_entities.columns and 'total_montant_fournisseur_payeur' in merged_common_entities.columns:
                fig_scatter1, ax_scatter1 = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x='client_total_montant', y='total_montant_fournisseur_payeur', data=merged_common_entities, alpha=0.7, ax=ax_scatter1)
                ax_scatter1.set_title('Total Montant Client vs Total Montant Fournisseur (Payeur)')
                ax_scatter1.set_xlabel('Total Montant Cheque Client')
                ax_scatter1.set_ylabel('Total Montant Cheque Fournisseur (Payeur)')
                ax_scatter1.set_xscale('log')
                ax_scatter1.set_yscale('log')
                st.pyplot(fig_scatter1)
                plt.close(fig_scatter1)
        with col_scatter2:
            if 'client_score_fragmentation' in merged_common_entities.columns and 'score_fragmentation_fournisseur_payeur' in merged_common_entities.columns:
                fig_scatter2, ax_scatter2 = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x='client_score_fragmentation', y='score_fragmentation_fournisseur_payeur', data=merged_common_entities, alpha=0.7, ax=ax_scatter2)
                ax_scatter2.set_title('Score de Fragmentation Client vs Fournisseur (Payeur) pour Entités Communes')
                ax_scatter2.set_xlabel('Score de Fragmentation Client')
                ax_scatter2.set_ylabel('Score de Fragmentation Fournisseur (en tant que Payeur)')
                ax_scatter2.set_xscale('log')
                ax_scatter2.set_yscale('log')
                st.pyplot(fig_scatter2)
                plt.close(fig_scatter2)
    else:
        st.info("Aucune entité commune identifiée avec le même 'Compte_key_Payeur' dans les deux fichiers en tant que payeur. La relation directe via cet identifiant n'est pas significative. D'autres approches sont nécessaires.")
        st.markdown("""
        **Autres pistes pour les relations :**
        * **Flux Client -> Fournisseur :** Si le `Rib_beneficiaire_encoded` (client) peut être mappé au `rib_encoded` (fournisseur) ou si `code_postal_beneficiaire` (client) peut être mappé à `code_postal` (fournisseur), pour comprendre où les clients paient leurs fournisseurs.
        * **Correspondance Activité Économique vs Segment Client :** Analyser quels segments de clients interagissent le plus avec quels types d'activités économiques de fournisseurs (agrégations et jointures sur ville/code postal).
        """)

# --- LLM Chat Function ---
def show_llm_chat():
    st.header("Discutez avec vos données (LLM)")
    st.write("Posez des questions sur les datasets clients et fournisseurs. Le modèle générera des résumés, des observations ou des analyses.")

    # Prepare data context for the LLM
    client_summary = df_clients.describe(include='all').to_markdown()
    supplier_summary = df_suppliers.describe(include='all').to_markdown()

    client_info = io.StringIO()
    df_clients.info(buf=client_info, verbose=True, show_counts=True)
    client_info_str = client_info.getvalue()

    supplier_info = io.StringIO()
    df_suppliers.info(buf=supplier_info, verbose=True, show_counts=True)
    supplier_info_str = supplier_info.getvalue()

    # Create a system prompt to guide the LLM
    system_prompt = f"""
    You are an expert data analyst assistant. You are analyzing two datasets: 'Client' and 'Fournisseur' (Supplier).
    The 'Client' dataset contains payment transaction information for clients. Key columns include:
    - 'Compte_key_Payeur': Unique client identifier.
    - 'Date_Operation': Date of the transaction.
    - 'Total montant cheque': Total amount for the operation.
    - 'Montant_cheque': Amount of a single cheque/transaction within the operation.
    - 'Nombre': Number of cheques/transactions for that operation (relevant for split payments).
    - 'Tranche_Age', 'Civilite', 'Sexe', 'Statut_Civil', 'Segment', 'Situation_Contractuelle': Client demographic and contractual information.
    - 'is_split_record': Binary flag (1 if 'Nombre' > 1, 0 otherwise).

    Here is a summary of the client data:
    {client_summary}

    And its structure:
    {client_info_str}

    The 'Fournisseur' (Supplier) dataset contains payment transaction information for suppliers. Key columns include:
    - 'Compte_Key': Unique supplier identifier.
    - 'Date_Operation': Date of the transaction.
    - 'Total montant cheque': Total amount for the operation.
    - 'Montant_cheque': Amount of a single cheque/transaction within the operation.
    - 'Nombre': Number of cheques/transactions for that operation (relevant for split payments).
    - 'Activite_Economique', 'Segment', 'Sexe', 'Statut_Civil': Supplier business/demographic information.
    - 'is_split_record': Binary flag (1 if 'Nombre' > 1, 0 otherwise).

    Here is a summary of the supplier data:
    {supplier_summary}

    And its structure:
    {supplier_info_str}

    Your goal is to answer questions about these datasets. If a question requires precise numerical calculations that are not immediately available in the summaries, state that you can provide insights based on the available data, or that a precise calculation would require more in-depth statistical analysis or direct querying of the raw data.
    Focus on providing clear, concise, and insightful answers.
    """

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    full_prompt = f"{system_prompt}\n\nUser Question: {prompt}"
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})

# --- Function to download questions ---
def show_download_questions():
    st.header("Télécharger des exemples de questions")
    st.write("Utilisez ces questions pour interroger le modèle LLM sur vos données.")

    questions_content = """
### General Questions About the Data Structure and Content:
- Tell me about the client dataset. What kind of information does it contain?
- What are the main differences between the client and supplier datasets?
- Can you list the columns available in the supplier data and explain what each one means?
- Are there any missing values in the 'Salaire' column for clients? If so, how are they handled?

### Questions Focusing on Quantitative Insights:
- What is the average 'Total montant cheque' for clients?
- What is the maximum 'Montant_cheque' observed in the supplier data?
- How many distinct 'Segment' categories are there for clients?
- Can you give me an idea of the distribution of 'Nombre' (number of checks/transactions) in the client dataset?
- What's the spread of 'score_fragmentation' for suppliers?

### Questions About Trends and Categorical Breakdowns:
- Which 'Segment' has the highest number of clients?
- What are the top 5 'Activite_Economique' categories among suppliers?
- Are there more male or female clients in the dataset?
- What can you tell me about the distribution of clients across different 'Tranche_Age' groups?

### Questions Specific to "Paiements Fractionnés" (Split Payments):
- Explain what 'score_fragmentation' represents and why it's important.
- Which client segment appears to have a higher average 'score_fragmentation'?
- Are there any specific supplier activities that tend to have a high fragmentation score?
- What is the threshold for a high fragmentation score for clients, according to the 95th percentile?

### More Complex / Interpretive Questions (Leveraging the prompt context):
- Based on the data summaries, what are some key insights about client payment behavior?
- Are there any potential anomalies visible in the supplier data based on the summary statistics?
- If I'm looking for clients who frequently split their payments, what columns should I pay close attention to?
- What business questions can be answered using the 'is_split_record' column?
- How can I identify 'high-value' clients or suppliers using the available data?

### Questions to Test Limitations/Refusals:
- What is the precise 'Compte_key_Payeur' of the client with the highest 'score_fragmentation'?
- Generate a graph showing the monthly 'Total montant cheque' for clients.
- Give me a detailed list of all client transactions.
"""
    st.download_button(
        label="Télécharger les questions (TXT)",
        data=questions_content,
        file_name="llm_test_questions.txt",
        mime="text/plain"
    )
    st.info("Cliquez sur le bouton pour télécharger un fichier texte contenant des exemples de questions à poser à l'LLM.")


# --- Main Application Logic (Simplified for no authentication) ---
st.title("📊 Analyse de Clients et Fournisseurs : Chèques et Paiements Fractionnés")
st.markdown("Ce tableau de bord interactif explore les données clients et fournisseurs pour identifier des patterns, des anomalies et des opportunités de ciblage, avec un focus particulier sur les paiements fractionnés.")

st.sidebar.header("Navigation")
selected_section = st.sidebar.radio(
    "Aller à",
    ["Aperçu des Données", "Exploration des Clients", "Exploration des Fournisseurs",
     "Paiements Fractionnés Approfondi", "Détection d'Anomalies (IA)", "Clustering (IA)",
     "Relation Clients-Fournisseurs", "Discuter avec les Données (LLM)", "Télécharger Questions LLM"]
)

if selected_section == "Aperçu des Données":
    show_data_overview()
elif selected_section == "Exploration des Clients":
    show_client_exploration()
elif selected_section == "Exploration des Fournisseurs":
    show_supplier_exploration()
elif selected_section == "Paiements Fractionnés Approfondi":
    show_split_payments_analysis()
elif selected_section == "Détection d'Anomalies (IA)":
    show_anomaly_detection()
elif selected_section == "Clustering (IA)":
    show_clustering_analysis()
elif selected_section == "Relation Clients-Fournisseurs":
    show_client_supplier_relation()
elif selected_section == "Discuter avec les Données (LLM)":
    show_llm_chat()
elif selected_section == "Télécharger Questions LLM":
    show_download_questions()