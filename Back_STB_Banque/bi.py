import pandas as pd
from flask import Flask, jsonify, request, send_file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from flask_cors import CORS

# --- Data Loading and Preprocessing ---
# Global variables for aggregated features (will be populated by load_and_preprocess_data)
client_features_extended_global = None
supplier_features_extended_global = None

def load_and_preprocess_data():
    """
    Loads client and supplier data from CSV files and performs initial preprocessing.
    Handles missing dates (NaT) by converting them to None for JSON serialization.
    Fills NaN values in categorical columns with 'Inconnu' and numerical columns with their median.
    Also aggregates features for split payment analysis.
    Raises a RuntimeError if a CSV file is not found.
    """
    df_clients_raw = None
    df_suppliers_raw = None

    # Load client data
    try:
        df_clients_raw = pd.read_csv('Client - Sheet1.csv')
        df_clients_raw['Source'] = 'Client'
    except FileNotFoundError:
        print("Error: 'Client - Sheet1.csv' not found. Ensure it's in the same directory as the API.")
        raise RuntimeError("Client file not found")

    # Load supplier data
    try:
        df_suppliers_raw = pd.read_csv('Fournisseur - Sheet1.csv')
        df_suppliers_raw['Source'] = 'Fournisseur'
    except FileNotFoundError:
        print("Error: 'Fournisseur - Sheet1.csv' not found. Ensure it's in the same directory as the API.")
        raise RuntimeError("Supplier file not found")

    # Common preprocessing function for DataFrames
    def preprocess_df_base(df):
        # Convert dates, coercing errors to NaT
        # Store original datetime objects in '_dt_object' columns for aggregation
        df['Date_Operation_dt_object'] = pd.to_datetime(df['Date_Operation'], errors='coerce')
        df['Date_Operation'] = df['Date_Operation_dt_object'].dt.strftime('%Y-%m-%d').replace({np.nan: None})

        if 'Date_Naissance' in df.columns:
            df['Date_Naissance_dt_object'] = pd.to_datetime(df['Date_Naissance'], errors='coerce')
            df['Date_Naissance'] = df['Date_Naissance_dt_object'].dt.strftime('%Y-%m-%d').replace({np.nan: None})
        else:
             df['Date_Naissance_dt_object'] = pd.NaT # Ensure column exists if not present

        # Fill NaN values for categorical columns with 'Inconnu'
        categorical_cols = ['Tranche_Age', 'Civilite', 'Sexe', 'Statut_Civil', 'Segment', 'Situation_Contractuelle', 'Activite_Economique']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Inconnu')

        # Drop 'Somme_Impaye' and 'Nombre_Impaye' if they exist (as per original Streamlit logic)
        if 'Somme_Impaye' in df.columns:
            df = df.drop(columns=['Somme_Impaye', 'Nombre_Impaye'], errors='ignore')

        # Fill NaN values for numerical columns with their median
        for col in ['Salaire', 'Total montant cheque', 'Montant_cheque', 'Nombre', 'Montant']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Create the is_split_record metric
        df['is_split_record'] = (df['Nombre'] > 1).astype(int)

        return df

    df_clients_processed = preprocess_df_base(df_clients_raw)
    df_suppliers_processed = preprocess_df_base(df_suppliers_raw)

    # --- Feature Aggregation for AI and Split Payment Focus ---
    def aggregate_features(df, id_col, prefix):
        # Ensure 'Date_Operation_dt_object' is not NaN before using it for recence calculation
        df_valid_dates = df.dropna(subset=['Date_Operation_dt_object'])
        max_date_overall = df_valid_dates['Date_Operation_dt_object'].max()

        features = df.groupby(id_col).agg(
            total_montant=(f'Total montant cheque', 'sum'),
            moy_montant_cheque=(f'Montant_cheque', 'mean'),
            max_montant_cheque=(f'Montant_cheque', 'max'),
            min_montant_cheque=(f'Montant_cheque', 'min'),
            freq_transactions=(f'Nombre', 'sum'),
            nb_operations_distinctes=('Date_Operation_dt_object', 'nunique'), # Use dt_object for nunique
            recence=('Date_Operation_dt_object', lambda x: (max_date_overall - x.max()).days if pd.notna(x.max()) else np.nan),
            nb_paiements_fractionnes_records=('is_split_record', 'sum')
        ).reset_index()

        # Score de fragmentation : somme des 'Nombre' par opération distincte
        features['score_fragmentation'] = features.apply(
            lambda row: row['freq_transactions'] / row['nb_operations_distinctes'] if row['nb_operations_distinctes'] > 0 else 0,
            axis=1
        )
        # Handle potential division by zero for score_fragmentation if nb_operations_distinctes is 0
        features['score_fragmentation'] = features['score_fragmentation'].replace([np.inf, -np.inf], np.nan).fillna(0)


        # Renommer les colonnes avec le préfixe
        features.columns = [id_col] + [f'{prefix}_{col}' for col in features.columns if col != id_col]
        return features

    # Aggregate client features
    client_features_agg = aggregate_features(df_clients_processed, 'Compte_key_Payeur', 'client')
    # Merge back some original categorical columns if needed for box plots
    # We need Segment from df_clients_processed for box plots, so merge it.
    client_features_extended = pd.merge(client_features_agg,
                                        df_clients_processed[['Compte_key_Payeur', 'Segment']].drop_duplicates(),
                                        on='Compte_key_Payeur', how='left')


    # Aggregate supplier features
    supplier_features_agg = aggregate_features(df_suppliers_processed, 'Compte_Key', 'supplier')
    # Merge back 'Activite_Economique' for box plots
    supplier_features_extended = pd.merge(supplier_features_agg,
                                          df_suppliers_processed[['Compte_Key', 'Activite_Economique']].drop_duplicates(),
                                          on='Compte_Key', how='left')


    return df_clients_processed, df_suppliers_processed, client_features_extended, supplier_features_extended

# --- Global Data Loading (executed once on app startup) ---
try:
    df_clients_global, df_suppliers_global, client_features_extended_global, supplier_features_extended_global = load_and_preprocess_data()
    print("Client and supplier data (raw and aggregated) loaded successfully.")
except RuntimeError as e:
    print(f"Critical data loading failure on startup: {e}")
    exit(1)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) 

# --- API Endpoint: Data Overview ---
# --- API Endpoint: Data Overview ---
@app.route("/data-overview", methods=['GET'])
def get_data_overview():
    """
    Returns an overview of raw client and supplier data.
    Includes the first 5 rows of each DataFrame, their shape, and column names.
    """
    if df_clients_global is None or df_suppliers_global is None:
        return jsonify({"error": "Data could not be loaded by the API on startup."}), 500

    # Explicitly convert NaT to None before converting to dictionary for JSON serialization
    clients_head = df_clients_global.head().replace({pd.NaT: None}).to_dict(orient='records')
    suppliers_head = df_suppliers_global.head().replace({pd.NaT: None}).to_dict(orient='records')

    response_data = {
        "clients": {
            "description": "First 5 rows of client data",
            "shape": list(df_clients_global.shape),
            "columns": df_clients_global.columns.tolist(),
            "first_5_rows": clients_head
        },
        "fournisseurs": {
            "description": "First 5 rows of supplier data",
            "shape": list(df_suppliers_global.shape),
            "columns": df_suppliers_global.columns.tolist(),
            "first_5_rows": suppliers_head
        }
    }
    return jsonify(response_data)
# @app.route("/data-overview", methods=['GET'])
# def get_data_overview():
#     """
#     Returns an overview of raw client and supplier data.
#     Includes the first 5 rows of each DataFrame, their shape, and column names.
#     """
#     if df_clients_global is None or df_suppliers_global is None:
#         return jsonify({"error": "Data could not be loaded by the API on startup."}), 500

#     clients_head = df_clients_global.head().to_dict(orient='records')
#     suppliers_head = df_suppliers_global.head().to_dict(orient='records')

#     response_data = {
#         "clients": {
#             "description": "First 5 rows of client data",
#             "shape": list(df_clients_global.shape),
#             "columns": df_clients_global.columns.tolist(),
#             "first_5_rows": clients_head
#         },
#         "fournisseurs": {
#             "description": "First 5 rows of supplier data",
#             "shape": list(df_suppliers_global.shape),
#             "columns": df_suppliers_global.columns.tolist(),
#             "first_5_rows": suppliers_head
#         }
#     }
#     return jsonify(response_data)

# --- Client Exploration - Image Endpoints ---

@app.route("/clients/exploration/total-montant-cheque-distribution-image", methods=['GET'])
def get_total_montant_cheque_distribution_image():
    """Generates and returns a PNG image of the histogram for 'Total montant cheque' for clients."""
    if df_clients_global is None: return jsonify({"error": "Client data not available."}), 500
    column = 'Total montant cheque'
    try:
        upper_bound = df_clients_global[column].quantile(0.99)
        filtered_data = df_clients_global[df_clients_global[column] <= upper_bound][column].dropna()
        if filtered_data.empty: return jsonify({"error": f"No valid data for '{column}' to generate histogram."}), 404
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_data, bins=50, kde=True, ax=ax)
        ax.set_title(f'Distribution du "{column}" (Clients)')
        ax.set_xlabel(column)
        ax.set_ylabel('Nombre de transactions (Fréquence)')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error generating image for '{column}': {str(e)}"}), 500


@app.route("/clients/exploration/montant-cheque-distribution-image", methods=['GET'])
def get_montant_cheque_distribution_image():
    """Generates and returns a PNG image of the histogram for 'Montant_cheque' for clients."""
    if df_clients_global is None: return jsonify({"error": "Client data not available."}), 500
    column = 'Montant_cheque'
    try:
        upper_bound = df_clients_global[column].quantile(0.99)
        filtered_data = df_clients_global[df_clients_global[column] <= upper_bound][column].dropna()
        if filtered_data.empty: return jsonify({"error": f"No valid data for '{column}' to generate histogram."}), 404
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_data, bins=50, kde=True, ax=ax)
        ax.set_title(f'Distribution du "{column}" (Clients)')
        ax.set_xlabel(column)
        ax.set_ylabel('Nombre de transactions (Fréquence)')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error generating image for '{column}': {str(e)}"}), 500


@app.route("/clients/exploration/nombre-distribution-image", methods=['GET'])
def get_nombre_distribution_image():
    """Generates and returns a PNG image of the histogram for 'Nombre' for clients."""
    if df_clients_global is None: return jsonify({"error": "Client data not available."}), 500
    column = 'Nombre'
    try:
        upper_bound = df_clients_global[column].quantile(0.99)
        filtered_data = df_clients_global[df_clients_global[column] <= upper_bound][column].dropna()
        if filtered_data.empty: return jsonify({"error": f"No valid data for '{column}' to generate histogram."}), 404
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_data, bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution du "{column}" de chèques/transactions (Clients)')
        ax.set_xlabel('Nombre de chèques/transactions')
        ax.set_ylabel('Fréquence')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error generating image for '{column}': {str(e)}"}), 500


@app.route("/clients/exploration/categorical-distribution-image", methods=['GET'])
def get_client_categorical_distribution_image():
    """Generates and returns a PNG image of a bar chart for a specific client categorical column."""
    if df_clients_global is None: return jsonify({"error": "Client data not available."}), 500
    column = request.args.get('column')
    if not column: return jsonify({"error": "Parameter 'column' is missing from the request."}), 400
    categorical_cols_available = ['Segment', 'Tranche_Age', 'Sexe', 'Statut_Civil', 'Situation_Contractuelle', 'Civilite']
    if column not in categorical_cols_available:
        return jsonify({"error": f"Column '{column}' is not valid or available for client categorical analysis. Possible columns: {', '.join(categorical_cols_available)}"}), 400
    if column not in df_clients_global.columns:
        return jsonify({"error": f"Column '{column}' does not exist in the client DataFrame."}), 404
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=column, data=df_clients_global,
                      order=df_clients_global[column].value_counts().index,
                      ax=ax, palette='viridis')
        ax.set_title(f'Répartition des clients par "{column}"')
        ax.set_xlabel('Nombre de clients')
        ax.set_ylabel(column)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error generating image for categorical distribution of '{column}': {str(e)}"}), 500


@app.route("/clients/exploration/time-series-image", methods=['GET'])
def get_client_time_series_image():
    """Generates and returns a PNG image of the line chart for monthly 'Total montant cheque' for clients."""
    if df_clients_global is None: return jsonify({"error": "Client data not available."}), 500
    try:
        temp_df = df_clients_global.copy()
        temp_df['Date_Operation_dt'] = pd.to_datetime(temp_df['Date_Operation_dt_object'], errors='coerce')
        temp_df = temp_df.dropna(subset=['Date_Operation_dt'])
        if temp_df.empty: return jsonify({"error": "No valid date data for client time series analysis."}), 404

        temp_df['Annee_Mois'] = temp_df['Date_Operation_dt'].dt.to_period('M')
        monthly_total_clients = temp_df.groupby('Annee_Mois')['Total montant cheque'].sum().reset_index()
        monthly_total_clients['Annee_Mois_str'] = monthly_total_clients['Annee_Mois'].astype(str)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='Annee_Mois_str', y='Total montant cheque', data=monthly_total_clients, ax=ax, color='blue')
        ax.set_title('Évolution mensuelle du "Total montant cheque" (Clients)')
        ax.set_xlabel('Mois')
        ax.set_ylabel('Total montant cheque')
        ax.tick_params(axis='x', rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error generating client time series image: {str(e)}"}), 500

# --- Supplier Exploration - Image Endpoints (EXISTING) ---

@app.route("/fournisseurs/exploration/total-montant-cheque-distribution-image", methods=['GET'])
def get_supplier_total_montant_cheque_distribution_image():
    """Generates and returns a PNG image of the histogram for 'Total montant cheque' for suppliers."""
    if df_suppliers_global is None: return jsonify({"error": "Données fournisseurs non disponibles."}), 500
    column = 'Total montant cheque'
    try:
        upper_bound = df_suppliers_global[column].quantile(0.99)
        filtered_data = df_suppliers_global[df_suppliers_global[column] <= upper_bound][column].dropna()
        if filtered_data.empty: return jsonify({"error": f"Pas de données valides pour '{column}' pour générer l'histogramme fournisseur."}), 404
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_data, bins=50, kde=True, ax=ax)
        ax.set_title(f'Distribution du "{column}" (Fournisseurs)')
        ax.set_xlabel(column)
        ax.set_ylabel('Nombre de transactions (Fréquence)')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image pour '{column}' (Fournisseurs) : {str(e)}"}), 500


@app.route("/fournisseurs/exploration/montant-cheque-distribution-image", methods=['GET'])
def get_supplier_montant_cheque_distribution_image():
    """Generates and returns a PNG image of the histogram for 'Montant_cheque' for suppliers."""
    if df_suppliers_global is None: return jsonify({"error": "Données fournisseurs non disponibles."}), 500
    column = 'Montant_cheque'
    try:
        upper_bound = df_suppliers_global[column].quantile(0.99)
        filtered_data = df_suppliers_global[df_suppliers_global[column] <= upper_bound][column].dropna()
        if filtered_data.empty: return jsonify({"error": f"Pas de données valides pour '{column}' pour générer l'histogramme fournisseur."}), 404
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_data, bins=50, kde=True, ax=ax)
        ax.set_title(f'Distribution du "{column}" (Fournisseurs)')
        ax.set_xlabel(column)
        ax.set_ylabel('Nombre de transactions (Fréquence)')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image pour '{column}' (Fournisseurs) : {str(e)}"}), 500


@app.route("/fournisseurs/exploration/nombre-distribution-image", methods=['GET'])
def get_supplier_nombre_distribution_image():
    """Generates and returns a PNG image of the histogram for 'Nombre' for suppliers."""
    if df_suppliers_global is None: return jsonify({"error": "Données fournisseurs non disponibles."}), 500
    column = 'Nombre'
    try:
        upper_bound = df_suppliers_global[column].quantile(0.99)
        filtered_data = df_suppliers_global[df_suppliers_global[column] <= upper_bound][column].dropna()
        if filtered_data.empty: return jsonify({"error": f"Pas de données valides pour '{column}' pour générer l'histogramme fournisseur."}), 404
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_data, bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution du "{column}" de chèques/transactions (Fournisseurs)')
        ax.set_xlabel('Nombre de chèques/transactions')
        ax.set_ylabel('Fréquence')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image pour '{column}' (Fournisseurs) : {str(e)}"}), 500


@app.route("/fournisseurs/exploration/categorical-distribution-image", methods=['GET'])
def get_supplier_categorical_distribution_image():
    """Generates and returns a PNG image of a bar chart for a specific supplier categorical column."""
    if df_suppliers_global is None: return jsonify({"error": "Données fournisseurs non disponibles."}), 500
    column = request.args.get('column')
    if not column: return jsonify({"error": "Parameter 'column' is missing from the request."}), 400
    categorical_cols_available = ['Activite_Economique', 'Segment', 'Sexe', 'Statut_Civil']
    if column not in categorical_cols_available:
        return jsonify({"error": f"Column '{column}' is not valid or available for supplier categorical analysis. Possible columns: {', '.join(categorical_cols_available)}"}), 400
    if column not in df_suppliers_global.columns:
        return jsonify({"error": f"Column '{column}' does not exist in the supplier DataFrame."}), 404
    try:
        df_to_plot = df_suppliers_global.copy()
        plot_order = None
        if column == 'Activite_Economique':
            top_items = df_to_plot['Activite_Economique'].value_counts().head(10).index
            df_to_plot = df_to_plot[df_to_plot['Activite_Economique'].isin(top_items)]
            plot_order = top_items
            title = f'Répartition des fournisseurs par "{column}" (Top 10)'
        else:
            plot_order = df_to_plot[column].value_counts().index
            title = f'Répartition des fournisseurs par "{column}"'
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=column, data=df_to_plot,
                      order=plot_order,
                      ax=ax, palette='rocket')
        ax.set_title(title)
        ax.set_xlabel('Nombre de fournisseurs')
        ax.set_ylabel(column)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error generating image for categorical distribution of '{column}' (Suppliers): {str(e)}"}), 500


@app.route("/fournisseurs/exploration/time-series-image", methods=['GET'])
def get_supplier_time_series_image():
    """Generates and returns a PNG image of the line chart for monthly 'Total montant cheque' for suppliers."""
    if df_suppliers_global is None: return jsonify({"error": "Données fournisseurs non disponibles."}), 500
    try:
        temp_df = df_suppliers_global.copy()
        temp_df['Date_Operation_dt'] = pd.to_datetime(temp_df['Date_Operation_dt_object'], errors='coerce')
        temp_df = temp_df.dropna(subset=['Date_Operation_dt'])
        if temp_df.empty: return jsonify({"error": "No valid date data for supplier time series analysis."}), 404

        temp_df['Annee_Mois'] = temp_df['Date_Operation_dt'].dt.to_period('M')
        monthly_total_suppliers = temp_df.groupby('Annee_Mois')['Total montant cheque'].sum().reset_index()
        monthly_total_suppliers['Annee_Mois_str'] = monthly_total_suppliers['Annee_Mois'].astype(str)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='Annee_Mois_str', y='Total montant cheque', data=monthly_total_suppliers, ax=ax, color='orange')
        ax.set_title('Évolution mensuelle du "Total montant cheque" (Fournisseurs)')
        ax.set_xlabel('Mois')
        ax.set_ylabel('Total montant cheque')
        ax.tick_params(axis='x', rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error generating supplier time series image: {str(e)}"}), 500

# --- NOUVEAUX ENDPOINTS : Analyse Approfondie des Paiements Fractionnés ---

@app.route("/clients/fragmentation/score-distribution-image", methods=['GET'])
def get_client_fragmentation_score_distribution_image():
    """
    Génère et renvoie une image PNG de l'histogramme du score de fragmentation pour les clients.
    """
    if client_features_extended_global is None:
        return jsonify({"error": "Données agrégées clients non disponibles."}), 500

    column = 'client_score_fragmentation'
    try:
        # Pas de filtration au 99e percentile ici, on veut voir toute la distribution des scores.
        # Mais on gère les NaN si jamais il y en a eu lors de l'agrégation
        data_to_plot = client_features_extended_global[column].dropna()

        if data_to_plot.empty:
            return jsonify({"error": f"Pas de données valides pour le score de fragmentation client."}), 404

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data_to_plot, bins=50, kde=True, ax=ax)
        ax.set_title('Distribution du Score de Fragmentation (Clients)')
        ax.set_xlabel('Score de Fragmentation Client')
        ax.set_ylabel('Nombre de clients')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image de la distribution du score de fragmentation client : {str(e)}"}), 500

@app.route("/fournisseurs/fragmentation/score-distribution-image", methods=['GET'])
def get_supplier_fragmentation_score_distribution_image():
    """
    Génère et renvoie une image PNG de l'histogramme du score de fragmentation pour les fournisseurs.
    """
    if supplier_features_extended_global is None:
        return jsonify({"error": "Données agrégées fournisseurs non disponibles."}), 500

    column = 'supplier_score_fragmentation'
    try:
        data_to_plot = supplier_features_extended_global[column].dropna()

        if data_to_plot.empty:
            return jsonify({"error": f"Pas de données valides pour le score de fragmentation fournisseur."}), 404

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data_to_plot, bins=50, kde=True, ax=ax)
        ax.set_title('Distribution du Score de Fragmentation (Fournisseurs)')
        ax.set_xlabel('Score de Fragmentation Fournisseur')
        ax.set_ylabel('Nombre de fournisseurs')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image de la distribution du score de fragmentation fournisseur : {str(e)}"}), 500

@app.route("/clients/fragmentation/score-by-segment-image", methods=['GET'])
def get_client_fragmentation_score_by_segment_image():
    """
    Génère et renvoie une image PNG d'un box plot du score de fragmentation
    des clients par Segment.
    """
    if client_features_extended_global is None:
        return jsonify({"error": "Données agrégées clients non disponibles."}), 500

    try:
        # Fusionner avec les données clients brutes pour obtenir la colonne 'Segment'
        # Attention: 'Segment' est déjà dans client_features_extended_global car mergé dans load_and_preprocess_data
        df_plot = client_features_extended_global.dropna(subset=['client_score_fragmentation', 'Segment'])

        if df_plot.empty:
            return jsonify({"error": "Pas de données valides pour le score de fragmentation par segment client."}), 404

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='client_score_fragmentation', y='Segment', data=df_plot,
                    order=df_plot['Segment'].value_counts().index, # Trier par fréquence des segments
                    ax=ax, palette='viridis', showfliers=False) # showfliers=False pour cacher les points extrêmes (outliers)
        ax.set_title('Score de Fragmentation par Segment Client')
        ax.set_xlabel('Score de Fragmentation')
        ax.set_ylabel('Segment')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image du score de fragmentation par segment client : {str(e)}"}), 500

@app.route("/fournisseurs/fragmentation/score-by-activity-image", methods=['GET'])
def get_supplier_fragmentation_score_by_activity_image():
    """
    Génère et renvoie une image PNG d'un box plot du score de fragmentation
    des fournisseurs par Activité Économique (Top 10).
    """
    if supplier_features_extended_global is None:
        return jsonify({"error": "Données agrégées fournisseurs non disponibles."}), 500

    try:
        df_plot = supplier_features_extended_global.dropna(subset=['supplier_score_fragmentation', 'Activite_Economique'])

        if df_plot.empty:
            return jsonify({"error": "Pas de données valides pour le score de fragmentation par activité fournisseur."}), 404

        # Sélectionner le Top 10 des activités économiques
        top_activities = df_plot['Activite_Economique'].value_counts().head(10).index
        df_filtered_activity = df_plot[df_plot['Activite_Economique'].isin(top_activities)]

        if df_filtered_activity.empty:
            return jsonify({"error": "Pas de données valides pour le Top 10 des activités économiques des fournisseurs."}), 404

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='supplier_score_fragmentation', y='Activite_Economique',
                    data=df_filtered_activity,
                    order=top_activities, # Conserver l'ordre du Top 10
                    ax=ax, palette='rocket', showfliers=False)
        ax.set_title('Score de Fragmentation par Activité Économique (Top 10 Fournisseurs)')
        ax.set_xlabel('Score de Fragmentation')
        ax.set_ylabel('Activité Économique')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image du score de fragmentation par activité fournisseur : {str(e)}"}), 500


# --- Main entry point for running the Flask app ---
if __name__ == "__main__":
    app.run(debug=True, port=8000)