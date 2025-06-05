import pandas as pd
import numpy as np # Needed for median fillna and other numeric operations
import io
import os # For getting the API key from environment variable if available
from flask_cors import CORS

from flask import Flask, request, jsonify # Import Flask specific modules
import google.generativeai as genai

# --- Configuration Gemini API ---
# IMPORTANT: Replace "AIzaSyBoCEyoVBzlT-Toc7j8I52qa9mjTM38UYY" with your actual API key.
# This key was found in your original Streamlit code.
GEMINI_API_KEY = "AIzaSyBoCEyoVBzlT-Toc7j8I52qa9mjTM38UYY" # Your API key from the Streamlit app

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}. Please check your API key.")
    exit(1) # Exit if API configuration fails

# Using gemini-1.5-flash as previously decided for speed and context
MODEL_NAME = "gemini-1.5-flash"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error initializing Gemini model '{MODEL_NAME}': {e}. Check API key and model name.")
    exit(1)

# --- Data Loading and Summary for LLM Context ---
def load_data_and_get_summaries(client_file='Client - Sheet1.csv', supplier_file='Fournisseur - Sheet1.csv'):
    """
    Loads client and supplier data, performs minimal preprocessing, and generates
    summaries and info strings to be used as context for the LLM.
    """
    client_summary = "Client data not available."
    supplier_summary = "Supplier data not available."
    client_info_str = "Client data structure not available."
    supplier_info_str = "Supplier data structure not available."

    # Helper for minimal preprocessing for summary/info
    def preprocess_for_summary(df):
        # Convert date columns for info() and describe()
        for col in ['Date_Operation', 'Date_Naissance']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        # Fill NaNs for describe(include='all') and info()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna('Inconnu')
        for col in df.select_dtypes(include=np.number).columns:
            # Ensure numeric_only=True for median() to avoid warnings/errors with non-numeric data
            df[col] = df[col].fillna(df[col].median(numeric_only=True))
        return df

    try:
        df_clients = pd.read_csv(client_file)
        df_clients = preprocess_for_summary(df_clients)
        client_summary = df_clients.describe(include='all').to_markdown()
        buffer_client = io.StringIO()
        df_clients.info(buf=buffer_client, verbose=True, show_counts=True)
        client_info_str = buffer_client.getvalue()
    except FileNotFoundError:
        print(f"Warning: Client file '{client_file}' not found for LLM context.")
    except Exception as e:
        print(f"Warning: Error processing client file for LLM context: {e}")

    try:
        df_suppliers = pd.read_csv(supplier_file)
        df_suppliers = preprocess_for_summary(df_suppliers)
        supplier_summary = df_suppliers.describe(include='all').to_markdown()
        buffer_supplier = io.StringIO()
        df_suppliers.info(buf=buffer_supplier, verbose=True, show_counts=True)
        supplier_info_str = buffer_supplier.getvalue()
    except FileNotFoundError:
        print(f"Warning: Supplier file '{supplier_file}' not found for LLM context.")
    except Exception as e:
        print(f"Warning: Error processing supplier file for LLM context: {e}")

    return client_summary, supplier_summary, client_info_str, supplier_info_str

# Load data summaries once at API startup
client_summary_global, supplier_summary_global, client_info_str_global, supplier_info_str_global = \
    load_data_and_get_summaries()

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) 
# --- Root Endpoint (optional) ---
@app.route("/", methods=['GET'])
def read_root():
    return jsonify({"message": "Chatbot API (Flask) is running. Use /chat to interact."})

# --- Chat Endpoint ---
@app.route("/chat", methods=['POST'])
def chat_with_data():
    """
    Allows interaction with the Gemini language model to ask questions about the data.
    Takes the user's message and conversation history.
    """
    data = request.get_json()
    user_message = data.get("user_message")
    chat_history = data.get("chat_history", []) # Default to empty list if not provided

    if not user_message:
        return jsonify({"error": "user_message is required."}), 400

    # Build the system prompt with data context
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
    {client_summary_global}

    And its structure:
    {client_info_str_global}

    The 'Fournisseur' (Supplier) dataset contains payment transaction information for suppliers. Key columns include:
    - 'Compte_Key': Unique supplier identifier.
    - 'Date_Operation': Date of the transaction.
    - 'Total montant cheque': Total amount for the operation.
    - 'Montant_cheque': Amount of a single cheque/transaction within the operation.
    - 'Nombre': Number of cheques/transactions for that operation (relevant for split payments).
    - 'Activite_Economique', 'Segment', 'Sexe', 'Statut_Civil': Supplier business/demographic information.
    - 'is_split_record': Binary flag (1 if 'Nombre' > 1, 0 otherwise).

    Here is a summary of the supplier data:
    {supplier_summary_global}

    And its structure:
    {supplier_info_str_global}

    Your goal is to answer questions about these datasets. If a question requires precise numerical calculations that are not immediately available in the summaries, state that you can provide insights based on the available data, or that a precise calculation would require more in-depth statistical analysis or direct querying of the raw data.
    Focus on providing clear, concise, and insightful answers.
    """

    try:
        # Initialize chat session with the provided history
        # Gemini's `start_chat` expects history to be in the format: [{"role": "user/model", "parts": ["content"]}]
        # For the first user message (when history is empty), we'll prepend the system_prompt to the user's message.
        if not chat_history:
            # For the first turn, combine system prompt with user's initial message
            # This is a common pattern when `system_instruction` is not explicitly supported by the model
            # or when you want to ensure the context is always present for the first message.
            messages_for_model = [{"role": "user", "parts": [f"{system_prompt}\n\nQuestion de l'utilisateur: {user_message}"]}]
        else:
            # If there's history, just add the current user message to it
            messages_for_model = chat_history + [{"role": "user", "parts": [user_message]}]


        # Create a new chat session for each request, passing the combined history
        # This is important for Flask as it's not inherently asynchronous like FastAPI,
        # so keeping state like 'chat' object across requests is not trivial.
        chat = model.start_chat(history=messages_for_model)

        # Send the last message (which is always the user's current message in messages_for_model)
        # Note: If messages_for_model already contains the full history, you'd usually call
        # chat.send_message(user_message). However, due to how system_prompt is handled on first turn,
        # we construct `messages_for_model` to effectively pass the full context.
        # A simpler way for Gemini models is to pass the full history to `start_chat`
        # and then send only the *new* message. Let's adjust for clarity.

        # Re-structure: The `start_chat` method takes a `history` argument
        # that should contain all previous messages *except* the current one.
        # So, we pass the `chat_history` from the request to `start_chat`,
        # and then `send_message` with the `user_message`.
        # The system_prompt is the initial context, it's not part of history in the same way.
        # For models like gemini-1.5-flash, the `system_instruction` parameter in `GenerativeModel`
        # or `start_chat` is the best way to handle this.
        # If using older `gemini-pro`, prepending to the first user message is common.

        # Let's adjust for the recommended way for models like Gemini 1.5 Flash:
        # Pass the system prompt once when starting chat.
        # However, `start_chat` might not always accept system_instruction directly if used after `GenerativeModel`.
        # The most reliable way for models that don't take a system_instruction arg in `start_chat`
        # is to include it as a "hidden" first message or prepend it to the first user message.

        # Let's use the simplest, most compatible approach for Flask:
        # Construct the full message list including system context for EACH call for simplicity.
        # This ensures the model always gets the full picture.
        conversation_with_context = [
            {"role": "user", "parts": [system_prompt]}, # Treat system prompt as an initial "user" instruction
            {"role": "model", "parts": ["Compris. Je suis prêt à analyser vos données. Comment puis-je vous aider ?"]} # A model's response to the system prompt
        ] + chat_history + [{"role": "user", "parts": [user_message]}] # Add the actual user history and current message

        # Now send the entire `conversation_with_context` to the model.
        # We need to construct a new `chat` object with the full history for each turn.
        # This is typical for stateless API requests.
        response = model.generate_content(conversation_with_context)


        # Extract text response, handling potential empty responses
        response_text = ""
        if response.parts:
            response_text = response.parts[0].text
        elif response.candidates and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
        else:
            print(f"Warning: Gemini returned an empty or unexpected response: {response}")
            response_text = "Désolé, je n'ai pas pu générer une réponse. Le modèle a retourné un contenu vide ou inattendu."


        return jsonify({"response": response_text})

    except Exception as e:
        # Capture specific API errors or general exceptions
        print(f"Error during Gemini API call: {e}") # Log the full error for debugging
        if "BlockedPromptException" in str(e):
            return jsonify({"error": "Votre requête a été bloquée en raison de violations des politiques de sécurité."}), 400
        elif "ResourceExhausted" in str(e):
            return jsonify({"error": "Limite de requêtes atteinte. Veuillez réessayer plus tard."}), 429
        return jsonify({"error": f"Erreur interne du chatbot : {str(e)}"}), 500

# --- Launch the Flask API ---
if __name__ == "__main__":
    # This block is for running this file directly for testing.
    # Ensure CSV files are in the same directory or adjust paths.
    app.run(debug=True, host="0.0.0.0", port=5002) # Run on port 5002 to avoid conflicts