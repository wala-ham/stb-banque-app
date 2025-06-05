import pyrebase
import json
from flask import Flask, request, jsonify # Import Flask specific modules
from flask_cors import CORS


# --- Firebase Configuration Import ---
# IMPORTANT: Ensure firebase_config.py is in the same directory
# and contains your Firebase project configuration dictionary.
try:
    from firebase_config import firebase_config
    firebase = pyrebase.initialize_app(firebase_config)
    auth = firebase.auth()
    print("Firebase initialized successfully.")
except ImportError:
    print("Error: firebase_config.py not found. Please create it with your Firebase project config.")
    exit(1)
except Exception as e:
    print(f"Error initializing Firebase: {e}. Check your firebase_config.py details.")
    exit(1)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) 

# --- Helper function to parse Firebase errors ---
def parse_firebase_error(e):
    """
    Parses Pyrebase exceptions to extract a readable error message.
    """
    try:
        # Pyrebase usually wraps the error in args[1] as a JSON string
        error_json = e.args[1]
        error_data = json.loads(error_json)
        error_message = error_data['error']['message']
        return error_message
    except (IndexError, TypeError, json.JSONDecodeError, KeyError):
        # Fallback if the error structure is different or unexpected
        return str(e)

# --- API Endpoints for Authentication ---

@app.route("/auth/signup", methods=['POST'])
def signup_api():
    """
    API endpoint for user registration with Firebase.
    Expects JSON body: {"email": "user@example.com", "password": "securepassword"}
    Returns JSON: {"success": True, "user": {...}} or {"success": False, "error": "message"}
    """
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400

    try:
        user = auth.create_user_with_email_and_password(email, password)
        # Return only necessary user info, avoid sending sensitive data
        return jsonify({'success': True, 'user': {'email': user['email'], 'localId': user['localId']}}), 201 # 201 Created
    except Exception as e:
        error_message = parse_firebase_error(e)
        return jsonify({'success': False, 'error': error_message}), 400 # 400 Bad Request

@app.route("/auth/signin", methods=['POST'])
def signin_api():
    """
    API endpoint for user login with Firebase.
    Expects JSON body: {"email": "user@example.com", "password": "securepassword"}
    Returns JSON: {"success": True, "user": {...}} or {"success": False, "error": "message"}
    """
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400

    try:
        user = auth.sign_in_with_email_and_password(email, password)
        # Pyrebase user object contains idToken, refreshToken, localId, etc.
        # These tokens are crucial for the frontend to maintain session with Firebase.
        return jsonify({'success': True, 'user': {
            'email': user['email'],
            'localId': user['localId'],
            'idToken': user['idToken'],        # Important for Firebase API calls
            'refreshToken': user['refreshToken'] # Important for refreshing idToken
        }}), 200 # 200 OK
    except Exception as e:
        error_message = parse_firebase_error(e)
        return jsonify({'success': False, 'error': error_message}), 401 # 401 Unauthorized

@app.route("/auth/logout", methods=['POST'])
def logout_api():
    """
    API endpoint for user logout.
    In Pyrebase/Firebase Auth, logout primarily means discarding tokens on the client side.
    This API endpoint serves as a clear indication for the frontend to clear its session/tokens.
    It does not perform a server-side "logout" on Firebase itself, as that's token-based.
    """
    # For Firebase Auth, logout is mainly a client-side action of discarding tokens.
    # We return success to indicate the API acknowledged the request.
    return jsonify({'success': True, 'message': 'Client-side logout recommended (clear tokens).'}), 200

# --- Launch the Flask API ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003) # Run on port 5003 to avoid conflicts