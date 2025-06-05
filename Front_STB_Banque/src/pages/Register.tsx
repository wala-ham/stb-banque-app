import React, { useState } from "react";
import { toast } from "react-toastify";

const Register = ({ open, onClose, onSwitchLogin }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  if (!open) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:5003/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (data.success) {
        toast.success("Compte créé avec succès ! Connectez-vous.");
        setEmail("");
        setPassword("");
        onClose();
        if (onSwitchLogin) onSwitchLogin();
      } else {
        setError(data.error || "Erreur lors de l'inscription.");
      }
    } catch (err) {
      setError("Erreur lors de l'inscription.");
    }
    setLoading(false);
  };

  return (
    <div className="fixed inset-0 min-h-screen flex items-center justify-center bg-black/40 z-50">
      <div className="bg-white rounded-lg shadow-lg p-8 w-full max-w-md relative">
        <button
          onClick={onClose}
          className="absolute top-2 right-2 text-gray-500 hover:text-gray-800"
        >
          ×
        </button>
        <h2 className="text-2xl font-bold mb-6 text-center">Créer un compte</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="email"
            placeholder="Email"
            className="w-full mb-4 p-2 border rounded"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Mot de passe"
            className="w-full mb-4 p-2 border rounded"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 rounded"
            disabled={loading}
          >
            {loading ? "Inscription..." : "S'inscrire"}
          </button>
          {error && (
            <div className="mt-2 text-red-600 text-sm text-center">{error}</div>
          )}
        </form>
        <div className="mt-4 text-center">
          <button
            onClick={() => {
              onClose();
              onSwitchLogin();
            }}
            className="text-blue-600 hover:underline"
          >
            Déjà un compte ? Se connecter
          </button>
        </div>
      </div>
    </div>
  );
};

export default Register;
