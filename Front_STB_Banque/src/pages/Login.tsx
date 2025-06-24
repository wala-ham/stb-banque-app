import React, { useState } from "react";
import { toast } from "react-toastify";

const Login = ({ open, onClose, onSwitchRegister, onLogin }) => {
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
      const res = await fetch("http://192.168.1.13:5003/auth/signin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (data.success) {
        localStorage.setItem("idToken", data.user.idToken);
        // Extraire le prénom depuis l'email (avant le premier point ou @)
        let prenom = data.user.email.split("@")[0];
        if (prenom.includes(".")) {
          prenom = prenom.split(".")[0];
        }
        prenom = prenom.charAt(0).toUpperCase() + prenom.slice(1);
        const userToStore = { ...data.user, prenom };
        localStorage.setItem("user", JSON.stringify(userToStore));
        if (onLogin) onLogin(userToStore);
        setEmail("");
        setPassword("");
        onClose();
        toast.success("Connexion réussie !");
      } else {
        setError(data.error || "Erreur de connexion.");
      }
    } catch (err) {
      setError("Erreur de connexion.");
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
        <h2 className="text-2xl font-bold mb-6 text-center">Se connecter</h2>
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
            {loading ? "Connexion..." : "Connexion"}
          </button>
          {error && (
            <div className="mt-2 text-red-600 text-sm text-center">{error}</div>
          )}
        </form>
        <div className="mt-4 text-center">
          <button
            onClick={() => {
              onClose();
              onSwitchRegister();
            }}
            className="text-blue-600 hover:underline"
          >
            Pas de compte ? S'inscrire
          </button>
        </div>
      </div>
    </div>
  );
};

export default Login;
