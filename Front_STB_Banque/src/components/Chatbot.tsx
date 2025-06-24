import React, { useState, useRef, useEffect } from "react";

interface ChatBotProps {
  open: boolean;
  onClose: () => void;
}

const ChatBot: React.FC<ChatBotProps> = ({ open, onClose }) => {
  const [messages, setMessages] = useState<{ from: string; text: string }[]>(
    []
  );
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null); // Nouvelle référence pour le conteneur des messages

  useEffect(() => {
    if (open) setMessages([]);
  }, [open]);

  useEffect(() => {
    // Faire défiler le conteneur des messages jusqu'en bas
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { from: "user", text: input };
    setMessages((msgs) => [...msgs, userMsg]);
    setInput("");
    try {
      // const res = await fetch("http://127.0.0.1:5003/chat", {
      const res = await fetch("http://192.168.1.13:5003/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_message: input }),
      });
      const data = await res.json();
      setMessages((msgs) => [...msgs, { from: "bot", text: data.response }]);
    } catch {
      setMessages((msgs) => [
        ...msgs,
        { from: "bot", text: "Erreur de connexion au bot." },
      ]);
    }
  };

  if (!open) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 200,
        left: 1100,
        bottom: 24,
        right: 24,
        width: "90vw",
        maxWidth: 400,
        minWidth: 260,
        height: "70vh",
        maxHeight: 600,
        minHeight: 320,
        background: "white",
        borderRadius: 12,
        boxShadow: "0 4px 24px rgba(0,0,0,0.15)",
        zIndex: 999999,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #eee",
          background: "#0a3d62",
          color: "white",
          borderTopLeftRadius: 12,
          borderTopRightRadius: 12,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>Chat Espace Client</span>
        <button
          onClick={onClose}
          style={{
            background: "transparent",
            border: "none",
            color: "white",
            fontSize: 18,
            cursor: "pointer",
          }}
        >
          ×
        </button>
      </div>
      <div
        ref={messagesContainerRef} // Ajout de la référence au conteneur
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 16,
          background: "#f9f9f9",
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              textAlign: msg.from === "user" ? "right" : "left",
              marginBottom: 8,
            }}
          >
            <span
              style={{
                display: "inline-block",
                background: msg.from === "user" ? "#0a3d62" : "#eee",
                color: msg.from === "user" ? "white" : "#222",
                borderRadius: 16,
                padding: "8px 14px",
                maxWidth: 220,
                wordBreak: "break-word",
              }}
            >
              {msg.text}
            </span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form
        style={{
          display: "flex",
          borderTop: "1px solid #eee",
          padding: 8,
          background: "white",
        }}
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage();
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Votre message..."
          style={{
            flex: 1,
            border: "none",
            outline: "none",
            padding: 8,
            borderRadius: 8,
            background: "#f1f1f1",
            marginRight: 8,
          }}
        />
        <button
          type="submit"
          style={{
            background: "#0a3d62",
            color: "white",
            border: "none",
            borderRadius: 8,
            padding: "8px 16px",
            cursor: "pointer",
          }}
        >
          Envoyer
        </button>
      </form>
    </div>
  );
};

export default ChatBot;
