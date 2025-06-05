import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Menu, X, User } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import Login from "@/pages/Login";
import Register from "@/pages/Register";
import stbLogo from "@/assets/stb.png";
import ChatBot from "./Chatbot";

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [showChat, setShowChat] = useState(false);
  const navigate = useNavigate();

  const navigationItems = user
    ? [
        { name: "Accueil", href: "/" },
        { name: "À Propos", href: "/about" },
        { name: "Services", href: "/services" },
        { name: "Dashboard BI", href: "/dashboard" },
        { name: "Intelligence Artificielle", href: "/ai" },
      ]
    : [
        { name: "Accueil", href: "/" },
        { name: "À Propos", href: "/about" },
        { name: "Services", href: "/services" },
      ];
  // Ajoute ceci :
  useEffect(() => {
    const token = localStorage.getItem("idToken");
    if (token) {
      const userData = localStorage.getItem("user");
      if (userData) {
        setUser(JSON.parse(userData));
      }
    }
  }, []);

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem("idToken");
    localStorage.removeItem("user");
    navigate("/"); // Redirige vers l'accueil après déconnexion
  };

  const handleNav = (href: string) => {
    navigate(href);
  };

  return (
    <header className="sticky top-0 z-50 glass-effect border-b">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <button
            onClick={() => handleNav("/")}
            className="flex items-center space-x-3 bg-transparent border-none outline-none cursor-pointer"
          >
            <img
              src={stbLogo}
              alt="STB Logo"
              className="w-12 h-12 object-contain"
            />
            <div className="hidden sm:block">
              <p className="text-xs text-gray-600">
                Société Tunisienne de Banque
              </p>
            </div>
          </button>
          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center space-x-8">
            {navigationItems.map((item) => (
              <button
                key={item.name}
                onClick={() => handleNav(item.href)}
                className="text-gray-700 hover:text-stb-blue transition-colors duration-200 font-medium bg-transparent border-none outline-none cursor-pointer"
              >
                {item.name}
              </button>
            ))}
          </nav>

          {/* Actions à droite */}
          <div className="hidden lg:flex items-center space-x-4">
            {user ? (
              <>
                <span className="flex items-center gap-1 bg-blue-50 px-3 py-1 rounded-full text-blue-700 font-medium">
                  <User size={16} />
                  Bonjour {user.prenom}
                </span>
                <Button
                  variant="outline"
                  className="ml-2"
                  onClick={handleLogout}
                >
                  Déconnexion
                </Button>
              </>
            ) : (
              <Button
                variant="outline"
                className="mr-2"
                onClick={() => setShowLogin(true)}
              >
                Se connecter
              </Button>
            )}
            <Button
              className="gradient-stb text-white hover:opacity-90 ml-2"
              onClick={() => setShowChat(true)}
            >
              Espace Client
            </Button>
          </div>
          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </Button>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="lg:hidden border-t bg-white/95 backdrop-blur-sm">
            <nav className="py-4 space-y-2">
              {navigationItems.map((item) => (
                <button
                  key={item.name}
                  onClick={() => {
                    handleNav(item.href);
                    setIsMenuOpen(false);
                  }}
                  className="block w-full text-left px-4 py-2 text-gray-700 hover:text-stb-blue hover:bg-gray-50 transition-colors duration-200 bg-transparent border-none outline-none cursor-pointer"
                >
                  {item.name}
                </button>
              ))}
              <div className="px-4 pt-4 border-t space-y-2">
                {user ? (
                  <>
                    <span className="flex items-center justify-center gap-1 bg-blue-50 px-3 py-1 rounded-full text-blue-700 font-medium">
                      <User size={16} />
                      Bonjour {user.prenom}
                    </span>
                    <Button
                      variant="outline"
                      className="w-full mb-2 mt-2"
                      onClick={handleLogout}
                    >
                      Déconnexion
                    </Button>
                  </>
                ) : (
                  <Button
                    variant="outline"
                    className="w-full mb-2"
                    onClick={() => {
                      setShowLogin(true);
                      setIsMenuOpen(false);
                    }}
                  >
                    Se connecter
                  </Button>
                )}
                <Button
                  className="w-full gradient-stb text-white hover:opacity-90 mt-2"
                  onClick={() => {
                    setShowChat(true);
                    setIsMenuOpen(false);
                  }}
                >
                  Espace Client
                </Button>
              </div>
            </nav>
          </div>
        )}
      </div>
      <Login
        open={showLogin}
        onClose={() => setShowLogin(false)}
        onSwitchRegister={() => {
          setShowLogin(false);
          setShowRegister(true);
        }}
        onLogin={setUser}
      />
      <Register
        open={showRegister}
        onClose={() => setShowRegister(false)}
        onSwitchLogin={() => {
          setShowRegister(false);
          setShowLogin(true);
        }}
      />
      <ChatBot open={showChat} onClose={() => setShowChat(false)} />
    </header>
  );
};

export default Header;
