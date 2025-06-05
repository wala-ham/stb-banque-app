import { Button } from "@/components/ui/button";
import { ArrowDown, ChevronDown } from "lucide-react";
import { Link } from "react-router-dom";

const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Gradient */}
      <div className="absolute inset-0 gradient-stb"></div>

      {/* Animated Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-32 h-32 bg-white/10 rounded-full blur-xl animate-pulse"></div>
        <div className="absolute bottom-20 right-10 w-48 h-48 bg-white/5 rounded-full blur-2xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/3 w-24 h-24 bg-stb-gold/20 rounded-full blur-lg animate-pulse delay-500"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 text-center text-white">
        <div className="max-w-4xl mx-auto animate-fade-in">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            Votre Banque
            <span className="block text-stb-gold">Digitale</span>
            <span className="block">de Confiance</span>
          </h1>

          <p className="text-xl md:text-2xl mb-8 opacity-90 max-w-3xl mx-auto leading-relaxed">
            Découvrez l'innovation bancaire avec la STB. Des solutions
            financières modernes alimentées par l'intelligence artificielle pour
            particuliers et entreprises.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
            <Link to="/services">
              <Button
                size="lg"
                className="bg-white text-stb-blue hover:bg-gray-100 px-8 py-4 text-lg font-semibold"
              >
                Découvrir nos Services
              </Button>
            </Link>
            <Link to="/dashboard">
              <Button
                size="lg"
                className="bg-white text-stb-blue hover:bg-gray-100 px-8 py-4 text-lg font-semibold"
              >
                Dashboard BI
              </Button>
            </Link>
          </div>

          {/* Key Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-16">
            <div className="text-center animate-slide-up">
              <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🤖</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">
                Intelligence Artificielle
              </h3>
              <p className="opacity-80">
                Analyse prédictive et insights avancés
              </p>
            </div>

            <div className="text-center animate-slide-up delay-200">
              <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">
                Business Intelligence
              </h3>
              <p className="opacity-80">
                Tableaux de bord interactifs en temps réel
              </p>
            </div>

            <div className="text-center animate-slide-up delay-300">
              <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">💬</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">
                Chatbot Intelligent
              </h3>
              <p className="opacity-80">Assistant virtuel 24h/24 et 7j/7</p>
            </div>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <ChevronDown className="w-8 h-8 text-white/70" />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
