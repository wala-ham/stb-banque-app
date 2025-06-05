import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Card, CardContent } from "@/components/ui/card";

const About = () => {
  const timeline = [
    {
      year: "1958",
      event: "Création de la STB",
      description: "Fondation de la Société Tunisienne de Banque",
    },
    {
      year: "1970",
      event: "Expansion nationale",
      description: "Ouverture de 25 nouvelles agences",
    },
    {
      year: "1990",
      event: "Modernisation",
      description: "Informatisation complète du réseau",
    },
    {
      year: "2005",
      event: "Banque digitale",
      description: "Lancement des services en ligne",
    },
    {
      year: "2020",
      event: "Intelligence artificielle",
      description: "Intégration de l'IA et du machine learning",
    },
    {
      year: "2024",
      event: "Innovation continue",
      description: "Nouvelle plateforme d'analyse prédictive",
    },
  ];

  const leaders = [
    {
      name: "Ahmed Ben Salem",
      position: "Président Directeur Général",
      description:
        "Expert bancaire avec 25 ans d'expérience dans le secteur financier tunisien",
    },
    {
      name: "Fatma Gharbi",
      position: "Directrice Générale Adjointe",
      description:
        "Spécialisée en transformation digitale et innovation bancaire",
    },
    {
      name: "Mohamed Triki",
      position: "Directeur des Risques",
      description: "Expert en gestion des risques et conformité réglementaire",
    },
  ];

  return (
    <div className="min-h-screen bg-white">
      <Header />

      {/* Hero Section */}
      <section className="gradient-stb text-white py-20">
        <div className="container mx-auto px-6 text-center">
          <h1 className="text-5xl font-bold mb-6">À Propos de la STB</h1>
          <p className="text-xl max-w-3xl mx-auto">
            Depuis plus de 65 ans, la Société Tunisienne de Banque accompagne le
            développement économique de la Tunisie avec innovation et
            excellence.
          </p>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="py-16">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <Card className="text-center hover-lift">
              <CardContent className="p-8">
                <div className="text-4xl mb-4">🎯</div>
                <h3 className="text-xl font-bold text-stb-blue mb-4">
                  Notre Mission
                </h3>
                <p className="text-gray-600">
                  Accompagner nos clients dans leurs projets en offrant des
                  solutions bancaires innovantes et personnalisées, tout en
                  contribuant au développement économique de la Tunisie.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center hover-lift">
              <CardContent className="p-8">
                <div className="text-4xl mb-4">👁️</div>
                <h3 className="text-xl font-bold text-stb-blue mb-4">
                  Notre Vision
                </h3>
                <p className="text-gray-600">
                  Être la banque de référence en Tunisie, reconnue pour son
                  expertise, son innovation technologique et la qualité de sa
                  relation client.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center hover-lift">
              <CardContent className="p-8">
                <div className="text-4xl mb-4">⭐</div>
                <h3 className="text-xl font-bold text-stb-blue mb-4">
                  Nos Valeurs
                </h3>
                <p className="text-gray-600">
                  Intégrité, excellence, innovation, proximité client et
                  responsabilité sociétale guident toutes nos actions au
                  quotidien.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Timeline */}
      <section className="py-16 bg-stb-gray">
        <div className="container mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-stb-blue mb-4">
              Notre Histoire
            </h2>
            <p className="text-xl text-gray-600">
              Plus de 6 décennies d'excellence bancaire
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            {timeline.map((item, index) => (
              <div key={index} className="flex items-center mb-8 last:mb-0">
                <div className="w-24 text-right mr-8">
                  <span className="text-2xl font-bold text-stb-blue">
                    {item.year}
                  </span>
                </div>
                <div className="w-4 h-4 bg-stb-blue rounded-full mr-8 flex-shrink-0"></div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-stb-blue mb-1">
                    {item.event}
                  </h3>
                  <p className="text-gray-600">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Leadership */}
      <section className="py-16">
        <div className="container mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-stb-blue mb-4">
              Notre Direction
            </h2>
            <p className="text-xl text-gray-600">
              Une équipe expérimentée au service de votre réussite
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {leaders.map((leader, index) => (
              <Card key={index} className="text-center hover-lift">
                <CardContent className="p-8">
                  <div className="w-24 h-24 bg-stb-blue rounded-full mx-auto mb-4 flex items-center justify-center">
                    <span className="text-white text-2xl font-bold">
                      {leader.name
                        .split(" ")
                        .map((n) => n[0])
                        .join("")}
                    </span>
                  </div>
                  <h3 className="text-xl font-bold text-stb-blue mb-2">
                    {leader.name}
                  </h3>
                  <p className="text-stb-gold font-semibold mb-4">
                    {leader.position}
                  </p>
                  <p className="text-gray-600 text-sm">{leader.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Network */}
      <section className="py-16 gradient-stb text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold mb-8">Notre Réseau</h2>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            <div>
              <div className="text-4xl font-bold text-stb-gold mb-2">120+</div>
              <p className="text-lg">Agences</p>
            </div>
            <div>
              <div className="text-4xl font-bold text-stb-gold mb-2">250+</div>
              <p className="text-lg">Distributeurs</p>
            </div>
            <div>
              <div className="text-4xl font-bold text-stb-gold mb-2">24</div>
              <p className="text-lg">Gouvernorats</p>
            </div>
            <div>
              <div className="text-4xl font-bold text-stb-gold mb-2">2.5M</div>
              <p className="text-lg">Clients</p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default About;
