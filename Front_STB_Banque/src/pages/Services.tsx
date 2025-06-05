import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Services = () => {
  const personalServices = [
    {
      title: "Comptes Courants",
      description:
        "Gestion quotidienne de vos finances avec carte bancaire incluse",
      features: [
        "Carte bancaire gratuite",
        "Virements illimités",
        "Relevés en ligne",
        "Découvert autorisé",
      ],
      icon: "💳",
    },
    {
      title: "Épargne",
      description: "Solutions d'épargne adaptées à vos objectifs",
      features: [
        "Livret d'épargne",
        "Compte à terme",
        "Plan épargne logement",
        "Épargne retraite",
      ],
      icon: "💰",
    },
    {
      title: "Crédits",
      description: "Financement de vos projets personnels et immobiliers",
      features: [
        "Crédit personnel",
        "Crédit immobilier",
        "Crédit auto",
        "Rachat de crédits",
      ],
      icon: "🏠",
    },
    {
      title: "Assurances",
      description: "Protection complète pour vous et votre famille",
      features: [
        "Assurance vie",
        "Assurance habitation",
        "Assurance auto",
        "Assurance santé",
      ],
      icon: "🛡️",
    },
  ];

  const businessServices = [
    {
      title: "Comptes Professionnels",
      description: "Solutions bancaires adaptées aux besoins des entreprises",
      features: [
        "Compte courant professionnel",
        "Terminal de paiement",
        "Chéquier professionnel",
        "Virements en masse",
      ],
      icon: "🏢",
    },
    {
      title: "Financement",
      description:
        "Accompagnement financier pour le développement de votre activité",
      features: [
        "Crédit d'investissement",
        "Crédit de fonctionnement",
        "Leasing",
        "Factoring",
      ],
      icon: "📈",
    },
    {
      title: "Commerce International",
      description: "Facilitation de vos échanges avec l'étranger",
      features: [
        "Crédit documentaire",
        "Remise documentaire",
        "Garanties internationales",
        "Change",
      ],
      icon: "🌍",
    },
    {
      title: "Cash Management",
      description: "Optimisation de la gestion de votre trésorerie",
      features: [
        "Placement automatique",
        "Consolidation de comptes",
        "Reporting en temps réel",
        "Prévisions de trésorerie",
      ],
      icon: "💼",
    },
  ];

  const digitalServices = [
    {
      title: "Application Mobile",
      description: "Banque dans votre poche 24h/24",
      features: [
        "Consultation de comptes",
        "Virements instantanés",
        "Paiement factures",
        "Géolocalisation agences",
      ],
      icon: "📱",
    },
    {
      title: "E-Banking",
      description: "Plateforme web sécurisée pour tous vos besoins bancaires",
      features: [
        "Interface intuitive",
        "Historique détaillé",
        "Téléchargement documents",
        "Support client intégré",
      ],
      icon: "💻",
    },
    {
      title: "Paiement Mobile",
      description: "Solutions de paiement innovantes",
      features: [
        "Paiement sans contact",
        "QR Code",
        "Portefeuille électronique",
        "Paiement entre particuliers",
      ],
      icon: "📲",
    },
    {
      title: "API Banking",
      description: "Intégration bancaire pour les développeurs",
      features: [
        "API REST sécurisées",
        "Documentation complète",
        "Sandbox de test",
        "Support technique dédié",
      ],
      icon: "⚙️",
    },
  ];

  return (
    <div className="min-h-screen bg-white">
      <Header />

      {/* Hero Section */}
      <section className="gradient-stb text-white py-20">
        <div className="container mx-auto px-6 text-center">
          <h1 className="text-5xl font-bold mb-6">Nos Services</h1>
          <p className="text-xl max-w-3xl mx-auto">
            Une gamme complète de solutions bancaires pour particuliers et
            entreprises, avec la technologie la plus avancée du secteur
            financier.
          </p>
        </div>
      </section>

      <div className="container mx-auto px-6 py-16">
        <Tabs defaultValue="personal" className="space-y-8">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="personal">Particuliers</TabsTrigger>
            <TabsTrigger value="business">Entreprises</TabsTrigger>
            <TabsTrigger value="digital">Solutions Digitales</TabsTrigger>
          </TabsList>

          {/* Personal Services */}
          <TabsContent value="personal" className="space-y-8">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-stb-blue mb-4">
                Services aux Particuliers
              </h2>
              <p className="text-xl text-gray-600">
                Des solutions personnalisées pour tous vos besoins financiers
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {personalServices.map((service, index) => (
                <Card key={index} className="hover-lift">
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <span className="text-3xl">{service.icon}</span>
                      <CardTitle className="text-stb-blue">
                        {service.title}
                      </CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-600 mb-4">{service.description}</p>
                    <ul className="space-y-2 mb-6">
                      {service.features.map((feature, idx) => (
                        <li key={idx} className="flex items-center text-sm">
                          <div className="w-2 h-2 bg-stb-blue rounded-full mr-3"></div>
                          {feature}
                        </li>
                      ))}
                    </ul>
                    <Button className="w-full gradient-stb text-white">
                      En savoir plus
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Business Services */}
          <TabsContent value="business" className="space-y-8">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-stb-blue mb-4">
                Services aux Entreprises
              </h2>
              <p className="text-xl text-gray-600">
                Accompagnement sur mesure pour le développement de votre
                business
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {businessServices.map((service, index) => (
                <Card key={index} className="hover-lift">
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <span className="text-3xl">{service.icon}</span>
                      <CardTitle className="text-stb-blue">
                        {service.title}
                      </CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-600 mb-4">{service.description}</p>
                    <ul className="space-y-2 mb-6">
                      {service.features.map((feature, idx) => (
                        <li key={idx} className="flex items-center text-sm">
                          <div className="w-2 h-2 bg-stb-blue rounded-full mr-3"></div>
                          {feature}
                        </li>
                      ))}
                    </ul>
                    <Button className="w-full gradient-stb text-white">
                      Demander un devis
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Digital Services */}
          <TabsContent value="digital" className="space-y-8">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-stb-blue mb-4">
                Solutions Digitales
              </h2>
              <p className="text-xl text-gray-600">
                L'innovation technologique au service de votre expérience
                bancaire
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {digitalServices.map((service, index) => (
                <Card key={index} className="hover-lift">
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <span className="text-3xl">{service.icon}</span>
                      <CardTitle className="text-stb-blue">
                        {service.title}
                      </CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-600 mb-4">{service.description}</p>
                    <ul className="space-y-2 mb-6">
                      {service.features.map((feature, idx) => (
                        <li key={idx} className="flex items-center text-sm">
                          <div className="w-2 h-2 bg-stb-blue rounded-full mr-3"></div>
                          {feature}
                        </li>
                      ))}
                    </ul>
                    <Button className="w-full gradient-stb text-white">
                      Télécharger
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>

        {/* CTA Section */}
        <section className="text-center py-16 mt-16 gradient-stb-light rounded-3xl">
          <h2 className="text-3xl font-bold text-stb-blue mb-4">
            Besoin d'aide pour choisir ?
          </h2>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Nos conseillers experts sont à votre disposition pour vous guider
            vers les solutions les plus adaptées à vos besoins.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="gradient-stb text-white px-8">
              Prendre rendez-vous
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-stb-blue text-stb-blue px-8"
            >
              Nous contacter
            </Button>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
};

export default Services;
