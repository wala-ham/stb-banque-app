
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

const ServicesSection = () => {
  const services = [
    {
      title: "Particuliers",
      description: "Comptes courants, épargne, crédits personnels et cartes bancaires",
      icon: "👤",
      features: ["Compte courant", "Livret d'épargne", "Crédit personnel", "Carte bancaire"]
    },
    {
      title: "Entreprises", 
      description: "Solutions bancaires complètes pour développer votre business",
      icon: "🏢",
      features: ["Compte professionnel", "Crédit entreprise", "Commerce international", "Cash management"]
    },
    {
      title: "Banque Digitale",
      description: "Services bancaires 100% en ligne avec sécurité maximale",
      icon: "📱",
      features: ["Application mobile", "Virement instantané", "Paiement mobile", "E-banking"]
    },
    {
      title: "Investissement",
      description: "Produits d'investissement et conseil en gestion de patrimoine",
      icon: "💰",
      features: ["Bourse en ligne", "OPCVM", "Assurance vie", "Conseil patrimonial"]
    }
  ];

  return (
    <section className="py-20 bg-stb-gray">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold text-stb-blue mb-6">
            Nos Services
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Une gamme complète de services bancaires adaptés à vos besoins, 
            avec la technologie la plus avancée du secteur financier tunisien.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {services.map((service, index) => (
            <Card 
              key={service.title} 
              className="hover-lift cursor-pointer animate-slide-up border-0 shadow-lg"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardContent className="p-8 text-center h-full flex flex-col">
                <div className="text-4xl mb-4">{service.icon}</div>
                <h3 className="text-xl font-bold text-stb-blue mb-3">{service.title}</h3>
                <p className="text-gray-600 mb-6 flex-grow">{service.description}</p>
                
                <ul className="text-sm text-gray-500 mb-6 space-y-2">
                  {service.features.map((feature) => (
                    <li key={feature} className="flex items-center justify-center">
                      <span className="w-2 h-2 bg-stb-blue rounded-full mr-2"></span>
                      {feature}
                    </li>
                  ))}
                </ul>
                
                <Button 
                  variant="outline" 
                  className="w-full border-stb-blue text-stb-blue hover:bg-stb-blue hover:text-white"
                >
                  En savoir plus
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="text-center mt-12">
          <Button 
            size="lg" 
            className="gradient-stb text-white hover:opacity-90 px-8 py-4"
          >
            Voir tous nos services
          </Button>
        </div>
      </div>
    </section>
  );
};

export default ServicesSection;
