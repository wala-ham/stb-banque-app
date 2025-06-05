
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-stb-blue text-white">
      <div className="container mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Logo and Description */}
          <div className="lg:col-span-1">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">STB</span>
              </div>
              <div>
                <h3 className="font-bold text-xl">STB</h3>
                <p className="text-sm opacity-80">Société Tunisienne de Banque</p>
              </div>
            </div>
            <p className="text-sm opacity-80 leading-relaxed">
              Votre partenaire financier de confiance depuis plus de 65 ans. 
              Innovation, sécurité et excellence au service de vos projets.
            </p>
          </div>

          {/* Services */}
          <div>
            <h4 className="font-semibold text-lg mb-4">Services</h4>
            <ul className="space-y-2 text-sm opacity-80">
              <li><Link to="/services" className="hover:text-stb-gold transition-colors">Comptes Particuliers</Link></li>
              <li><Link to="/services" className="hover:text-stb-gold transition-colors">Services Entreprises</Link></li>
              <li><Link to="/services" className="hover:text-stb-gold transition-colors">Banque Digitale</Link></li>
              <li><Link to="/services" className="hover:text-stb-gold transition-colors">Investissements</Link></li>
              <li><Link to="/services" className="hover:text-stb-gold transition-colors">Crédits</Link></li>
            </ul>
          </div>

          {/* Solutions Digitales */}
          <div>
            <h4 className="font-semibold text-lg mb-4">Solutions Digitales</h4>
            <ul className="space-y-2 text-sm opacity-80">
              <li><Link to="/dashboard" className="hover:text-stb-gold transition-colors">Dashboard BI</Link></li>
              <li><Link to="/ai" className="hover:text-stb-gold transition-colors">Intelligence Artificielle</Link></li>
              <li><a href="#" className="hover:text-stb-gold transition-colors">Application Mobile</a></li>
              <li><a href="#" className="hover:text-stb-gold transition-colors">E-Banking</a></li>
              <li><a href="#" className="hover:text-stb-gold transition-colors">API Banking</a></li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="font-semibold text-lg mb-4">Contact</h4>
            <div className="space-y-3 text-sm opacity-80">
              <div>
                <p className="font-medium">Siège Social</p>
                <p>Rue Hédi Nouira, 1001 Tunis</p>
              </div>
              <div>
                <p className="font-medium">Téléphone</p>
                <a href="tel:+21671340733" className="hover:text-stb-gold transition-colors">
                  +216 71 340 733
                </a>
              </div>
              <div>
                <p className="font-medium">Email</p>
                <a href="mailto:contact@stb.com.tn" className="hover:text-stb-gold transition-colors">
                  contact@stb.com.tn
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-white/20 mt-12 pt-8">
          <div className="flex flex-col lg:flex-row justify-between items-center space-y-4 lg:space-y-0">
            <div className="text-sm opacity-80">
              © 2024 Société Tunisienne de Banque. Tous droits réservés.
            </div>
            
            <div className="flex space-x-6 text-sm">
              <a href="#" className="hover:text-stb-gold transition-colors">Mentions Légales</a>
              <a href="#" className="hover:text-stb-gold transition-colors">Politique de Confidentialité</a>
              <a href="#" className="hover:text-stb-gold transition-colors">CGU</a>
            </div>

            <div className="flex space-x-4">
              <a href="#" className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition-colors">
                <span className="text-sm">f</span>
              </a>
              <a href="#" className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition-colors">
                <span className="text-sm">in</span>
              </a>
              <a href="#" className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition-colors">
                <span className="text-sm">@</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
