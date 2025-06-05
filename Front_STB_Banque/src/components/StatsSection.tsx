
import { useEffect, useState } from 'react';

const StatsSection = () => {
  const [counters, setCounters] = useState({
    clients: 0,
    agences: 0,
    experience: 0,
    satisfaction: 0
  });

  const finalStats = {
    clients: 2500000,
    agences: 120,
    experience: 65,
    satisfaction: 96
  };

  useEffect(() => {
    const duration = 2000; // 2 seconds
    const interval = 50; // Update every 50ms
    const steps = duration / interval;

    const incrementValues = {
      clients: finalStats.clients / steps,
      agences: finalStats.agences / steps,
      experience: finalStats.experience / steps,
      satisfaction: finalStats.satisfaction / steps
    };

    let currentStep = 0;

    const timer = setInterval(() => {
      if (currentStep < steps) {
        setCounters(prev => ({
          clients: Math.floor(incrementValues.clients * currentStep),
          agences: Math.floor(incrementValues.agences * currentStep),
          experience: Math.floor(incrementValues.experience * currentStep),
          satisfaction: Math.floor(incrementValues.satisfaction * currentStep)
        }));
        currentStep++;
      } else {
        setCounters(finalStats);
        clearInterval(timer);
      }
    }, interval);

    return () => clearInterval(timer);
  }, []);

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(0) + 'K';
    }
    return num.toString();
  };

  const stats = [
    {
      value: formatNumber(counters.clients),
      label: "Clients satisfaits",
      suffix: "+"
    },
    {
      value: counters.agences,
      label: "Agences en Tunisie",
      suffix: "+"
    },
    {
      value: counters.experience,
      label: "Années d'expérience",
      suffix: ""
    },
    {
      value: counters.satisfaction,
      label: "Satisfaction client",
      suffix: "%"
    }
  ];

  return (
    <section className="py-20 gradient-stb text-white">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            La STB en Chiffres
          </h2>
          <p className="text-xl opacity-90 max-w-3xl mx-auto">
            Des décennies d'expertise au service de l'économie tunisienne
          </p>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <div 
              key={stat.label} 
              className="text-center animate-scale-in"
              style={{ animationDelay: `${index * 200}ms` }}
            >
              <div className="text-5xl md:text-6xl font-bold mb-4 text-stb-gold">
                {stat.value}{stat.suffix}
              </div>
              <p className="text-lg opacity-90">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default StatsSection;
