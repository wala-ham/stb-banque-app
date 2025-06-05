import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import totalMontantImg from "@/assets/cheques/total-montant1.png";
import montantChequeImg from "@/assets/cheques/montant1.png";
import nombreImg from "@/assets/cheques/nombre1.png";
import totalMontantFournisseurImg from "@/assets/cheques/total-montant-fournisseur.png";
import montantFournisseurImg from "@/assets/cheques/montant-fournisseur.png";
import nombreFournisseurImg from "@/assets/cheques/nombre-fournisseur.png";
import timeSeriesFournisseurImg from "@/assets/cheques/time-serie.png";
import timeSeriesClientImg from "@/assets/cheques/time-series-client.png";
import categorieSegmentImg from "@/assets/cheques/categorie-segment.png";
import categorieTrancheAgeImg from "@/assets/cheques/categorie-tranche-age.png";
import categorieSexeImg from "@/assets/cheques/categorie-sexe.png";
import categorieStatutCivilImg from "@/assets/cheques/categorie-Statut_Civil.png";
import categorieSituationContractuelleImg from "@/assets/cheques/categorie-Situation_Contractuelle.png";
import categorieCiviliteImg from "@/assets/cheques/categorie-civilite.png";
import categorieActiviteEconomiqueFournisseurImg from "@/assets/cheques/categorie-activite-economique-fournisseur.png";
import categorieSegmentFournisseurImg from "@/assets/cheques/categorie-segment-fournisseur.png";
import categorieSexeFournisseurImg from "@/assets/cheques/categorie-sexe-fournisseur.png";
import fragmentationSegmentClientImg from "@/assets/performance/fragmentation-segment-clien.png";
import fragmentationActiviteFournisseurImg from "@/assets/performance/fragmentation-activite-fournisseur.png";
import categorieStatutCivilFournisseurImg from "@/assets/cheques/categorie-statut-civil-fournisseur.png";
const Dashboard = () => {
  // Sample data for charts
  const clientData = [
    { month: "Jan", particuliers: 15245, entreprises: 11538, total: 26783 },
    { month: "Fév", particuliers: 9938, entreprises: 2450, total: 22388 },
    { month: "Mar", particuliers: 2152, entreprises: 5099, total: 6452 },
    { month: "Avr", particuliers: 9278, entreprises: 3150, total: 12428 },
    { month: "Mai", particuliers: 11323, entreprises: 15847, total: 27170 },
    { month: "Jun", particuliers: 13525, entreprises: 17581, total: 31106 },
  ];
  const categoryImages: { [key: string]: string } = {
    Segment: categorieSegmentImg,
    Tranche_Age: categorieTrancheAgeImg,
    Sexe: categorieSexeImg,
    Statut_Civil: categorieStatutCivilImg,
    Situation_Contractuelle: categorieSituationContractuelleImg,
    Civilite: categorieCiviliteImg,
  };
  const fournisseurCategoryImages: { [key: string]: string } = {
    Activite_Economique: categorieActiviteEconomiqueFournisseurImg,
    Segment: categorieSegmentFournisseurImg,
    Sexe: categorieSexeFournisseurImg,
    Statut_Civil: categorieStatutCivilFournisseurImg,
  };
  const performanceData = [
    { category: "Dépôts", value: 2.8, target: 3.2 },
    { category: "Crédits", value: 4.2, target: 4.0 },
    { category: "Cartes", value: 15.5, target: 16.0 },
    { category: "Services", value: 8.7, target: 9.0 },
  ];

  const segmentData = [
    { name: "Premium", value: 25, color: "#003366" },
    { name: "Gold", value: 35, color: "#0066CC" },
    { name: "Standard", value: 40, color: "#DAA520" },
  ];

  const supplierData = [
    {
      name: "Commerce de détail",
      cost: 37827.174,
      performance: 85,
      contracts: 7,
    },
    {
      name: "Commerce de gros",
      cost: 278683.022,
      performance: 92,
      contracts: 15,
    },
    {
      name: "Fabrication et industrie",
      cost: 86054.745,
      performance: 88,
      contracts: 10,
    },
    {
      name: "Services financiers et assurances",
      cost: 11036.514,
      performance: 90,
      contracts: 5,
    },
    {
      name: "Services professionnels et techniques",
      cost: 3939.7,
      performance: 80,
      contracts: 3,
    },
    { name: "Construction", cost: 154500, performance: 86, contracts: 6 },
    { name: "Éducation", cost: 10657.8, performance: 93, contracts: 4 },
    {
      name: "Services personnels et loisirs",
      cost: 91512.683,
      performance: 83,
      contracts: 8,
    },
    {
      name: "Activités liées à l'énergie",
      cost: 67983.69,
      performance: 89,
      contracts: 6,
    },
    {
      name: "Tourisme et hôtellerie",
      cost: 37865,
      performance: 94,
      contracts: 5,
    },
    { name: "Autres services", cost: 9090, performance: 81, contracts: 2 },
    {
      name: "Agriculture et élevage",
      cost: 3600,
      performance: 79,
      contracts: 2,
    },
    { name: "Transport", cost: 45000, performance: 85, contracts: 10 },
  ];
  // Ajout consommation d'api pour Fournisseur
  const [fournisseurImages, setFournisseurImages] = useState<{
    [key: string]: string;
  }>({});
  const [selectedFournisseurCategory, setSelectedFournisseurCategory] =
    useState("Activite_Economique");
  const [fournisseurCategoryImg, setFournisseurCategoryImg] = useState<
    string | null
  >(null);
  const [fournisseurTimeSeriesImg, setFournisseurTimeSeriesImg] = useState<
    string | null
  >(null);

  const FOURNISSEUR_CATEGORICAL_COLUMNS = [
    { label: "Activité Économique", value: "Activite_Economique" },
    { label: "Segment", value: "Segment" },
    { label: "Sexe", value: "Sexe" },
    { label: "Statut Civil", value: "Statut_Civil" },
  ];

  useEffect(() => {
    const endpoints = {
      totalMontant:
        "/fournisseurs/exploration/total-montant-cheque-distribution-image",
      montantCheque:
        "/fournisseurs/exploration/montant-cheque-distribution-image",
      nombre: "/fournisseurs/exploration/nombre-distribution-image",
    };
    Object.entries(endpoints).forEach(([key, url]) => {
      fetch(`http://127.0.0.1:8000${url}`)
        .then((res) => res.blob())
        .then((blob) =>
          setFournisseurImages((imgs) => ({
            ...imgs,
            [key]: URL.createObjectURL(blob),
          }))
        );
    });
  }, []);

  useEffect(() => {
    if (!selectedFournisseurCategory) return;
    setFournisseurCategoryImg(null);
    fetch(
      `http://127.0.0.1:8000/fournisseurs/exploration/categorical-distribution-image?column=${selectedFournisseurCategory}`
    )
      .then((res) => res.blob())
      .then((blob) => setFournisseurCategoryImg(URL.createObjectURL(blob)));
  }, [selectedFournisseurCategory]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/fournisseurs/exploration/time-series-image")
      .then((res) => res.blob())
      .then((blob) => setFournisseurTimeSeriesImg(URL.createObjectURL(blob)));
  }, []);
  // --- Ajout consommation API Python pour Analyse Client ---
  const [overview, setOverview] = useState<any>(null);
  const [images, setImages] = useState<{ [key: string]: string }>({});

  useEffect(() => {
    // Aperçu des données
    fetch("http://127.0.0.1:8000/data-overview")
      .then((res) => res.json())
      .then(setOverview);

    // Images à charger
    const endpoints = {
      totalMontant:
        "/clients/exploration/total-montant-cheque-distribution-image",
      montantCheque: "/clients/exploration/montant-cheque-distribution-image",
      nombre: "/clients/exploration/nombre-distribution-image",
      segment:
        "/clients/exploration/categorical-distribution-image?column=Segment",
      timeSeries: "/clients/exploration/time-series-image",
    };

    Object.entries(endpoints).forEach(([key, url]) => {
      fetch(`http://127.0.0.1:8000${url}`)
        .then((res) => res.blob())
        .then((blob) =>
          setImages((imgs) => ({
            ...imgs,
            [key]: URL.createObjectURL(blob),
          }))
        );
    });
  }, []);

  const CATEGORICAL_COLUMNS = [
    { label: "Segment", value: "Segment" },
    { label: "Tranche d'âge", value: "Tranche_Age" },
    { label: "Sexe", value: "Sexe" },
    { label: "Statut Civil", value: "Statut_Civil" },
    { label: "Situation Contractuelle", value: "Situation_Contractuelle" },
    { label: "Civilité", value: "Civilite" },
  ];

  // ...dans le composant Dashboard...
  const [selectedCategory, setSelectedCategory] = useState("Segment");
  const [categoryImg, setCategoryImg] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedCategory) return;
    setCategoryImg(null); // reset while loading
    fetch(
      `http://127.0.0.1:8000/clients/exploration/categorical-distribution-image?column=${selectedCategory}`
    )
      .then((res) => res.blob())
      .then((blob) => setCategoryImg(URL.createObjectURL(blob)));
  }, [selectedCategory]);

  const [timeSeriesImg, setTimeSeriesImg] = useState<string | null>(null);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/clients/exploration/time-series-image")
      .then((res) => res.blob())
      .then((blob) => setTimeSeriesImg(URL.createObjectURL(blob)));
  }, []);

  const [clientFragmentationImg, setClientFragmentationImg] = useState<
    string | null
  >(null);
  const [supplierFragmentationImg, setSupplierFragmentationImg] = useState<
    string | null
  >(null);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/clients/fragmentation/score-by-segment-image")
      .then((res) => res.blob())
      .then((blob) => setClientFragmentationImg(URL.createObjectURL(blob)));
    fetch(
      "http://127.0.0.1:8000/fournisseurs/fragmentation/score-by-activity-image"
    )
      .then((res) => res.blob())
      .then((blob) => setSupplierFragmentationImg(URL.createObjectURL(blob)));
  }, []);
  // --------------------

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white">
      <Header />

      <div className="container mx-auto px-6 py-8">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-stb-blue mb-4">
            Dashboard Business Intelligence
          </h1>
          <p className="text-xl text-gray-600">
            Analyse en temps réel des performances et insights stratégiques
          </p>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4 mb-8">
          <Select defaultValue="2024">
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Année" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="2024">2024</SelectItem>
              <SelectItem value="2023">2023</SelectItem>
              <SelectItem value="2022">2022</SelectItem>
            </SelectContent>
          </Select>

          <Select defaultValue="all">
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Région" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Toutes les régions</SelectItem>
              <SelectItem value="tunis">Grand Tunis</SelectItem>
              <SelectItem value="sfax">Sfax</SelectItem>
              <SelectItem value="sousse">Sousse</SelectItem>
            </SelectContent>
          </Select>

          <Button className="gradient-stb text-white">Exporter Rapport</Button>
        </div>

        {/* KPIs Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="hover-lift">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">
                    Clients Total
                  </p>
                  <p className="text-3xl font-bold text-stb-blue">1.5M</p>
                  <p className="text-sm text-green-600">+2.3% ce mois</p>
                </div>
                <div className="text-4xl">👥</div>
              </div>
            </CardContent>
          </Card>

          <Card className="hover-lift">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">
                    Chiffre d'Affaires
                  </p>
                  <p className="text-3xl font-bold text-stb-blue">745M DT</p>
                  <p className="text-sm text-green-600">+7.8% cette année</p>
                </div>
                <div className="text-4xl">💰</div>
              </div>
            </CardContent>
          </Card>

          <Card className="hover-lift">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">
                    Satisfaction
                  </p>
                  <p className="text-3xl font-bold text-stb-blue">96%</p>
                  <p className="text-sm text-green-600">+6.2% ce mois</p>
                </div>
                <div className="text-4xl">⭐</div>
              </div>
            </CardContent>
          </Card>

          <Card className="hover-lift">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">
                    Agences Actives
                  </p>
                  <p className="text-3xl font-bold text-stb-blue">154</p>
                  <p className="text-sm text-blue-600">+3 nouvelles</p>
                </div>
                <div className="text-4xl">🏢</div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard Content */}
        <Tabs defaultValue="clients" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="clients">Analyse Clients</TabsTrigger>
            <TabsTrigger value="suppliers">Fournisseurs</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
          </TabsList>

          {/* Clients Tab */}
          <TabsContent value="clients" className="space-y-6">
            {/* Rapport Power BI - Clients en haut, prend toute la largeur */}
            <Card>
              <CardHeader>
                <CardTitle>Rapport Power BI - Clients</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="w-full flex justify-center">
                  <iframe
                    title="client_report"
                    width="100%"
                    height="450"
                    src="https://app.powerbi.com/view?r=eyJrIjoiMDAwZjQ1NWQtZTFjNC00MmQ5LThkODMtZDU3NDE3OGUzOTA1IiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9"
                    frameBorder="0"
                    allowFullScreen={true}
                    className="rounded-lg shadow w-full"
                  ></iframe>
                </div>
              </CardContent>
            </Card>

            {/* Évolution du Portefeuille Client en dessous */}
            <Card>
              <CardHeader>
                <CardTitle>Analyse des Transactions par Chèque</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={clientData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="particuliers"
                      stroke="#003366"
                      strokeWidth={3}
                    />
                    <Line
                      type="monotone"
                      dataKey="entreprises"
                      stroke="#0066CC"
                      strokeWidth={3}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* --- Analyse Client : Consommation API Python --- */}
            <Card>
              <CardHeader>
                <CardTitle>Exploration des Clients (Python)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <p className="font-semibold mb-2">
                      Distribution Total Montant Chèque
                    </p>
                    <img
                      src={totalMontantImg}
                      alt="Distribution Total Montant Chèque"
                      className="rounded shadow w-full"
                    />
                  </div>
                  <div>
                    <p className="font-semibold mb-2">
                      Distribution Montant Chèque
                    </p>
                    <img
                      src={montantChequeImg}
                      alt="Distribution Montant Chèque"
                      className="rounded shadow w-full"
                    />
                  </div>
                  <div>
                    <p className="font-semibold mb-2">
                      Distribution Nombre de Chèques
                    </p>
                    <img
                      src={nombreImg}
                      alt="Distribution Nombre de Chèques"
                      className="rounded shadow w-full"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Distribution Catégorielle des Clients</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-4 w-64">
                  <Select
                    value={selectedCategory}
                    onValueChange={setSelectedCategory}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Choisir une catégorie" />
                    </SelectTrigger>
                    <SelectContent>
                      {CATEGORICAL_COLUMNS.map((col) => (
                        <SelectItem key={col.value} value={col.value}>
                          {col.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <img
                    src={categoryImages[selectedCategory]}
                    alt={`Distribution ${selectedCategory}`}
                    className="rounded shadow w-full max-w-2xl"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Analyse Temporelle des Clients</CardTitle>
              </CardHeader>
              <CardContent>
                <img
                  src={timeSeriesClientImg}
                  alt="Analyse Temporelle des Clients"
                  className="rounded shadow w-full max-w-3xl mx-auto"
                />
              </CardContent>
            </Card>
            {/* --- Fin Analyse Client API Python --- */}
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Score de Fragmentation par Segment Client</CardTitle>
              </CardHeader>
              <CardContent>
                <img
                  src={fragmentationSegmentClientImg}
                  alt="Score de Fragmentation par Segment Client"
                  className="rounded shadow w-full max-w-3xl mx-auto"
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>
                  Score de Fragmentation par Activité Économique (Top 10
                  Fournisseurs)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <img
                  src={fragmentationActiviteFournisseurImg}
                  alt="Score de Fragmentation par Activité Économique Fournisseurs"
                  className="rounded shadow w-full max-w-3xl mx-auto"
                />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Indicateurs de Performance (KPIs)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#003366" name="Réalisé" />
                    <Bar dataKey="target" fill="#DAA520" name="Objectif" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Suppliers Tab */}
          <TabsContent value="suppliers" className="space-y-6">
            {/* Rapport Power BI - Fournisseurs */}
            <Card>
              <CardHeader>
                <CardTitle>Rapport Power BI - Fournisseurs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="w-full flex justify-center">
                  <iframe
                    title="fournisseur_report"
                    width="100%"
                    height="450"
                    src="https://app.powerbi.com/view?r=eyJrIjoiY2I2ZjQyNDItZDgxOC00YmM5LThhZjMtZGI0ODE1NzFhMWQ2IiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9"
                    frameBorder="0"
                    allowFullScreen={true}
                    className="rounded-lg shadow w-full"
                  ></iframe>
                </div>
              </CardContent>
            </Card>
            {/* Analyse des Fournisseurs */}
            <Card>
              <CardHeader>
                <CardTitle>Analyse des Fournisseurs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-4">Fournisseur</th>
                        <th className="text-left p-4">Coût Annuel</th>
                        <th className="text-left p-4">Performance</th>
                        <th className="text-left p-4">Contrats</th>
                        <th className="text-left p-4">Statut</th>
                      </tr>
                    </thead>
                    <tbody>
                      {supplierData.slice(0, 5).map((supplier) => (
                        <tr
                          key={supplier.name}
                          className="border-b hover:bg-gray-50"
                        >
                          <td className="p-4 font-medium">{supplier.name}</td>
                          <td className="p-4">
                            {supplier.cost.toLocaleString()} DT
                          </td>
                          <td className="p-4">
                            <div className="flex items-center space-x-2">
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-stb-blue h-2 rounded-full"
                                  style={{ width: `${supplier.performance}%` }}
                                ></div>
                              </div>
                              <span className="text-sm">
                                {supplier.performance}%
                              </span>
                            </div>
                          </td>
                          <td className="p-4">{supplier.contracts}</td>
                          <td className="p-4">
                            <span
                              className={`px-3 py-1 rounded-full text-sm ${
                                supplier.performance >= 90
                                  ? "bg-green-100 text-green-800"
                                  : "bg-yellow-100 text-yellow-800"
                              }`}
                            >
                              {supplier.performance >= 90 ? "Excellent" : "Bon"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
            {/* --- Analyse Fournisseur : Consommation API Python --- */}
            <Card>
              <CardHeader>
                <CardTitle>Exploration des Fournisseurs (Python)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <p className="font-semibold mb-2">
                      Distribution Total Montant Chèque Fournisseurs
                    </p>
                    <img
                      src={totalMontantFournisseurImg}
                      alt="Distribution Total Montant Chèque Fournisseurs"
                      className="rounded shadow w-full"
                    />
                  </div>
                  <div>
                    <p className="font-semibold mb-2">
                      Distribution Montant Chèque Fournisseurs
                    </p>
                    <img
                      src={montantFournisseurImg}
                      alt="Distribution Montant Chèque Fournisseurs"
                      className="rounded shadow w-full"
                    />
                  </div>
                  <div>
                    <p className="font-semibold mb-2">
                      Distribution Nombre de Chèques Fournisseurs
                    </p>
                    <img
                      src={nombreFournisseurImg}
                      alt="Distribution Nombre de Chèques Fournisseurs"
                      className="rounded shadow w-full"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>
                  Distribution Catégorielle des Fournisseurs
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-4 w-64">
                  <Select
                    value={selectedFournisseurCategory}
                    onValueChange={setSelectedFournisseurCategory}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Choisir une catégorie" />
                    </SelectTrigger>
                    <SelectContent>
                      {FOURNISSEUR_CATEGORICAL_COLUMNS.map((col) => (
                        <SelectItem key={col.value} value={col.value}>
                          {col.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <img
                    src={fournisseurCategoryImages[selectedFournisseurCategory]}
                    alt={`Distribution ${selectedFournisseurCategory}`}
                    className="rounded shadow w-full max-w-2xl"
                  />
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Analyse Temporelle des Fournisseurs</CardTitle>
              </CardHeader>
              <CardContent>
                <img
                  src={timeSeriesFournisseurImg}
                  alt="Analyse Temporelle des Fournisseurs"
                  className="rounded shadow w-full max-w-3xl mx-auto"
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      <Footer />
    </div>
  );
};

export default Dashboard;
