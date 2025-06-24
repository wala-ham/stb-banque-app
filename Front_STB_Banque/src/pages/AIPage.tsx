import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { useEffect, useState } from "react";
import clusteringClientsImg_3 from "@/assets/ai/clustering-clients.png";
import clusteringClientsImg_4 from "@/assets/ai/clustering-clients4.png";
import clusteringClientsImg_5 from "@/assets/ai/clustering-clients5.png";
import clusteringClientsImg_6 from "@/assets/ai/clustering-clients6.png";
import clusteringClientsImg_7 from "@/assets/ai/clustering-clients7.png";
import clusteringFournisseursImg_3 from "@/assets/ai/clustering-fournisseurs3.png";
import clusteringFournisseursImg_4 from "@/assets/ai/clustering-fournisseurs4.png";
import clusteringFournisseursImg_5 from "@/assets/ai/clustering-fournisseurs5.png";
import clusteringFournisseursImg_6 from "@/assets/ai/clustering-fournisseurs6.png";
import clusteringFournisseursImg_7 from "@/assets/ai/clustering-fournisseurs7.png";

// import clusteringFournisseursImg from "@/assets/ai/clustering-fournisseurs.png";
import anomalyClientsPcaImg from "@/assets/ai/anomaly-clients-pca.png";
import anomalyFournisseursPcaImg from "@/assets/ai/anomaly-fournisseurs-pca.png";
import anomalyClientsFragmentationImg from "@/assets/ai/anomaly-clients-fragmentation.png";

import figure1 from "@/assets/shared/Figure_1.png";
import figure2 from "@/assets/shared/Figure_2.png";
import figure3 from "@/assets/shared/Figure_3.png";
import figure4 from "@/assets/shared/Figure_4.png";
import figure5 from "@/assets/shared/Figure_5.png";
import figure6 from "@/assets/shared/Figure_6.png";
import figure7 from "@/assets/shared/Figure_7.png";
import figure8 from "@/assets/shared/Figure_8.png";
import figure9 from "@/assets/shared/Figure_9.png";
import figure10 from "@/assets/shared/Figure_10.png";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Cell,
  PieChart,
  Pie,
} from "recharts";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const AIPage = () => {
  const [selectedCluster, setSelectedCluster] = useState("all");

  // Sample clustering data
  const clusterData = [
    {
      x: 2987.45,
      y: 42.3,
      cluster: "Entrepreneurs",
      color: "#003366",
      size: 74,
    },
    { x: 1234.67, y: 22.8, cluster: "Étudiants", color: "#00CC66", size: 58 },
    { x: 1892.33, y: 67.5, cluster: "Retraités", color: "#DAA520", size: 62 },
    { x: 2345.88, y: 35.7, cluster: "Salariés", color: "#CC0066", size: 104 },
  ];
  const anomalyData = [
    { time: "00:00", normal: 1200, anomalies: 2 },
    { time: "04:00", normal: 800, anomalies: 1 },
    { time: "08:00", normal: 2500, anomalies: 3 },
    { time: "12:00", normal: 3200, anomalies: 15 },
    { time: "16:00", normal: 2800, anomalies: 8 },
    { time: "20:00", normal: 1800, anomalies: 4 },
  ];
  const [selectedK, setSelectedK] = useState(3);
  const kMeansImages = {
    3: clusteringClientsImg_3,
    4: clusteringClientsImg_4,
    5: clusteringClientsImg_5,
    6: clusteringClientsImg_6,
    7: clusteringClientsImg_7,
  };
  const kMeansFournisseursImages = {
    3: clusteringFournisseursImg_3,
    4: clusteringFournisseursImg_4,
    5: clusteringFournisseursImg_5,
    6: clusteringFournisseursImg_6,
    7: clusteringFournisseursImg_7,
  };

  const riskScores = [
    { category: "Très Faible", value: 65, color: "#00CC66" },
    { category: "Faible", value: 25, color: "#66CC00" },
    { category: "Moyen", value: 8, color: "#DAA520" },
    { category: "Élevé", value: 2, color: "#CC6600" },
  ];

  const recentAnomalies = [
    {
      id: 1,
      type: "Transaction suspecte",
      amount: "15,000 DT",
      account: "****2847",
      risk: "Élevé",
      time: "14:32",
    },
    {
      id: 2,
      type: "Connexion inhabituelle",
      amount: "-",
      account: "****1923",
      risk: "Moyen",
      time: "13:45",
    },
    {
      id: 3,
      type: "Montant anormal",
      amount: "50,000 DT",
      account: "****5629",
      risk: "Élevé",
      time: "12:18",
    },
    {
      id: 4,
      type: "Géolocalisation",
      amount: "2,500 DT",
      account: "****7384",
      risk: "Faible",
      time: "11:22",
    },
  ];

  const predictions = [
    {
      model: "Churn Prediction",
      accuracy: 94.2,
      lastUpdate: "2025-06-05 09:30",
      status: "Active",
    },
    {
      model: "Credit Scoring",
      accuracy: 91.8,
      lastUpdate: "2025-06-04 08:15",
      status: "Active",
    },
    {
      model: "Fraud Detection",
      accuracy: 97.5,
      lastUpdate: "2025-06-04 10:00",
      status: "Active",
    },
  ];
  const [kClients, setKClients] = useState(3);
  const [kFournisseurs, setKFournisseurs] = useState(3);
  const [clientClusterImg, setClientClusterImg] = useState<string | null>(null);
  const [fournisseurClusterImg, setFournisseurClusterImg] = useState<
    string | null
  >(null);

  useEffect(() => {
    fetch(`http://127.0.0.1:5001/ai/clustering/clients-image?k=${kClients}`)
      .then((res) => res.blob())
      .then((blob) => setClientClusterImg(URL.createObjectURL(blob)));
  }, [kClients]);

  useEffect(() => {
    fetch(
      `http://127.0.0.1:5001/ai/clustering/fournisseurs-image?k=${kFournisseurs}`
    )
      .then((res) => res.blob())
      .then((blob) => setFournisseurClusterImg(URL.createObjectURL(blob)));
  }, [kFournisseurs]);

  //--------------------------------------

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white">
      <Header />

      <div className="container mx-auto px-6 py-8">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-stb-blue mb-4">
            Intelligence Artificielle & Machine Learning
          </h1>
          <p className="text-xl text-gray-600">
            Analyse avancée et insights prédictifs pour optimiser vos opérations
            bancaires
          </p>
        </div>

        {/* AI Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="hover-lift border-green-200 bg-green-50">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-700">
                    Modèles Actifs
                  </p>
                  <p className="text-3xl font-bold text-green-800">12</p>
                  <p className="text-sm text-green-600">Performance optimale</p>
                </div>
                <div className="text-4xl">🤖</div>
              </div>
            </CardContent>
          </Card>

          <Card className="hover-lift border-blue-200 bg-blue-50">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-700">
                    Précision Moyenne
                  </p>
                  <p className="text-3xl font-bold text-blue-800">93.2%</p>
                  <p className="text-sm text-blue-600">+2.1% ce mois</p>
                </div>
                <div className="text-4xl">🎯</div>
              </div>
            </CardContent>
          </Card>

          <Card className="hover-lift border-orange-200 bg-orange-50">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-orange-700">
                    Anomalies Détectées
                  </p>
                  <p className="text-3xl font-bold text-orange-800">47</p>
                  <p className="text-sm text-orange-600">Dernières 24h</p>
                </div>
                <div className="text-4xl">⚠️</div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main AI Content */}
        <Tabs defaultValue="clustering" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="clustering">Clustering</TabsTrigger>
            <TabsTrigger value="anomalies">Détection d'Anomalies</TabsTrigger>
            {/* <TabsTrigger value="prediction">Modèles Prédictifs</TabsTrigger> */}
            {/* <TabsTrigger value="insights">Insights IA</TabsTrigger> */}
          </TabsList>

          {/* Clustering Tab */}
          <TabsContent value="clustering" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
              <div className="lg:col-span-2 space-y-5">
                <Card>
                  <CardHeader>
                    <CardTitle>Clustering K-Means Clients</CardTitle>
                    <p className="text-sm text-gray-600">
                      Analyse K-Means sur les clients
                    </p>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-8">
                      <div>
                        <h4 className="font-semibold mb-2">
                          Dendrogramme - Clustering Hiérarchique des Clients
                        </h4>
                        <img
                          src={figure1}
                          alt="Silhouette Scores Clients"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div>
                      {/* <div>
                        <h4 className="font-semibold mb-2">
                          Visualisation des Clusters Hiérarchiques des Clients
                          (PCA 2D)
                        </h4>
                        <img
                          src={figure2}
                          alt="Silhouette Scores Clients"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div> */}
                      <div>
                        <h4 className="font-semibold mb-2">
                          Analyse du Nombre de Clusters (Méthode du Coude &
                          Silhouette) pour les Clients
                        </h4>
                        <img
                          src={figure3}
                          alt="Clusters Hiérarchiques Clients"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">
                          Visualisation des Clusters K-Means des Clients (PCA
                          2D)
                        </h4>
                        <img
                          src={figure4}
                          alt="Clusters K-Means Clients"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">
                          Visualisation des Clusters K-Means des Clients (PCA
                          3D)
                        </h4>
                        <img
                          src={figure5}
                          alt="Clusters Features Clients"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Clustering K-Means Fournisseurs</CardTitle>
                    <p className="text-sm text-gray-600">
                      Analyse K-Means sur les fournisseurs
                    </p>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-8">
                      <div>
                        <h4 className="font-semibold mb-2">
                          Dendrogramme - Clustering Hiérarchique des
                          Fournisseurs
                        </h4>
                        <img
                          src={figure6}
                          alt="Variance expliquée Fournisseurs"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div>
                      {/* <div>
                        <h4 className="font-semibold mb-2">
                          Visualisation des Clusters Hiérarchiques des
                          Fournisseurs (PCA 2D)
                        </h4>
                        <img
                          src={figure7}
                          alt="Silhouette Scores Fournisseurs"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div> */}
                      <div>
                        <h4 className="font-semibold mb-2">
                          Analyse du Nombre de Clusters (Méthode du Coude &
                          Silhouette) pour les Fournisseurs
                        </h4>
                        <img
                          src={figure8}
                          alt="Clusters Hiérarchiques Fournisseurs"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">
                          Visualisation des Clusters K-Means des Fournisseurs
                          (PCA 2D)
                        </h4>
                        <img
                          src={figure9}
                          alt="Clusters K-Means Fournisseurs"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">
                          Visualisation des Clusters K-Means des Fournisseurs
                          (PCA 3D)
                        </h4>
                        <img
                          src={figure10}
                          alt="Clusters Features Fournisseurs"
                          className="w-full max-w-xl h-auto mx-auto"
                          style={{ maxHeight: 400 }}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>
                      Segmentation des Clients selon leurs Profils Financiers et
                      Comportementaux
                    </CardTitle>
                    <p className="text-sm text-gray-600">
                      Analyse comparative des caractéristiques moyennes par
                      cluster via (K-Means)
                    </p>
                  </CardHeader>
                  <CardContent className="p-0">
                    <img
                      src="src/assets/ai/clusters.png"
                      alt="Segmentation des clients"
                      className="block w-full h-auto"
                      style={{ maxHeight: "100vh", objectFit: "contain" }}
                    />
                  </CardContent>
                </Card>
              </div>

              <div>
                <Card>
                  <CardHeader>
                    <CardTitle>Profils des Clusters</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {clusterData.map((cluster, index) => (
                      <div
                        key={index}
                        className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer"
                      >
                        <div className="flex items-center space-x-3 mb-2">
                          <div
                            className="w-4 h-4 rounded-full"
                            style={{ backgroundColor: cluster.color }}
                          ></div>
                          <h4 className="font-semibold">{cluster.cluster}</h4>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">
                          {cluster.size} clients
                        </p>
                        <div className="text-xs space-y-1">
                          <p>Revenus moyen: {cluster.x.toLocaleString()} DT</p>
                          <p>Âge moyen: {cluster.y} ans</p>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Anomalies Tab */}
          <TabsContent value="anomalies" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
              <Card>
                <CardHeader>
                  <CardTitle>Anomalies Clients (PCA)</CardTitle>
                </CardHeader>
                <CardContent>
                  <img
                    src={anomalyClientsPcaImg}
                    alt="Anomalies Clients PCA"
                    className="rounded shadow w-full"
                    style={{ maxHeight: 320 }}
                  />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Anomalies Fournisseurs (PCA)</CardTitle>
                </CardHeader>
                <CardContent>
                  <img
                    src={anomalyFournisseursPcaImg}
                    alt="Anomalies Fournisseurs PCA"
                    className="rounded shadow w-full"
                    style={{ maxHeight: 320 }}
                  />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Anomalies Clients (Fragmentation)</CardTitle>
                </CardHeader>
                <CardContent>
                  <img
                    src={anomalyClientsFragmentationImg}
                    alt="Anomalies Clients Fragmentation"
                    className="rounded shadow w-full"
                    style={{ maxHeight: 320 }}
                  />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Prediction Tab */}
          <TabsContent value="prediction" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>État des Modèles Prédictifs</CardTitle>
                <p className="text-sm text-gray-600">
                  Performance et statut en temps réel
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {predictions.map((model, index) => (
                    <div key={index} className="p-6 border rounded-lg">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold">{model.model}</h3>
                        <Badge
                          variant={
                            model.status === "Active" ? "default" : "secondary"
                          }
                        >
                          {model.status}
                        </Badge>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <p className="text-sm text-gray-600 mb-1">
                            Précision
                          </p>
                          <div className="flex items-center space-x-2">
                            <Progress
                              value={model.accuracy}
                              className="flex-1"
                            />
                            <span className="text-sm font-medium">
                              {model.accuracy}%
                            </span>
                          </div>
                        </div>

                        <div>
                          <p className="text-sm text-gray-600 mb-1">
                            Dernière mise à jour
                          </p>
                          <p className="text-sm font-medium">
                            {model.lastUpdate}
                          </p>
                        </div>

                        <div className="flex justify-end">
                          <Button variant="outline" size="sm">
                            Configurer
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Insights Tab */}
          <TabsContent value="insights" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Recommandations IA</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
                    <h4 className="font-semibold text-blue-800">
                      Optimisation des Produits
                    </h4>
                    <p className="text-sm text-blue-700 mt-1">
                      Le segment "Jeunes Professionnels" montre un fort intérêt
                      pour les produits d'épargne digitaux. Recommandation:
                      Lancer une campagne ciblée avec 15% d'augmentation de
                      conversion prévue.
                    </p>
                  </div>

                  <div className="p-4 bg-green-50 border-l-4 border-green-500 rounded">
                    <h4 className="font-semibold text-green-800">
                      Réduction des Risques
                    </h4>
                    <p className="text-sm text-green-700 mt-1">
                      Les modèles prédictifs indiquent une réduction de 23% des
                      fraudes grâce aux nouveaux algorithmes. Économies
                      estimées: 2.3M DT annuels.
                    </p>
                  </div>

                  <div className="p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded">
                    <h4 className="font-semibold text-yellow-800">
                      Prévention Churn
                    </h4>
                    <p className="text-sm text-yellow-700 mt-1">
                      147 clients à risque élevé de départ identifiés. Actions
                      de rétention recommandées avec 67% de chance de succès.
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Métriques d'Impact</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">
                        ROI des Modèles IA
                      </span>
                      <span className="text-lg font-bold text-green-600">
                        92%
                      </span>
                    </div>
                    <Progress value={85} className="h-2" />
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">
                        Réduction des Coûts
                      </span>
                      <span className="text-lg font-bold text-blue-600">
                        18.5M DT
                      </span>
                    </div>
                    <Progress value={92} className="h-2" />
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">
                        Augmentation Revenus
                      </span>
                      <span className="text-lg font-bold text-purple-600">
                        12.8%
                      </span>
                    </div>
                    <Progress value={78} className="h-2" />
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">
                        Satisfaction Client
                      </span>
                      <span className="text-lg font-bold text-stb-blue">
                        96.2%
                      </span>
                    </div>
                    <Progress value={96} className="h-2" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      <Footer />
    </div>
  );
};

export default AIPage;
