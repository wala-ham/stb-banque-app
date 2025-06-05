# 💸 Analyse Avancée des Paiements : Clients & Fournisseurs

## 🚀 Vue d’Ensemble du Projet

Cette application web interactive développée avec **Streamlit** propose une **analyse avancée** des paiements des **clients et fournisseurs**. En combinant **Business Intelligence (BI)** et **Intelligence Artificielle (IA)**, elle permet d’explorer, de détecter, de segmenter et d’interagir avec les données via un **chat intelligent**.

Objectif :  
➡️ Identifier des **schémas de paiement**, détecter les **comportements atypiques** (ex : paiements fractionnés), **segmenter les entités**, et extraire des **insights exploitables** à partir des données brutes.

---

## ✨ Fonctionnalités Clés

L’application est divisée en **deux niveaux d’analyse** :

- **Business Intelligence (BI)** : Exploration et visualisation.
- **IA Avancée** : Détection d’anomalies, clustering et LLM.
- **Chatbot Intelligent** : Interaction en langage naturel avec les données grâce à un modèle d’IA générative (Gemini Pro), permettant d’obtenir des analyses, résumés et observations directement depuis les datasets.

---

## 📊 Exploration des Données (Business Intelligence - BI)

### ▶️ Aperçu des Données

- Visualisation des premières lignes des datasets clients/fournisseurs.
- Infos générales : types de données, valeurs manquantes.

### 🧑‍💼 Clients

- **Distributions** : Montant total des chèques, Montant moyen, Nombre de chèques.
- **Répartition** : Segment, Tranche d'âge, Sexe, Statut civil, Situation contractuelle.
- **Tendances temporelles** : Évolution mensuelle/annuelle des montants.

### 🏭 Fournisseurs

- **Distributions** : Montants, fréquences.
- **Répartition** : Activité économique, segment, top 10 catégories.
- **Tendances temporelles** : Évolution mensuelle/annuelle des volumes.

### 🔍 Paiements Fractionnés

- Visualisation du **score de fragmentation**.
- Comparaison par **segment client** et **activité fournisseur**.
- Définition de **seuils de risque** basés sur les percentiles.

---

## 🧠 Capacités IA & ML

### 🛑 Détection d'Anomalies (Isolation Forest)

- Identification automatique des comportements de paiement inhabituels.
- Visualisation des anomalies sur **PCA 2D**.
- Paramètre interactif de **contamination** (% d’anomalies attendues).
- Corrélation avec les **scores de fragmentation**.

### 🎯 Clustering & Segmentation (K-Means)

- Segmentation des entités selon leurs comportements de paiement.
- Affichage des clusters sur **PCA 2D**.
- Description des **profils de segments** : fragmentation, volume, fréquence...
- Choix interactif du nombre de clusters (**k**).

### 🔗 Analyse des Relations Clients-Fournisseurs

- Détection des entités présentes des deux côtés.
- Corrélations croisées de leurs comportements.
- Visualisation via **matrices de corrélation** et **scatter plots**.

---

## 💬 Interaction avec l'IA Générative (LLM)

### 🤖 Chat avec les Données

- Interface conversationnelle via **Gemini Pro API (Google Generative AI)**.
- Posez vos questions en **langage naturel**.
- Résumés et analyses générées à partir des données fournies.
- Historique de conversation **persistant**.
- Possibilité de **télécharger une liste d’exemples de questions**.

---
