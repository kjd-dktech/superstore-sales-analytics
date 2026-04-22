# 🚀 Superstore Profit Predictor - Microservice API

**Inference API for Financial Predictions and Business Scenarios**

## Architecture & Finalité

Le service d'inférence (Analytics & ML Backend) a été structuré comme un microservice indépendant afin de garantir la scalabilité et la sécurité des données décisionnelles :
- **Framework de livraison** : L'API est développée sous `FastAPI`, optimisée pour une gestion asynchrone intensive.
- **Continuité Analytique** : Le modèle prédictif embarqué (basé sur l'historique d'exploitation du système d'information) évalue la rentabilité sur-le-champ de scénarios transactionnels (Simulations unitaires et traitement par lots - Batching).

## Capacités Stratégiques (SLA & Sécurité)

Dans une optique de déploiement en production, l'architecture a été blindée contre divers vecteurs tout en permettant un Monitoring des consultations :

1. **Gatekeeping API Key**
   - Protection rigoureuse de toutes les requêtes (Simulations et Batching) par `X-API-KEY`. 
   - Hachage cryptographique avancé via `bcrypt` garantissant une étanchéité des accréditations en base de données de production.

2. **Mitigation Rate-Limiting & Redis**
   - Implémentation de `SlowAPI` et provision d'une file d'attente sur `Redis` afin de limiter rationnellement le traffic (ex: throttling des appels B2B massifs sur le endpoint d'inférence Batch).

3. **Administration Monitoring**
   - Panneau de gouvernance sécurisé (Admin Dashboard) pour l'attribution des clés clients. Identité de session chiffrée par symétrie (AES) avec `Fernet` via cookies signés.

## Stratégie de Déploiement

Afin d'assurer sa portabilité et sa tolérance de charge :

1. **Modèle de distribution (Hugging Face / S3)** : Le pipeline Scikit-Learn (Joblib) est hébergé indépendamment du code applicatif. À l'initialisation (Lifespan), l'API s'assure d'importer le poids fonctionnel le plus récent.
2. **Containerisation** : Image Docker structurée avec des privilèges de base afin de convenir aux contraintes des orchestrateurs cloud sécurisés (Hugging Face Spaces, Kubernetes). Optimisation interne du parallélisme du modèle via `OMP_NUM_THREADS="1"` pour minimiser les contentions CPUs au sein d'environnements hyper-threadés.

---
*L'architecture de service est prévue pour assurer une évolutivité avec l'augmentation du nombre de transactions et de catégories traitées.*

---
Kodjo Jean DEGBEVI — [LinkedIn](https://www.linkedin.com/in/kodjo-jean-degbevi-ba5170369) — [GitHub](https://github.com/kjd-dktech) — [Portfolio](https://mayal.tech)