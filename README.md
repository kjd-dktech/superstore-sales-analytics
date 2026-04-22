# 📊 Dashboard de Performance & Stratégie Superstore

**Analyse des performances commerciales et recommandations stratégiques (2014–2017)**

## Contexte et Objectifs

Ce projet a été conçu pour répondre à une problématique métier claire pour une enseigne de distribution américaine : **Optimiser la rentabilité opérationnelle** en analysant l'historique de ventes de 2014 à 2017.

L’objectif opérationnel n’est pas uniquement descriptif, mais bel et bien prescriptif. Il s'agit de fournir à la direction des recommandations actionnables pour :
- Identifier les leviers de rentabilité (régions, catégories, segments).
- Comprendre l'impact des politiques de remises (Discount) sur la Marge Nette.
- Optimiser la valeur unitaire par client.

## Principales Conclusions et Recommandations

À travers l'analyse des +9 900 transactions, mes recommandations stratégiques pour la direction répondent à 4 leviers prioritaires, basés sur les constats diagnostiques :

1. **Instauration d'un Plafond de Remise (Hard Cap)**
   - *Constat* : L'analyse du point mort (Break-Even Point) démontre que toute remise supérieure à 20% entraîne quasi systématiquement une vente à perte, particulièrement sur les produits dont la structure de coût est rigide.
   - *Recommandation* : Implémenter une règle métier stricte dans le système de vente bloquant techniquement toute remise au-delà de 20%. Les exceptions doivent être limitées aux produits à très haute marge brute (ex: Classeurs/Binders) et soumises à validation managériale.

2. **Redressement des États "Trous Noirs"**
   - *Constat* : Seuls quatre États (Texas, Ohio, Pennsylvanie, Illinois) cumulent plus de 70 000 $ de pertes nettes en raison de remises moyennes insoutenables dépassant les 30%, contre seulement 11% dans la région Ouest.
   - *Recommandation* : Aligner d'urgence la politique commerciale de ces zones sur le modèle de discipline tarifaire de la Californie ou de New York. Il est préférable de sacrifier une partie du volume de ventes au profit d'un retour immédiat à la rentabilité par la réduction drastique des promotions.

3. **Rationalisation de l'Offre Tables & Machines**
   - *Constat* : Contrairement aux fournitures de bureau, les catégories "Tables" et "Machines" ne supportent pas les stratégies de remise agressive. Elles sont actuellement les principales contributrices au déficit opérationnel.
   - *Recommandation* : Cesser d'utiliser ces catégories comme produits d'appel promotionnels. Elles doivent être repositionnées comme des produits de destination vendus à prix plein ou avec des remises symboliques (inférieures à 5%), ou être retirées du catalogue dans les régions les moins rentables.

4. **Pivot Marketing vers le Segment Home Office**
   - *Constat* :  Le segment Home Office (TPE/Indépendants) s'avère 24% plus rentable que le segment Consumer (Particuliers) pour un coût d'acquisition client (CAC) similaire. 
   - *Recommandation* : Réallouer prioritairement le budget marketing vers le segment Home Office. Concevoir des offres d'équipement technologique (catégorie à forte marge) spécifiquement packagées pour les travailleurs à domicile afin de maximiser le retour sur investissement des campagnes.

Cette proposition stratégique est soutenue par le simulateur prédictif (Random Forest Backend) annexé dans l'outil permettant de valider et anticiper manuellement toutes transactions de ce genre.

## Architecture du Projet

L'architecture est entièrement axée sur la scalabilité, dissociant le moteur analytique de l'interface de restitution :

- **Front-End Décisionnel (Streamlit)** : Interface interactive permettant l'exploration dynamique des KPI, des comportements clients et des simulations de rentabilité. <br>*DIsponible sur [mayal-superstore.streamlit.app](https://mayal-superstore.streamlit.app)*
- **Backend Prédictif (FastAPI)** : Microservice d'inférence sécurisé exposant notre modèle de Machine Learning entraîné sur les données historiques. <br>*Disponible sur [kjd-dktech-superstore-api.hf.space](https://kjd-dktech-superstore-api.hf.space)*
- **Intégration Continue / Modèle** : Le modèle de prédiction est hébergé et versionné sur Hugging Face Model Hub, garantissant des mises à jour transparentes selon l'évolution du marché.

---
*Ce dépôt présente le code source complet de la solution : de la modélisation des données brutes jusqu'à la création des services web et du reporting analytique.*

---
Kodjo Jean DEGBEVI — [LinkedIn](https://www.linkedin.com/in/kodjo-jean-degbevi-ba5170369) — [GitHub](https://github.com/kjd-dktech) — [Portfolio](https://mayal.tech)