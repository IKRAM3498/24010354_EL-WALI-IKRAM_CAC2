# Compte rendu du projet de classification médicale

## Informations générales
- **Auteur :** EL-WALI IKRAM  
- **Matricule :** 24010354  
- **Date :** 10 décembre 2025  
- **Thématique :** Diagnostic du cancer du sein par classification supervisée  
- **Jeu de données :** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Lien :** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## 1. Introduction et Contexte

Le cancer du sein est l’un des cancers les plus fréquents chez les femmes dans le monde. La détection précoce est essentielle pour améliorer les chances de survie et réduire la mortalité. Les méthodes traditionnelles reposent sur des examens médicaux et des biopsies, mais l’intégration du **Machine Learning** permet d’assister les médecins en fournissant des prédictions rapides et fiables.

Ce projet s’inscrit dans cette logique : utiliser un dataset médical reconnu pour construire des modèles de classification capables de prédire si une tumeur est bénigne ou maligne. L’objectif est double :
1. **Scientifique :** comparer plusieurs modèles de classification et analyser leurs performances.  
2. **Pratique :** identifier les variables les plus influentes et proposer une approche applicable dans un contexte médical réel.

---

## 2. Description du Dataset

- **Nombre d’observations :** 569 patients  
- **Nombre de variables :** 32 (ID, diagnostic, 30 mesures)  
- **Variable cible :** `diagnosis` (B = bénin, M = malin)  
- **Caractéristiques principales :**
  - **Radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension**  
  - Déclinées en trois versions : _mean, _se (erreur standard), _worst (valeur extrême)

### Pertinence du dataset
- Données propres, sans valeurs manquantes.  
- Problème médical critique.  
- Structure adaptée aux modèles supervisés.  
- Permet une classification binaire claire.

---

## 3. Méthodologie

### Étapes principales
1. **Prétraitement :**
   - Suppression des doublons et de la colonne ID.  
   - Normalisation des variables pour stabiliser les modèles sensibles à l’échelle.  
   - Séparation en X (features) et y (target).

2. **Exploration des données (EDA) :**
   - Histogrammes et boxplots pour analyser la distribution et détecter les outliers.  
   - Heatmap de corrélations pour identifier les variables les plus liées au diagnostic.  
   - Scatterplots pour visualiser la relation entre certaines variables et la cible.

3. **Modélisation :**
   - Division en train/test (80/20).  
   - Entraînement de trois modèles : Logistic Regression, Random Forest, XGBoost.  
   - Évaluation par accuracy, precision, recall, F1-score et matrices de confusion.

---

## 4. Analyse Exploratoire des Données

### Distribution des variables
- Les variables _mean sont proches d’une distribution normale.  
- Les variables _se et _worst présentent une forte asymétrie à droite.  
- Présence d’outliers significatifs, surtout dans les mesures extrêmes.

### Corrélations
- Les variables liées à la taille et à la forme (radius, perimeter, area, concave points) sont fortement corrélées avec la malignité.  
- Multicollinéarité élevée entre radius_mean, perimeter_mean et area_mean.  
- Importance de la sélection de variables ou de la réduction de dimension (PCA).

---

## 5. Résultats des Modèles

### Logistic Regression
- **Accuracy :** 98.2 %  
- **Recall :** 97.7 %  
- **Faux négatifs :** 1  
- **Interprétation :** excellent équilibre, très fiable pour détecter les cas malins.

### Random Forest
- **Accuracy :** 96.5 %  
- **Recall :** élevé mais moins bon que LR  
- **Faux négatifs :** 3  
- **Interprétation :** robuste mais plus de cancers non détectés.

### XGBoost
- **Accuracy :** 95.6 %  
- **Recall :** élevé  
- **Faux négatifs :** 3  
- **Interprétation :** performant mais légèrement inférieur à LR sur ce split.

---

## 6. Analyse Comparative

| Critère         | Logistic Regression | Random Forest | XGBoost |
|-----------------|---------------------|---------------|---------|
| Accuracy        | 98.2 %              | 96.5 %        | 95.6 %  |
| Recall          | 97.7 %              | Élevé         | Élevé   |
| F1-Score        | Très élevé          | Élevé         | Élevé   |
| Faux négatifs   | 1                   | 3             | 3       |

**Conclusion intermédiaire :** La Régression Logistique est le modèle le plus performant sur ce split, car elle minimise les faux négatifs, ce qui est crucial en contexte médical.

---

## 7. Discussion

- **Importance des variables :** radius_mean, perimeter_mean et area_mean sont les plus influentes.  
- **Multicollinéarité :** nécessite une régularisation ou une réduction de dimension.  
- **Risque médical :** les faux négatifs sont les erreurs les plus critiques.  
- **Améliorations possibles :**
  - Validation croisée pour confirmer la robustesse.  
  - Optimisation des hyperparamètres (GridSearchCV).  
  - Ajustement du seuil de classification pour maximiser le recall.  
  - Utilisation de techniques d’ensemble (bagging, boosting).

---

## 8. Conclusion Générale

Ce projet démontre l’efficacité du Machine Learning pour assister le diagnostic médical.  
- **Meilleur modèle :** Logistic Regression, grâce à son excellent rappel et sa faible proportion de faux négatifs.  
- **Arbres de décision (RF, XGB) :** très performants mais nécessitent un tuning plus poussé.  
- **Application pratique :** un modèle fiable peut aider les médecins à détecter précocement les cancers du sein.  
- **Perspectives :**
  - Intégrer davantage de données médicales (images, antécédents).  
  - Développer des systèmes hybrides combinant plusieurs modèles.  
  - Étudier l’impact de la calibration des probabilités sur la prise de décision clinique.

---

## 9. Références
- Breast Cancer Wisconsin (Diagnostic) Dataset — Kaggle  
- Documentation Scikit-learn (classification, métriques, prétraitement)  
- Articles scientifiques sur l’application du Machine Learning en oncologie
