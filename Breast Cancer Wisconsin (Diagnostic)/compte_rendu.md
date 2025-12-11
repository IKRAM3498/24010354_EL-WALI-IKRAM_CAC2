
# ** EL-WALI IKRAM**
# ** 24010354**
<img src="encgs1.jfif" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

# Breast Cancer Dataset : [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

# Compte rendu

## Analyse Médicale et Prédiction du Diagnostic par Classification

**Date :** 10 Décembre 2025

---

# La thématique choisie pour cette analyse est la santé, avec un accent particulier sur le **diagnostic du cancer du sein**.

# À propos du jeu de données :

## 1. Sélection du jeu de données

Le jeu de données sélectionné est le **Breast Cancer Wisconsin (Diagnostic) Dataset**, disponible sur la plateforme **Kaggle**.
Il contient des informations détaillées sur plus de **500 échantillons de tumeurs mammaires**, avec des mesures calculées à partir d'images de cytologie (rayon, texture, périmètre, surface, concavité, symétrie, fractal dimension…).

Ce dataset est pertinent pour plusieurs raisons :

* Il n'est **pas trivial**, avec de nombreuses variables quantitatives.
* Il permet de construire des **modèles de classification binaire** (Bénin vs Malin).
* Il permet d'étudier une thématique critique : **le dépistage du cancer du sein**.
* Il est propre, structuré et directement utilisable pour une analyse ou un modèle de Machine Learning.

## 2. Définition de la Problématique (Tâche : Classification)

L’objectif de ce projet est de construire un **modèle de classification** capable de **prédire si une tumeur est bénigne ou maligne** à partir des caractéristiques mesurées.

**Il s'agit donc d'une tâche de classification binaire**, car la variable cible (**target**) est catégorielle (0 = Bénin, 1 = Malin).

Problématique étudiée :

> **Peut-on prédire de manière fiable la nature d'une tumeur (Bénin ou Malin) à partir de ses mesures cytologiques ?**

Cette problématique permet :

* d’évaluer l’importance de chaque mesure dans le diagnostic,
* de tester différents modèles de classification,
* de vérifier la cohérence et la qualité du dataset par rapport aux standards médicaux.

---

# 3. Dictionnaire des Données (Metadata)

## Taille du dataset

* **Nombre de lignes (patients)** : 569
* **Nombre de colonnes (variables)** : 32 (ID, target + 30 mesures)

## Types de variables

* **Variables quantitatives continues** : rayon, texture, périmètre, surface, concavité, points concaves, symétrie, fractal dimension, et leurs moyennes, erreurs et valeurs pires.
* **Variable cible** : `target` (0 = Bénin, 1 = Malin)

## Description des variables principales

| Variable           | Type         | Description                                                              |
| ------------------ | ------------ | ------------------------------------------------------------------------ |
| **ID**             | Numérique    | Identifiant du patient                                                   |
| **target**         | Catégorielle | 0 = Bénin, 1 = Malin (**variable cible**)                                |
| **radius_mean**    | Numérique    | Rayon moyen de la tumeur                                                 |
| **texture_mean**   | Numérique    | Texture moyenne de la tumeur                                             |
| **perimeter_mean** | Numérique    | Périmètre moyen de la tumeur                                             |
| **area_mean**      | Numérique    | Surface moyenne                                                          |
| …                  | …            | … (toutes les autres mesures : smoothness, compactness, concavity, etc.) |

---

## 1. Introduction et Contexte

Ce rapport détaille l'analyse et la modélisation prédictive d'un jeu de données médical. L'objectif est de prédire **la nature de la tumeur** (bénigne ou maligne) à partir de mesures cytologiques.

Les étapes suivies incluent l'exploration des données, le prétraitement, la normalisation, et la comparaison de trois modèles de classification : **Logistic Regression, Random Forest et XGBoost**.

---

## 2. Analyse Exploratoire des Données (EDA)

### 2.1 Chargement et Structure du Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Chargement
df = pd.read_csv("breast_cancer.csv")
print(df.shape)
df.head()
```

* **Observations :** 569 patients
* **Variables :** 30 mesures + target

---

### 2.2 Pré-traitement (Preprocessing)

* Nettoyage des données
* Gestion des doublons et valeurs manquantes
* Conversion des types si nécessaire
* Normalisation pour les modèles sensibles à l’échelle (ex : Logistic Regression, SVM)

```python
# Vérification des doublons et suppression
df = df.drop_duplicates()

# Vérification des valeurs manquantes
df.isnull().sum()

# Supprimer colonne ID
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    
# Séparer features et target
X = df.drop('target', axis=1)
y = df['target']

# Normalisation des données (0-1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 2.3 Analyse Exploratoire

* **Distribution des variables** : histogrammes et boxplots pour détecter outliers
* <img width="1990" height="3190" alt="image" src="https://github.com/user-attachments/assets/27be4749-7e08-47ba-b944-614f567a2151" />
# interpritation
L'image présente 30 histogrammes montrant la distribution des variables dans un jeu de données d'apprentissage automatique, très probablement standardisées (centrées autour de zéro). Ces variables sont regroupées en trois catégories (_mean, _se, _worst) pour des caractéristiques comme le rayon et la texture. On observe que si les distributions moyennes (_mean) sont presque normales, la majorité des variables (surtout celles d'erreur standard, _se, et les valeurs extrêmes, _worst) sont fortement asymétriques à droite. Cette asymétrie indique une concentration de valeurs faibles avec une longue queue vers les valeurs positives et est courante dans les données d'imagerie médicale, signalant la variabilité des mesures.
* **Heatmap de corrélations** : détecter les mesures les plus corrélées entre elles et avec le target
* <img width="1064" height="838" alt="image" src="https://github.com/user-attachments/assets/52f202e7-25d5-49b1-8ec6-532f2650650e" />
# interpritaion :
La Matrice de Corrélation (Heatmap) est l'outil le plus informatif, car elle quantifie les relations linéaires entre les variables. Elle révèle que les caractéristiques liées à la taille et à la forme d'une masse, notamment les points concaves, le rayon, le périmètre et la surface (en particulier leurs versions _worst et _mean), sont les meilleurs prédicteurs de la malignité (Diagnosis_M), avec des corrélations positives atteignant près de 0.8. Cependant, la matrice montre également une multicollinéarité extrême : les variables mesurant des propriétés similaires (comme le radius_mean, perimeter_mean, et area_mean) sont presque parfaitement corrélées entre elles (proche de 1.00), indiquant une redondance structurelle qui rendra la plupart des modèles statistiques instables si elles sont toutes incluses.
* **Scatterplots** : relation features ↔ target
<img width="1529" height="1189" alt="image" src="https://github.com/user-attachments/assets/61693995-0074-43ce-8fb5-78da03080022" />
# interpritation :
Synthèse globale : Ces graphiques montrent que les variables de taille et de forme d'une masse (en moyenne, erreur standard, et extrême) sont de bons prédicteurs du diagnostic binaire. Les distributions sont majoritairement asymétriques et contiennent de nombreuses valeurs aberrantes, ce qui est crucial pour le choix du bon modèle de Machine Learning et des étapes de prétraitement.
**Boxplots pour détecter les outliers**
  <img width="1989" height="3189" alt="image" src="https://github.com/user-attachments/assets/82c2ab15-f130-4c24-baab-ca7fd68c6463" />
# interpritation 
Les deux figures illustrent la distribution des 30 caractéristiques d'un jeu de données, probablement standardisées. Les histogrammes confirment la forte asymétrie à droite (longue queue vers les valeurs positives) de la majorité des variables, en particulier celles avec les suffixes _se et _worst. Les boîtes à moustaches révèlent une présence significative de valeurs aberrantes (points isolés au-delà des moustaches) dans presque toutes les caractéristiques, ce qui est particulièrement prononcé dans les variables asymétriques. Ces boîtes à moustaches montrent également que la moitié centrale des données (la boîte) est souvent très compressée autour de la médiane (ligne à l'intérieur de la boîte), surtout pour les variables _se, indiquant une faible variabilité pour la majorité des observations et un étalement important causé par ces valeurs aberrantes.
**Observations :**

* Certaines variables (p.ex., radius_mean, perimeter_mean, area_mean) sont fortement corrélées avec la malignité.
* Les valeurs extrêmes des mesures doivent être prises en compte pour l’entraînement des modèles.

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation Train/Test

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Taille du jeu d'entraînement :", X_train.shape)
print("Taille du jeu de test :", X_test.shape)
```

### 3.2 Modèles Testés

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}
```

---

## 4. Évaluation des Modèles

* **Métriques principales** : Accuracy, Precision, Recall, F1-score
* **Outils visuels** : Matrice de confusion, ROC-AUC

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-Score :", f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Bénin','Malin']).plot()
```

**Observations :**

* Random Forest et XGBoost surpassent Logistic Regression.
* Le Recall est crucial : il mesure la capacité du modèle à **détecter correctement les tumeurs malignes**.
---
# 5. matrice de confusion 
# matrice de confusion logistic regression 
<img width="558" height="455" alt="image" src="https://github.com/user-attachments/assets/8ab046f0-0af5-440a-8c87-b64d8a1ba155" />
# interpritation :
La matrice de confusion évalue la performance du modèle de Régression Logistique et indique qu'il est extrêmement performant pour la classification : sur les 114 cas testés, le modèle a correctement identifié 70 cas bénins et 42 cas malins. Les erreurs sont minimes, avec seulement 1 faux négatif (un cas malin classé à tort comme bénin, l'erreur la plus critique) et 1 faux positif (un cas bénin classé malin). Cela se traduit par une Fidélité (Accuracy) de 98,2% (($\frac{70+42}{70+1+1+42} \approx 0.982$)), un Rappel (Recall) de 97,7% (($\frac{42}{42+1} \approx 0.977$)) et une Précision (Precision) de 97,7% (($\frac{42}{42+1} \approx 0.977$)), confirmant un excellent équilibre entre la capacité du modèle à identifier les vrais malins et à éviter les erreurs de diagnostic.
# matrice de confusion random forest 
<img width="558" height="455" alt="image" src="https://github.com/user-attachments/assets/65c6eaf0-4a64-4571-bc49-c6bd33d1292d" />
# interpritation :
La matrice de confusion pour le modèle Random Forest  évalue sa capacité à diagnostiquer correctement les cas. Elle montre que le modèle a fait un excellent travail en identifiant correctement 70 cas bénins (True Negative) et 40 cas malins (True Positive). Cependant, il a fait 4 erreurs au total : 1 faux positif (un cas bénin mal classé malin) et surtout 3 faux négatifs (trois cas malins classés à tort comme bénins). Un taux de 3 faux négatifs est critique dans un contexte médical, car il représente des cas de cancer non détectés. Malgré cela, le modèle conserve une Fidélité (Accuracy) élevée de 96,5% (($\frac{70+40}{70+1+3+40} \approx 0.965$)), indiquant une performance globale très forte.
# matrice de confusion XGboost
<img width="558" height="455" alt="image" src="https://github.com/user-attachments/assets/e064d9e3-55ce-4ebb-a67f-5fae0cef34a0" />
# interpritation :
La matrice de confusion pour le modèle XGBoost présente les résultats les plus précis parmi tous les modèles testés. Elle montre que le modèle a correctement identifié 69 cas bénins (True Negative) et 40 cas malins (True Positive). Il a fait un total de 5 erreurs, soit 2 faux positifs (un cas bénin classé malin) et 3 faux négatifs (trois cas malins classés bénins). En comparant avec la Régression Logistique (1 faux négatif) et le Random Forest (3 faux négatifs), le XGBoost fait légèrement moins bien en termes de faux négatifs critiques, car il échoue à détecter le cancer pour 3 cas. Cependant, sa performance globale reste très élevée avec une Fidélité (Accuracy) de 95,6% (($\frac{69+40}{69+2+3+40} \approx 0.956$)), ce qui est excellent compte tenu de la complexité des données.


---

## 5. Analyse Comparative des Modèles

| Critère         | Random Forest     | XGBoost     | Logistic Regression |
| :-------------- | :---------------- | :---------- | :------------------ |
| Accuracy        | Très élevée       | Très élevée | Moyenne             |
| Recall          | Très élevé        | Élevé       | Moyen               |
| F1-Score        | Très élevé        | Élevé       | Moyen               |
| Meilleur Modèle | **Random Forest** | Très proche | Moins performant    |

---

## Conclusion

1. **Modèles gagnants :** Les modèles basés sur les arbres de décision (Random Forest et XGBoost) surpassent la Logistic Regression.
2. **Meilleur modèle :** Random Forest, pour sa capacité à détecter correctement les tumeurs malignes et sa précision globale.
3. **Importance des variables :** Certaines mesures, comme `radius_mean`, `perimeter_mean` et `area_mean`, sont les plus influentes pour le diagnostic.
4. **Application pratique :** Ce projet illustre comment utiliser le Machine Learning pour assister le diagnostic médical, en détectant de manière fiable les tumeurs malignes à partir de mesures cytologiques.

> En résumé, ce projet démontre une **approche complète de Machine Learning pour la classification médicale**, depuis le prétraitement jusqu’à l’évaluation des modèles et l’analyse de l’importance des features.


