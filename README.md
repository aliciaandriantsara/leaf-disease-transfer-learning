# Détection de maladies sur feuilles de tomate — Transfer Learning

## Contexte et objectif

Projet réalisé dans le cadre d'une candidature au Madagascar Data Science Hackathon 2026 (Orange Digital Center). L'objectif est de construire un classificateur d'images capable de distinguer une feuille de tomate saine de feuilles atteintes de différentes maladies, à partir d'une simple photo — un cas d'usage concret pour un agriculteur ou un agent agricole, sans expertise technique requise.

## Données

Dataset public **PlantVillage** (Kaggle : emmarex/plantdisease). Sur les 15 classes disponibles (plusieurs plantes et maladies), un sous-ensemble de **5 classes de tomate** a été sélectionné pour garder le projet cohérent et les effectifs équilibrés :

- Tomato_healthy
- Tomato_Bacterial_spot
- Tomato_Late_blight
- Tomato_Septoria_leaf_spot
- Tomato_Spider_mites_Two_spotted_spider_mite

Total : 9 074 images, réparties en train (70%), validation (15%) et test (15%).

## Méthode

**Transfer learning** avec **MobileNetV2** pré-entraîné sur ImageNet :

1. Chargement de MobileNetV2 sans sa couche de classification finale (`include_top=False`)
2. Ajout de couches personnalisées : `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(5, softmax)`
3. Data augmentation (flip horizontal, rotation, zoom) et preprocessing spécifique à MobileNetV2
4. **Phase 1** : entraînement des couches ajoutées uniquement, modèle de base gelé (8 epochs)
5. **Phase 2** : fine-tuning des 30 dernières couches de MobileNetV2 avec un learning rate réduit (1e-5, 5 epochs)

## Résultats

**Accuracy sur le jeu de test : 95%**

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Tomato_Bacterial_spot | 0.99 | 0.92 | 0.95 |
| Tomato_Late_blight | 0.98 | 0.93 | 0.96 |
| Tomato_Septoria_leaf_spot | 0.91 | 0.92 | 0.91 |
| Tomato_Spider_mites | 0.90 | 0.98 | 0.94 |
| Tomato_healthy | 0.94 | 1.00 | 0.97 |

Matrice de confusion disponible dans `images_resultats_matrice_confusion.png`.

## Outils

Python, TensorFlow/Keras, Google Colab (GPU), scikit-learn, Matplotlib/Seaborn.

## Reproduire le projet

1. Ouvrir `notebook_classification.ipynb` dans Google Colab
2. Installer les dépendances : `pip install -r requirements.txt`
3. Configurer les tokens API (GitHub, Kaggle) dans les Secrets Colab
4. Exécuter les cellules dans l'ordre

## Statut

Projet terminé.
