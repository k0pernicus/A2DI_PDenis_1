# TP 1

## Sujet

Analyse en Composantes Principales

## Procédure

0.  Télécharger les données IRIS : https://archive.ics.uci.edu/ml/datasets/Iris
1.  Projeter les données en 1, 2 et 3 dimensions.  
    Dans chaque cas, visualiser les "nouvelles" données en *scatterplots*.  
    Utiliser des couleurs différentes pour chaque classe.
2.  Les nouvelles représentations permettent-elles de séparer clairement les différentes classes d'IRIS?
3.  Calculer le pourcentage de variance totale, expliquée par chaque projection.  
    Commenter.
4.  Expliquer comment reconstruire les données originales à partir des données trasnformées et des vecteurs propres.  
    Essayer de reconstruire les données en utilisant 1,2 et 3 vecteurs propres.
5.  Calculer l'erreur de reconstruction pour les projections en 1,2 et 3 dimensions.
6.  Quelle reconstruction est la meilleure?
    Expliquer.

## Utilisation

Pour lancer le programme, utiliser la commande :
```
python src/main.py [k]
```

k peut valoir 1, 2 ou 3 et correspond aux nombre de dimensions des données réduites.
