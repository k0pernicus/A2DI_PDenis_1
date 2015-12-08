#-*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets

def load_iris_data():
    """
    Méthode permettant de charger en mémoire les données IRIS, contenues dans scikit learn
    """

    iris = datasets.load_iris()
    X = iris.data[:, :4]  # we only take the first two features.
    Y = iris.target

    return (X, Y)

def center_reduction(X):
    """
    Méthode permettant de centrer-réduire la matrice X, donnée en paramètre
    """

    # Initialisation d'une matrice vide, de taille X
    X_star = np.empty(X.shape)
    (n, p) = X.shape

    # Pour chaque donnée...
    for j in range(p):
        # Prendre la colonne
        col = X[:, j]
        # Calculer la moyenne
        mean = np.mean(col)
        # Calculer l'écart-type
        scol = np.std(col)

        # Pour chaque valeur i,j
        for i in range(n):
            # Mettre à jour!
            X_star[i][j] = (X[i][j] - mean) / scol

    return X_star

def main():
    """
    Main
    """

    (X, _) = load_iris_data()
    X_star = center_reduction(X)

    print(X_star)

if __name__ == '__main__':
    main()
