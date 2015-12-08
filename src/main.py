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

    # Obtention de la taille de la matrice
    (n, p) = X.shape

    # Initialisation d'une matrice vide, de taille X
    X_star = np.empty((n, p))

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

def compute_singular_value_decomposition(X_star):
    """
    Méthode permettant de calculer la SVD (ou décomposition spectrale) de la matrice X*, donnée en paramètre
    On retournera ainsi un tuple : (U, s, V)
    U: Les vecteurs propres de X* * X*.T
    s: La diagonale de la matrice contenant les racines des vecteurs propres
    V: Les vecteurs propres de X*.T * X*
    """

    return np.linalg.svd(X_star)

def main():
    """
    Main
    """

    (X, _) = load_iris_data()
    X_star = center_reduction(X)

    print(X_star)

if __name__ == '__main__':
    main()
