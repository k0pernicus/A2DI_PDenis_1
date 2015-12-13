#-*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def compute_covariance_matrix(X_star):
    """
    Méthode permettant de calculer la matrice de covariance à partir de la matrice X*, donnée en paramètre
    """

    (n, p) = X_star.shape

    return np.dot(X_star.T, X_star) / n

def compute_singular_value_decomposition(R):
    """
    Méthode permettant de calculer la SVD (ou décomposition spectrale) de la matrice R, donnée en paramètre
    On retournera ainsi un tuple : (U, s, V) où V = U.T et s contient les valeurs propres associés au vecteurs propres u_i
    """

    U,s,_ = np.linalg.svd(R, full_matrices=True)

    return U,s

def sort_proper_vectors(U, s):
    """
    Trie les vecteur propres dans l'ordre décroissant de leurs valeurs propres.
    """

    sAndU = zip(U, s)
    sortedSAndU = sorted(sAndU, key=lambda (_, s_i): s_i, reverse=True)

    return np.asarray(zip(*sortedSAndU)[0])

def compute_pca(X_star, U, k):
    """
    Réduit matrice X_star de dimension (n, p) en une matrice de dimension (n, k) avec k << p.
    """

    return np.dot(X_star, U[:k].T)

def plot_data(Y, k):
    if k == 1:
        plt.plot(Y[0:50, 0], np.zeros_like(Y[0:50, 0]) + 0., 'x', c='g')
        plt.plot(Y[50:100, 0], np.zeros_like(Y[50:100, 0]) + 0., 'x', c='r')
        plt.plot(Y[100:150, 0], np.zeros_like(Y[100:150, 0]) + 0., 'x', c='y')
        plt.xlabel("Variable 1")
    elif k == 2:
        plt.scatter(Y[0:50, 0], Y[0:50, 1], c='g')
        plt.scatter(Y[50:100, 0], Y[50:100, 1], c='r')
        plt.scatter(Y[100:150, 0], Y[100:150, 1], c='y')
        plt.xlabel("Variable 1")
        plt.ylabel("Variable 2")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Y[0:50, 0], Y[0:50, 1], Y[0:50, 2], c='g')
        ax.scatter(Y[50:100, 0], Y[50:100, 1], Y[50:100, 2], c='r')
        ax.scatter(Y[100:150, 0], Y[100:150, 1], Y[100:150, 2], c='y')
        ax.set_xlabel("Variable 1")
        ax.set_ylabel("Variable 2")
        ax.set_zlabel("Variable 3")

    plt.show()

def main():
    """
    Main
    """

    (X, _) = load_iris_data()
    X_star = center_reduction(X)
    (n, p) = X.shape
    R = compute_covariance_matrix(X_star)
    U,s = compute_singular_value_decomposition(R)
    U = sort_proper_vectors(U, s)
    k = 1
    Y = compute_pca(X_star, U, k)


    print("R : {0}".format(R))
    print("X.shape : {0}".format(X_star.shape))
    print("R.shape : {0}".format(R.shape))
    print("U : {0}".format(U))
    print("s : {0}".format(s))

    plot_data(Y, k)

if __name__ == '__main__':
    main()
