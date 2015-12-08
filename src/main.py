from sklearn import datasets

def load_iris_data():
    """
    Méthode permettant de charger en mémoire les données IRIS, contenues dans scikit learn
    """
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target

    print("X: {}".format(X))
    print("Y: {}".format(Y))

def main():
    """
    Main
    """
    pass

if __name__ == '__main__':
    main()
