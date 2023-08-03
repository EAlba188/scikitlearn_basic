import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA # de decomposition modulos de reduccion de dimensionalidad
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv("./data/Heart.csv")

    """Features son las caracteristicas, como queremos saber sobre target entonces la quitamos de la ecuacion"""
    dt_features = dt_heart.drop(["target"], axis=1)

    """Aqui aislamos nuestro objetivo"""
    dt_target = dt_heart["target"]

    """Transformamos para normalizar la funcion"""
    dt_features = StandardScaler().fit_transform(dt_features)

    """
    - train test split  Esta es una función de la biblioteca scikit-learn que se utiliza para dividir un conjunto de datos en dos partes: una para entrenamiento y otra para pruebas
    - Random state hace que siempre haga la prueba sobre la misma parte para que siempre de la misma conclusion
    si no estuviera entonces cada vez se haria sobre una parte en especifico
    - test size es el porcentaje sobre el que se entrenara
    - las Y son las etiquetas, osea en este caso los target
    """
    x_train, x_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    """
    Que hace PCA?
    PCA busca encontrar las combinaciones lineales de las variables originales que capturan la mayor parte de la variabilidad en los datos. Estas combinaciones lineales son los componentes principales, que son nuevas variables que son ortogonales entre sí y están ordenadas de manera que el primer componente principal captura la mayor varianza posible, el segundo componente principal captura la segunda mayor varianza, y así sucesivamente.
    """

    """Por defecto si no pasamos n_componens sera igual al minimo entre el num de muestras y el numero de features"""
    pca = PCA(n_components=3)
    pca.fit(x_train)

    """
    Ipca tiene batch_size que es que no manda todos los datos a entrenar al mismo tiempo sino por bloquecitos
    """
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(x_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    """
    Implementa un algoritmo de regresion logistica, el solver es el tipo de algoritmo de optimizacion
    """
    logistic = LogisticRegression(solver="lbfgs")

    dt_train = pca.transform(x_train)
    dt_test = pca.transform(x_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    dt_train = ipca.transform(x_train)
    dt_test = ipca.transform(x_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))






