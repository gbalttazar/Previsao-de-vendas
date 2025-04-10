import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def preparar_dados(df):
    """
    Agrupa as vendas por mês e cria uma feature numérica 'Mes_Num' para regressão.
    """
    vendas_mensais = df.groupby('Mes_Ano')['Total'].sum().reset_index()
    vendas_mensais['Mes_Num'] = range(len(vendas_mensais))
    return vendas_mensais[['Mes_Num']], vendas_mensais['Total']

def treinar_modelo_polinomial(X, y, grau=2):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=grau)),
        ('linear', LinearRegression())
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    erro = np.sqrt(mse)
    return pipeline, erro

def prever(modelo, mes_futuro):
    import pandas as pd
    X_novo = pd.DataFrame({'Mes_Num': [mes_futuro]})
    return modelo.predict(X_novo)
