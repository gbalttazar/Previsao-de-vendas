from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def preparar_dados(df):
    df_grouped = df.groupby('Mes_Ano')['Total'].sum().reset_index()
    df_grouped['Mes_Ano'] = df_grouped['Mes_Ano'].astype(str)
    df_grouped['Mes'] = range(len(df_grouped))
    return df_grouped[['Mes']], df_grouped['Total']

def treinar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    erro = mean_squared_error(y_test, y_pred, squared=False)
    return modelo, erro
