from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def valida_modelo_polinomial(X, y, grau=2, cv=5):
    """
    Utiliza cross_val_score para validar o modelo polinomial e retorna a média do RMSE.
    """
    # Cria um pipeline com transformação polinomial e regressão linear
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=grau)),
        ('linear', LinearRegression())
    ])
    # Usando cross_val_score com scoring negativo de RMSE (o cross_val_score retorna valores negativos para erros)
    # Precisamos tirar a raiz quadrada e inverter o sinal para obter o RMSE.
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean()

# Exemplo de uso:
if __name__ == '__main__':
    from analise import carregar_dados
    from modelo_regressao import preparar_dados, treinar_modelo_polinomial, valida_modelo_polinomial
    CAMINHO_CSV = 'data/supermarket_sales.csv'
    df = carregar_dados(CAMINHO_CSV)
    X, y = preparar_dados(df)
    
    graus_teste = [1, 2, 3, 4]
    print("=== Validação Cruzada para Modelos Polinomiais ===")
    for grau in graus_teste:
        rmse_cv = valida_modelo_polinomial(X, y, grau=grau)

        print(f"Grau: {grau} - RMSE (CV): {rmse_cv:.2f}")
