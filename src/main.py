import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_modelo_ajuste(modelo, X, y, grau, titulo="Ajuste do Modelo"):
    """
    Plota os dados reais (pontos) e a curva predita pelo modelo.
    """
    # Gerando um range para a previsão: dos meses existentes até um valor um pouco além
    X_range = np.linspace(X.min(), X.max() + 1, 100)
    X_range_df = pd.DataFrame({'Mes_Num': X_range})
    
    # Previsões do modelo para esse range
    y_pred_range = modelo.predict(X_range_df)
    
    # Plot dos dados reais
    plt.scatter(X, y, color='blue', label='Dados Reais')
    
    # Plot da curva predita
    plt.plot(X_range, y_pred_range, color='red', label=f'Modelo Polinomial (grau {grau})')
    
    # Configurar gráfico
    plt.xlabel('Mes_Num')
    plt.ylabel('Total de Vendas')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Exemplo de uso no main.py
if __name__ == '__main__':
    from analise import carregar_dados, vendas_por_mes, vendas_por_produto
    from modelo_regressao import preparar_dados, treinar_modelo_polinomial, prever
    CAMINHO_CSV = 'data/supermarket_sales.csv'

    # Carregar dados
    df = carregar_dados(CAMINHO_CSV)
    # Prepara os dados agregados por mês
    X, y = preparar_dados(df)

    # Treinar modelo polinomial para um grau de interesse, por exemplo 2
    grau_escolhido = 2
    modelo, erro = treinar_modelo_polinomial(X, y, grau=grau_escolhido)
    
    print(f"(Apenas para visualização) Grau {grau_escolhido} - RMSE: {erro:.2f}")
    
    # Plotar o ajuste do modelo
    plot_modelo_ajuste(modelo, X, y, grau_escolhido)
