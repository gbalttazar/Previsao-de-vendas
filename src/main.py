import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_modelo_ajuste(modelo, X, y, grau, titulo="Ajuste do Modelo", salvar=False, caminho_salvar='grafico_ajuste.png'):
    """
    Plota os dados reais (pontos) e a curva predita pelo modelo.
    Se salvar=True, o gráfico será salvo no caminho especificado.
    """
    # Gerar um range para previsão
    X_range = np.linspace(X['Mes_Num'].min(), X['Mes_Num'].max() + 1, 100)
    X_range_df = pd.DataFrame({'Mes_Num': X_range})

    # Previsões do modelo
    y_pred_range = modelo.predict(X_range_df)

    # Plotar os dados reais
    plt.scatter(X, y, color='blue', label='Dados Reais')
    
    # Plotar a curva ajustada
    plt.plot(X_range, y_pred_range, color='red', label=f'Modelo Polinomial (grau {grau})')
    
    # Configurações do gráfico
    plt.xlabel('Mes_Num')
    plt.ylabel('Total de Vendas')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()

    if salvar:
        plt.savefig(caminho_salvar)
        print(f"Gráfico salvo como: {caminho_salvar}")
    else:
        plt.show()


if __name__ == '__main__':
    from analise import carregar_dados, vendas_por_mes, vendas_por_produto
    from modelo_regressao import preparar_dados, treinar_modelo_polinomial, prever

    CAMINHO_CSV = 'data/supermarket_sales.csv'

    # Carregar dados
    df = carregar_dados(CAMINHO_CSV)

    # Prepara os dados agregados por mês
    X, y = preparar_dados(df)

    # Treinar modelo polinomial
    grau_escolhido = 2
    modelo, erro = treinar_modelo_polinomial(X, y, grau=grau_escolhido)

    print(f"(Apenas para visualização) Grau {grau_escolhido} - RMSE: {erro:.2f}")

    # Plotar gráfico (pode salvar com salvar=True)
    plot_modelo_ajuste(modelo, X, y, grau_escolhido, salvar=False)

    # Previsões futuras
    print("\nPrevisões futuras:")
    for i in range(1, 4):
        mes_futuro = X['Mes_Num'].max() + i
        previsao = prever(modelo, mes_futuro)
        print(f"Mês {int(mes_futuro)}: R$ {previsao[0]:,.2f}")
