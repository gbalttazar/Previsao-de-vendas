import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analise import carregar_dados, vendas_por_mes, vendas_por_produto
from modelo_regressao import preparar_dados, treinar_modelo_polinomial, prever

def plot_modelo_ajuste(modelo, X, y, grau, titulo="Ajuste do Modelo", salvar=False, caminho_salvar='assets/grafico_ajuste.png'):
    """
    Plota os dados reais e a curva ajustada do modelo.
    Se salvar=True, salva o gráfico no caminho especificado.
    """
    X_range = np.linspace(X['Mes_Num'].min(), X['Mes_Num'].max() + 1, 100)
    X_range_df = pd.DataFrame({'Mes_Num': X_range})
    y_pred_range = modelo.predict(X_range_df)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Dados Reais')
    plt.plot(X_range, y_pred_range, color='red', label=f'Modelo Polinomial (grau {grau})')
    plt.xlabel('Mês')
    plt.ylabel('Total de Vendas')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()

    if salvar:
        plt.savefig(caminho_salvar)
        print(f"[✔] Gráfico salvo em: {caminho_salvar}")
    else:
        plt.show()

if __name__ == '__main__':
    CAMINHO_CSV = 'data/supermarket_sales.csv'

    
    os.makedirs('assets', exist_ok=True)

  
    df = carregar_dados(CAMINHO_CSV)
    X, y = preparar_dados(df)

    
    grau_escolhido = 2
    modelo, erro = treinar_modelo_polinomial(X, y, grau=grau_escolhido)

    print(f"(Apenas para visualização) Grau {grau_escolhido} - RMSE: {erro:.2f}")

 
    plot_modelo_ajuste(modelo, X, y, grau_escolhido, salvar=True)

    
    print("\nPrevisões futuras:")
    for i in range(1, 4):
        mes_futuro = X['Mes_Num'].max() + i
        previsao = prever(modelo, mes_futuro)
        print(f"Mês {int(mes_futuro)}: R$ {previsao[0]:,.2f}")
