import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados(caminho_csv):
    """Carrega os dados do CSV e adiciona a coluna 'Mes_Ano'."""
    df = pd.read_csv(caminho_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Mes_Ano'] = df['Date'].dt.to_period('M')
    return df

def vendas_por_mes(df):
    """Plota o total de vendas por mês."""
    vendas = df.groupby('Mes_Ano')['Total'].sum()
    vendas.plot(kind='bar', color='mediumseagreen')
    plt.title('Total de Vendas por Mês')
    plt.xlabel('Mês')
    plt.ylabel('Total (R$)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def vendas_por_produto(df):
    """Plota as vendas por linha de produto."""
    vendas = df.groupby('Product line')['Total'].sum().sort_values()
    vendas.plot(kind='barh', color='royalblue')
    plt.title('Vendas por Linha de Produto')
    plt.xlabel('Total (R$)')
    plt.ylabel('Linha de Produto')
    plt.tight_layout()
    plt.show()
