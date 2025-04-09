import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Mes_Ano'] = df['Date'].dt.to_period('M')
    return df

def plot_vendas_por_mes(df):
    vendas = df.groupby('Mes_Ano')['Total'].sum()
    vendas.plot(kind='bar', color='mediumseagreen')
    plt.title('Vendas por Mês')
    plt.ylabel('Total (R$)')
    plt.xlabel('Mês')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

