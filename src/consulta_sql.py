import sqlite3
import pandas as pd

def salvar_em_sqlite(df, nome_banco='vendas.db'):
    conn = sqlite3.connect(nome_banco)
    df.to_sql('vendas', conn, if_exists='replace', index=False)
    conn.close()

def consulta_top_produtos(nome_banco='vendas.db'):
    conn = sqlite3.connect(nome_banco)
    query = """
    SELECT [Product line], SUM(Total) as Total_Vendido
    FROM vendas
    GROUP BY [Product line]
    ORDER BY Total_Vendido DESC
    """
    resultado = pd.read_sql_query(query, conn)
    conn.close()
    return resultado
