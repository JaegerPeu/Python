import pandas as pd

def calcular_kpis(df):
    """Calcula os principais KPIs agregados com receita ajustada para base mensal."""
    kpis = {
        'PL Total': df['PL Atual'].sum(),
        'Receita Bruta Total': (df['Receita Gestora (R$)'].sum()) / 12,
        'Receita Assessor': (df['Receita Assessor (R$)'].sum()) / 12,
        'Receita Líquida': (df['Receita Líquida (R$)'].sum()) / 12,
        'Lucro Estimado': (df['Lucro Estimado (R$)'].sum()) / 12,
        'Receita Gestora': (df['Receita Gestora (R$)'].sum()/12)-(df['Receita Assessor (R$)'].sum()/12)
    }
    return kpis


def top_receita_bruta_por_origem(df, n=10):
    """Top N por origem de receita bruta:
       - Receita Assessor para assessores
       - Receita Gestora (bruta) para Avin Asset"""
    
    df = df.copy()

    df['Receita Bruta por Origem'] = df.apply(
        lambda row: row['Receita Assessor (R$)'] if row['Assessor'] != 'Avin Asset' else row['Receita Gestora (R$)'],
        axis=1
    )

    agrupado = (
        df.groupby('Assessor')['Receita Bruta por Origem']
        .sum()
        .sort_values(ascending=False)
    )

    top = agrupado.head(n)
    outros = agrupado[n:].sum()

    if outros > 0:
        top['Outros Assessores'] = outros

    return top



def filtrar_por_data(df, data_referencia=None):
    """Filtra os dados para uma data específica se fornecida"""
    if data_referencia:
        df = df[df['Data Referência'] == pd.to_datetime(data_referencia)]
    return df
