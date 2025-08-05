import pandas as pd

def carregar_dados(caminho_excel=r"C:\Users\PedroAugustoBernarde\OneDrive - SWM\Área de Trabalho\base.xlsx"):
    # Leitura
    df = pd.read_excel(caminho_excel)
    df.columns = [col.strip() for col in df.columns]
    df['Data Referência'] = pd.to_datetime(df['Data Referência'])

    # Renomeia
    df.rename(columns={
        'Patrimônio': 'PL Atual',
        'Taxa': 'Taxa Adm (%)',
        'Assessor': 'Assessor',
    }, inplace=True)

    # Preenche e limpa 'Assessor'
    df['Assessor'] = df['Assessor'].fillna('Avin Asset')

    repasse_aai = 0.35

    # Converte para numérico
    df['PL Atual'] = pd.to_numeric(df['PL Atual'], errors='coerce').fillna(0)
    df['Taxa Adm (%)'] = pd.to_numeric(df['Taxa Adm (%)'], errors='coerce').fillna(0)

    # Receita Bruta (Gestora)
    df['Receita Gestora (R$)'] = df['PL Atual'] * df['Taxa Adm (%)']

    # Receita do Assessor (somente se != Avin Asset)
    df['Receita Assessor (R$)'] = 0
    df.loc[df['Assessor'] != 'Avin Asset', 'Receita Assessor (R$)'] = (
        df['Receita Gestora (R$)'] * repasse_aai
    )

    # Receita Líquida
    df['Receita Líquida (R$)'] = df['Receita Gestora (R$)'] - df['Receita Assessor (R$)']
    df['Lucro Estimado (R$)'] = df['Receita Líquida (R$)']

    return df


# Teste local (caso rode o script isoladamente)
if __name__ == "__main__":
    df = carregar_dados()
    print(df.head())
