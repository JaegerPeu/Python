import plotly.graph_objects as go
import plotly.colors as pc

def gerar_sankey(df, modo='geral', assessor_especifico=None, incluir_instituicoes=False, incluir_veiculo=False, incluir_onshore=False):
    df = df.copy()
    df['Origem Receita'] = df['Assessor'].fillna('Avin Asset')

    # Ajusta a receita para base mensal, assumindo taxa anual
    df['Receita Gestora (R$)'] = df['Receita Gestora (R$)'] / 12
    df['Receita Líquida (R$)'] = df['Receita Líquida (R$)'] / 12
    df['Receita Assessor (R$)'] = df['Receita Assessor (R$)'] / 12
    if 'Custo Fixo Rateado (R$)' in df.columns:
        df['Custo Fixo Rateado (R$)'] = df['Custo Fixo Rateado (R$)'] / 12

    labels = []
    source = []
    target = []
    value = []

    def add_label(label, label_dict):
        if label not in label_dict:
            label_dict[label] = len(labels)
            labels.append(label)
        return label_dict[label]

    if modo == 'geral':
        receita_bruta_total = df['Receita Gestora (R$)'].sum()
        label_dict = {}
        idx_receita = add_label(f'Receita Bruta\nR$ {receita_bruta_total:,.2f}', label_dict)

        niveis = []

        if incluir_onshore:
            niveis.append("Onshore_Offshore")
            df["Onshore_Offshore"] = df["Onshore_Offshore"].fillna("Indefinido")

        if incluir_veiculo:
            niveis.append("Veículo")
            df["Veículo"] = df["Veículo"].fillna("Indefinido")

        if incluir_instituicoes:
            niveis.append("Instituição")
            df["Instituição"] = df["Instituição"].fillna("Indefinido")

        niveis.append("Origem Receita")

        # Unifica origem
        df['Origem Receita'] = df['Origem Receita'].apply(lambda x: 'Avin Asset' if x == 'Avin Asset' else 'Assessor')

        # Agrupamento por todos os níveis
        agrupado = df.groupby(niveis)['Receita Gestora (R$)'].sum().reset_index()

        fluxos = {}
        caminho_valores = {}

        for _, row in agrupado.iterrows():
            caminho = [f'Receita Bruta\nR$ {receita_bruta_total:,.2f}']
            prefixo = ()
            for nivel in niveis:
                prefixo += (row[nivel],)
                valor = df.loc[(df[niveis[:len(prefixo)]].apply(tuple, axis=1) == prefixo), 'Receita Gestora (R$)'].sum()
                label_formatada = f"{row[nivel]}\nR$ {valor:,.2f}"
                caminho.append(label_formatada)

            for i in range(len(caminho) - 1):
                key = (caminho[i], caminho[i+1])
                fluxos[key] = fluxos.get(key, 0) + row['Receita Gestora (R$)']

        label_dict = {}
        for (pai, filho), val in fluxos.items():
            pai_idx = add_label(pai, label_dict)
            filho_idx = add_label(filho, label_dict)
            source.append(pai_idx)
            target.append(filho_idx)
            value.append(val)

    elif modo == 'gestora':
        receita_liquida = df[df['Origem Receita'] == 'Avin Asset']['Receita Líquida (R$)'].sum()
        custo_total = df[df['Origem Receita'] == 'Avin Asset']['Custo Fixo Rateado (R$)'].sum() if 'Custo Fixo Rateado (R$)' in df.columns else 0
        lucro = receita_liquida - custo_total

        labels = [
            f'Avin Asset\nR$ {receita_liquida + custo_total:,.2f}',
            f'Custos Fixos\nR$ {custo_total:,.2f}',
            f'Lucro Estimado\nR$ {lucro:,.2f}'
        ]
        source = [0, 0]
        target = [1, 2]
        value = [custo_total, lucro]

    elif modo == 'assessor' and assessor_especifico:
        valor = df[df['Origem Receita'] == assessor_especifico]['Receita Assessor (R$)'].sum()
        labels = [
            f'Receita Bruta\nR$ {valor:,.2f}',
            f'{assessor_especifico}\nR$ {valor:,.2f}'
        ]
        source = [0]
        target = [1]
        value = [valor]

    else:
        raise ValueError("Modo inválido ou assessor não informado.")

    paleta = pc.qualitative.Set3
    cores = [paleta[i % len(paleta)] for i in range(len(labels))]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=cores
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(150,150,255,0.3)"
        )
    )])

    fig.update_layout(title_text="🔀 Fluxo da Receita", font_size=12)
    return fig