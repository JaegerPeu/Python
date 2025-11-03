import streamlit as st
import pandas as pd



def primeiros_uteis_do_periodo(df_dates, periodo='M', dia_util=1):
    df_dates = pd.Series(df_dates).copy()
    df_dates = pd.to_datetime(df_dates)
    uteis = df_dates[df_dates.dt.weekday < 5]

    if periodo == 'M':
        grupos = uteis.dt.to_period('M')
    elif periodo == 'Q':
        trimestre = ((uteis.dt.month - 1)//3) + 1
        grupos = uteis.dt.year.astype(str) + '-Q' + trimestre.astype(str)
    elif periodo == '6M':
        semestre = ((uteis.dt.month - 1)//6) + 1
        grupos = uteis.dt.year.astype(str) + '-S' + semestre.astype(str)
    elif periodo == 'A':
        grupos = uteis.dt.to_period('Y')
    else:
        raise ValueError('Período inválido')

    result_dates = []
    for group_key, dates_group in uteis.groupby(grupos):
        ultimo_dia = dates_group.max()
        prox_periodo_inicio = None
        if periodo == 'M':
            prox_periodo_inicio = (ultimo_dia + pd.offsets.MonthBegin(1)).to_period('M').start_time
        elif periodo == 'Q':
            mes_prox = (((ultimo_dia.month - 1)//3) + 1)*3 + 1
            ano_prox = ultimo_dia.year
            if mes_prox > 12:
                mes_prox -= 12
                ano_prox += 1
            prox_periodo_inicio = pd.Timestamp(year=ano_prox, month=mes_prox, day=1)
        elif periodo == '6M':
            semestre_atual = ((ultimo_dia.month - 1)//6) + 1
            ano_prox = ultimo_dia.year
            semestre_prox = semestre_atual + 1
            if semestre_prox > 2:
                semestre_prox = 1
                ano_prox += 1
            mes_prox = (semestre_prox - 1)*6 + 1
            prox_periodo_inicio = pd.Timestamp(year=ano_prox, month=mes_prox, day=1)
        elif periodo == 'A':
            ano_prox = ultimo_dia.year + 1
            prox_periodo_inicio = pd.Timestamp(year=ano_prox, month=1, day=1)

        prox_periodo_uteis = pd.bdate_range(start=prox_periodo_inicio, periods=dia_util)
        if len(prox_periodo_uteis) < dia_util:
            data_pagto = prox_periodo_uteis[-1]
        else:
            data_pagto = prox_periodo_uteis[dia_util - 1]
        result_dates.append(data_pagto)

    return pd.Series(result_dates)

# ========== Caminho para arquivo Excel ==========
# Para rodar localmente no Windows, use caminho absoluto ou relativo corretamente,
# Exemplo local: 'C:/Users/.../pl_diario_todos_clientes.xlsx'
# Para publicar no GitHub e rodar Streamlit via GitHub Codespaces ou outros, usar caminho relativo:
# arquivo na raiz do repo: caminho_arquivo = 'pl_diario_todos_clientes.xlsx'
caminho_arquivo = "pl_diario_todos_clientes.xlsx" 
df = pd.read_excel(caminho_arquivo, sheet_name='Ativos')
clientes = df['customerCode'].astype(str).unique().tolist()

st.title("Provisionamento da Taxa de Administração")

# Opção 'Todos os clientes' adicionada
clientes_opcoes = ['Todos os clientes'] + clientes
cliente_sel = st.selectbox('Selecione o cliente', clientes_opcoes)
taxa_anual = st.number_input('Taxa anual (%)', min_value=0.0, max_value=100.0, value=1.0, step=0.01)
periodicidade = st.selectbox('Periodicidade do pagamento', ['Mensal', 'Trimestral', 'Semestral', 'Anual'])
dia_util_pagto = st.number_input('Dia útil para pagamento', min_value=1, max_value=22, value=1)

# Filtra cliente ou usa todos
if cliente_sel == 'Todos os clientes':
    df_cli = df.copy()
else:
    df_cli = df[df['customerCode'].astype(str) == cliente_sel].copy()

df_cli['date'] = pd.to_datetime(df_cli['date'].str[:10])
df_cli = df_cli.sort_values('date').reset_index(drop=True)
taxa_dia = taxa_anual / 100 / 252
df_cli['taxa_diaria'] = df_cli['plDaily'] * taxa_dia

periodo_map = {
    'Mensal': 'M',
    'Trimestral': 'Q',
    'Semestral': '6M',
    'Anual': 'A'
}

periodo = periodo_map[periodicidade]

df_cli['competencia'] = (df_cli['date'] - pd.offsets.MonthBegin(1)).dt.to_period(periodo)
dias_pagto = primeiros_uteis_do_periodo(df_cli['date'], periodo, dia_util_pagto)

pagamentos_list = []
for pagto in dias_pagto:
    comp_antes = (pagto - pd.offsets.MonthBegin(1)).to_period(periodo)
    periodo_inicio = comp_antes.start_time
    periodo_fim = comp_antes.end_time
    linhas_comp = df_cli[(df_cli['date'] >= periodo_inicio) & (df_cli['date'] <= periodo_fim)]
    valor_pago = linhas_comp['taxa_diaria'].sum()
    if valor_pago > 0:
        pagamentos_list.append({'data_pagamento': pagto, 'competencia': str(comp_antes), 'valor_pago': valor_pago})

df_pagamentos = pd.DataFrame(pagamentos_list)

df_pagamentos['valor_pago'] = df_pagamentos['valor_pago'].apply(lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
df_pagamentos['data_pagamento'] = pd.to_datetime(df_pagamentos['data_pagamento']).dt.strftime('%d/%m/%Y')

df_pagamentos = df_pagamentos.rename(columns={
    'data_pagamento': 'Data do Pagamento',
    'competencia': 'Competência',
    'valor_pago': 'Valor Pago'
})

st.subheader('Pagamentos realizados')
st.dataframe(df_pagamentos)

if st.button('Exportar para Excel'):
    df_pagamentos.to_excel(f'pagamentos_{cliente_sel}.xlsx', index=False)
    st.success(f'Arquivo exportado: pagamentos_{cliente_sel}.xlsx')
