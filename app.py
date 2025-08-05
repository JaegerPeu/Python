import streamlit as st
import pandas as pd
import calendar
import plotly.express as px
from veiculos import mostrar_veiculos



from data_loader import carregar_dados
from utils import calcular_kpis, top_receita_bruta_por_origem
from sankey_plot import gerar_sankey

# === CONFIGURAÇÕES DA PÁGINA ===
st.set_page_config(
    page_title="Dashboard DRE - Gestora",
    layout="wide",
)

aba1, aba2 = st.tabs(["📊 Visão Geral", "📂 Veículos de Alocação"])

with aba1:
    # === CARREGAMENTO DOS DADOS ===
    df = carregar_dados()
    
    # === SIDEBAR: FILTROS ===
    st.sidebar.title("Filtros")
    
    # Extrai ano e mês da data
    df['Ano'] = df['Data Referência'].dt.year
    df['Mês'] = df['Data Referência'].dt.month
    
    # Filtros interativos
    ano = st.sidebar.selectbox("Ano", sorted(df['Ano'].unique(), reverse=True))
    
    # Converte números dos meses para nomes
    meses_unicos = sorted(df[df['Ano'] == ano]['Mês'].unique(), reverse=True)
    mes_nomes = [calendar.month_name[m] for m in meses_unicos]
    mes_nome = st.sidebar.selectbox("Mês", mes_nomes)
    mes = list(calendar.month_name).index(mes_nome)
    
    # Gera a data selecionada e filtra a base
    data_selecionada = pd.to_datetime(f"{ano}-{mes:02d}-01")
    df_filtrado = df[df['Data Referência'].dt.to_period('M') == data_selecionada.to_period('M')]
    
    # Verifica se há dados para o mês selecionado
    if df_filtrado.empty:
        st.warning("⚠️ Nenhum dado disponível para o mês e ano selecionados.")
        st.stop()
    
    # === TÍTULO DINÂMICO ===
    st.title(f"📊 Dashboard Financeiro – {mes_nome} {ano}")
    
    # === KPIs com Delta em relação ao mês anterior ===
    kpis = calcular_kpis(df_filtrado)
    
    # Data do mês anterior
    data_anterior = (data_selecionada - pd.DateOffset(months=1)).to_period('M')
    df_anterior = df[df['Data Referência'].dt.to_period('M') == data_anterior]
    
    # Calcula KPIs do mês anterior se existir
    if not df_anterior.empty:
        kpis_ant = calcular_kpis(df_anterior)
    else:
        kpis_ant = {k: 0 for k in kpis.keys()}
    
    def calcular_delta(atual, anterior):
        if anterior == 0:
            return "N/A"
        delta = ((atual - anterior) / anterior) * 100
        return f"{delta:+.1f}%"
    
    # Primeira linha
    col1, col2, col3 = st.columns(3)
    col1.metric("💼 PL Total", f"R$ {kpis['PL Total']:,.2f}", calcular_delta(kpis['PL Total'], kpis_ant['PL Total']))
    col2.metric("📈 Receita Bruta", f"R$ {kpis['Receita Bruta Total']:,.2f}", calcular_delta(kpis['Receita Bruta Total'], kpis_ant['Receita Bruta Total']))
    col3.metric("🏦 Receita Líquida - falta custos", f"R$ {kpis['Receita Líquida']:,.2f}", calcular_delta(kpis['Receita Líquida'], kpis_ant['Receita Líquida']))
    
    
    # Segunda linha
    col4, col5, col6 = st.columns(3)
    
    col4.metric("💰 Receita Bruta Gestora", f"R$ {kpis['Receita Gestora']:,.2f}", calcular_delta(kpis['Receita Gestora'], kpis_ant['Receita Gestora']))
    col5.metric("🤝 Receita Bruta AAI", f"R$ {kpis['Receita Assessor']:,.2f}", calcular_delta(kpis['Receita Assessor'], kpis_ant['Receita Assessor']))
    #col6.metric("Lucro Estimado - TBD", f"R$ {kpis['Lucro Estimado']:,.2f}", calcular_delta(kpis['Lucro Estimado'], kpis_ant['Lucro Estimado']))
    col6.empty()
    
    
    st.markdown("---")
    
    # === TABELA TOP ASSESSORES COM FILTRO DE DATA E AJUSTE MENSAL + OUTROS ===
    st.subheader("🔝 Receita Bruta Mensal por Origem")
    
    df_top = df_filtrado.copy()
    df_top['Origem Receita'] = df_top['Assessor'].fillna('Avin Asset')
    df_top['Receita Gestora (R$)'] = df_top['Receita Gestora (R$)'] / 12  # Ajuste para mensal
    
    # Soma total por origem
    receita_por_origem = df_top.groupby('Origem Receita')['Receita Gestora (R$)'].sum()
    
    # Top N assessores
    n_top = 5
    top_assessores = receita_por_origem.sort_values(ascending=False).head(n_top)
    outros_valor = receita_por_origem.sort_values(ascending=False)[n_top:].sum()
    
    # Criação do DataFrame final
    df_resultado = top_assessores.reset_index()
    df_resultado.columns = ["Origem", "Receita Bruta Mensal (R$)"]
    
    # Adiciona linha de "Outros"
    if outros_valor > 0:
        df_resultado.loc[len(df_resultado.index)] = ["Outros", outros_valor]
    
    # Formatação para exibição
    df_resultado["Receita Bruta Mensal (R$)"] = df_resultado["Receita Bruta Mensal (R$)"].map(lambda x: f"R$ {x:,.2f}")
    st.dataframe(df_resultado, use_container_width=True)
    
    
    
    # === Sankey ===
    st.markdown("---")
    st.subheader("🔀 Fluxo da Receita Bruta")
    
    modo = st.selectbox("Visualização", ["geral", "gestora", "assessor"])
    incluir_inst = st.checkbox("Incluir Instituições", value=False)
    incluir_veic = st.checkbox("Incluir Veículo", value=False)
    incluir_onsh = st.checkbox("Incluir Onshore/Offshore", value=False)
    
    if modo == "assessor":
        assessores_unicos = sorted(df_filtrado['Assessor'].dropna().unique())
        assessor_escolhido = st.selectbox("Escolha o Assessor", assessores_unicos)
        fig_sankey = gerar_sankey(
            df_filtrado,
            modo='assessor',
            assessor_especifico=assessor_escolhido,
            receita_bruta_total=kpis['Receita Bruta Total']
        )
    
    elif modo == "gestora":
        fig_sankey = gerar_sankey(
            df_filtrado,
            modo='gestora',
            receita_bruta_total=kpis['Receita Bruta Total']
        )
    
    else:
        fig_sankey = gerar_sankey(
        df_filtrado,
        modo='geral',
        incluir_instituicoes=incluir_inst,
        incluir_veiculo=incluir_veic,
        incluir_onshore=incluir_onsh
    )
    
    
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    
    # === AUM Evolução ===
    st.subheader("📈 Evolução do Patrimônio da Gestora")
    
    # Converte a coluna de data
    df['Data Referência'] = pd.to_datetime(df['Data Referência'], errors='coerce')
    
    # Filtros de período
    min_data = df['Data Referência'].min().date()
    max_data = df['Data Referência'].max().date()
    
    col1, col2 = st.columns(2)
    data_inicio = col1.date_input("Data Início", min_data, min_value=min_data, max_value=max_data)
    data_fim = col2.date_input("Data Fim", max_data, min_value=min_data, max_value=max_data)
    
    # Validação
    if data_inicio > data_fim:
        st.warning("⚠️ Data início maior que data fim.")
    else:
        df_filtrado = df[(df['Data Referência'] >= pd.to_datetime(data_inicio)) & (df['Data Referência'] <= pd.to_datetime(data_fim))]
    
        if df_filtrado.empty:
            st.info("Nenhum dado no período selecionado.")
        else:
            df_aum = df_filtrado.groupby('Data Referência')['PL Atual'].sum().reset_index()
            fig_aum = px.line(df_aum, x='Data Referência', y='PL Atual', title='Evolução do AUM da Gestora',
                              labels={'PL Atual': 'Patrimônio Líquido (R$)', 'Data Referência': 'Data Referência'})
            fig_aum.update_traces(line=dict(width=2))
            st.plotly_chart(fig_aum, use_container_width=True)

# === CAMINHO E ABA ===



with aba2:
    st.subheader("📁 Detalhamento dos Fundos e Veículos")
    caminho_arquivo = "numeros.xlsx"
    mostrar_veiculos(caminho_arquivo)

