import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# ============================================================================
# SEÇÃO 0: CONFIGURAÇÃO GERAL
# ============================================================================

st.set_page_config(page_title="Taxa de Administração", layout="wide")

# Configuração do login
SENHA_CORRETA = "taxaswitz"
USUARIO_CORRETO = "witz"

# ============================================================================
# SEÇÃO 1: FUNÇÕES AUXILIARES - TRATAMENTO DE DATAS
# ============================================================================

def parse_data(data_str):
    """Converte string de data em formato DD/MM/YYYY para datetime"""
    if pd.isna(data_str) or data_str == '':
        return None
    try:
        return pd.to_datetime(data_str, format='%d/%m/%Y')
    except:
        return pd.to_datetime(data_str)

def primeiros_uteis_do_periodo(df_dates, periodo='M', dia_util=1):
    """
    Retorna as datas dos primeiros dias úteis de cada período.
    
    Parâmetros:
    - df_dates: Serie de datas
    - periodo: 'M' (mês), 'B' (bimestre), 'Q' (trimestre), 'S' (semestre), 'A' (ano)
    - dia_util: qual dia útil do período (1 = primeiro, 2 = segundo, etc)
    """
    df_dates = pd.Series(df_dates).copy()
    df_dates = pd.to_datetime(df_dates)
    
    # Filtrar apenas dias úteis
    uteis = df_dates[df_dates.dt.weekday < 5]
    
    if len(uteis) == 0:
        return pd.Series([], dtype='datetime64[ns]')
    
    # Agrupar por período
    if periodo == 'M':
        grupos = uteis.dt.to_period('M')
    elif periodo == 'B':  # Bimestre
        bimestre = ((uteis.dt.month - 1) // 2) + 1
        grupos = uteis.dt.year.astype(str) + '-B' + bimestre.astype(str)
    elif periodo == 'Q':  # Trimestre
        trimestre = ((uteis.dt.month - 1) // 3) + 1
        grupos = uteis.dt.year.astype(str) + '-Q' + trimestre.astype(str)
    elif periodo == 'S':  # Semestre
        semestre = ((uteis.dt.month - 1) // 6) + 1
        grupos = uteis.dt.year.astype(str) + '-S' + semestre.astype(str)
    elif periodo == 'A':  # Ano
        grupos = uteis.dt.to_period('Y')
    else:
        raise ValueError('Período inválido')
    
    result_dates = []
    for group_key, dates_group in uteis.groupby(grupos):
        dates_sorted = dates_group.sort_values()
        if len(dates_sorted) >= dia_util:
            result_dates.append(dates_sorted.iloc[dia_util - 1])
        else:
            result_dates.append(dates_sorted.iloc[-1])
    
    return pd.Series(result_dates, dtype='datetime64[ns]').reset_index(drop=True)

def contar_dias_uteis(data_inicio, data_fim):
    """Conta dias úteis entre duas datas"""
    dias = pd.bdate_range(start=data_inicio, end=data_fim)
    return len(dias)

# ============================================================================
# SEÇÃO 2: FUNÇÕES DE CÁLCULO - PROVISÃO DE TAXA
# ============================================================================

def calcular_provisao_diaria(patrimonio, taxa_percentual):
    """
    Calcula a provisão diária de taxa percentual.
    Assume 252 dias úteis no ano.
    """
    taxa_diaria = taxa_percentual / 100 / 252
    return patrimonio * taxa_diaria

def calcular_provisao_escalonada_progressiva(patrimonio, faixas_ativas):
    """
    Calcula provisão com escalonamento progressivo (como IR).
    
    faixas_ativas: list de dicts com {'min': valor_min, 'max': valor_max, 'taxa': taxa_pct}
    """
    faixas_ativas = sorted(faixas_ativas, key=lambda x: x['min'])
    provisao_total = 0
    
    for faixa in faixas_ativas:
        min_faixa = faixa['min']
        max_faixa = faixa['max']
        taxa = faixa['taxa']
        
        # Determinar a porção do patrimônio nesta faixa
        valor_faixa_inicio = max(min_faixa, 0)
        valor_faixa_fim = min(max_faixa, patrimonio)
        
        if valor_faixa_fim > valor_faixa_inicio:
            valor_na_faixa = valor_faixa_fim - valor_faixa_inicio
            taxa_diaria = taxa / 100 / 252
            provisao_total += valor_na_faixa * taxa_diaria
    
    return provisao_total

def calcular_provisao_escalonada_simples(patrimonio, faixas_ativas):
    """
    Calcula provisão com escalonamento simples (faixa substitui a anterior).
    """
    faixas_ativas = sorted(faixas_ativas, key=lambda x: x['min'])
    
    taxa_aplicavel = 0
    for faixa in faixas_ativas:
        if patrimonio >= faixa['min'] and patrimonio <= faixa['max']:
            taxa_aplicavel = faixa['taxa']
            break
    
    taxa_diaria = taxa_aplicavel / 100 / 252
    return patrimonio * taxa_diaria

def obter_regra_vigente(cliente_codigo, data_verificacao, df_regras):
    """
    Obtém a regra de taxa vigente para um cliente em uma data específica.
    """
    df_filtrado = df_regras[df_regras['CodigoCliente'] == cliente_codigo].copy()
    
    regra_vigente = None
    for idx, row in df_filtrado.iterrows():
        data_inicio = parse_data(row['DataInicioVigencia'])
        data_fim = parse_data(row['DataFimVigencia'])
        
        # Se não houver data fim, considera vigente até hoje
        if pd.isna(data_fim):
            data_fim = datetime.now()
        
        if data_inicio <= data_verificacao <= data_fim:
            regra_vigente = row
            break
    
    return regra_vigente

def obter_faixas_vigentes(cliente_codigo, data_verificacao, df_faixas):
    """
    Obtém as faixas de escalonamento vigentes para um cliente em uma data.
    """
    df_filtrado = df_faixas[df_faixas['CodigoCliente'] == cliente_codigo].copy()
    
    faixas_vigentes = []
    for idx, row in df_filtrado.iterrows():
        data_inicio = parse_data(row['DataInicioVigencia'])
        data_fim = parse_data(row['DataFimVigencia'])
        
        # Se não houver data fim, considera vigente até hoje
        if pd.isna(data_fim):
            data_fim = datetime.now()
        
        if data_inicio <= data_verificacao <= data_fim:
            faixas_vigentes.append({
                'min': row['PatrimonioMin'],
                'max': row['PatrimonioMax'],
                'taxa': row['TaxaAplicavel']
            })
    
    return faixas_vigentes

def calcular_provisoes_periodo(cliente_codigo, df_dados, df_regras, df_faixas):
    """
    Calcula todas as provisões e pagamentos de taxa para um cliente.
    """
    
    # Filtrar dados do cliente
    df_cli = df_dados[df_dados['CodigoCliente'] == cliente_codigo].copy()
    
    if df_cli.empty:
        return pd.DataFrame(), {'erro': f'Nenhum dado encontrado para cliente {cliente_codigo}'}
    
    # Converter datas
    df_cli['Data'] = pd.to_datetime(df_cli['Data'], format='%d/%m/%Y')
    df_cli = df_cli.sort_values('Data').reset_index(drop=True)
    
    detalhes = {
        'datas_processadas': len(df_cli),
        'patrimonio_min': df_cli['Patrimonio'].min(),
        'patrimonio_max': df_cli['Patrimonio'].max()
    }
    
    # Obter a regra de taxa vigente
    data_inicio = df_cli['Data'].min()
    regra = obter_regra_vigente(cliente_codigo, data_inicio, df_regras)
    
    if regra is None:
        return pd.DataFrame(), {'erro': f'Nenhuma regra de taxa vigente para cliente {cliente_codigo}'}
    
    detalhes['regra_tipo'] = regra['TipoTaxa']
    detalhes['regra_periodicidade'] = regra['Periodicidade']
    detalhes['regra_escalonamento'] = regra['EscalonamentoTipo']
    detalhes['id_taxa'] = regra.get('IDTaxa', 'N/A')
    detalhes['nome_taxa'] = regra.get('NomeTaxa', 'N/A')
    
    # Calcular provisão diária
    if regra['EscalonamentoTipo'] == 'Progressiva':
        faixas = obter_faixas_vigentes(cliente_codigo, data_inicio, df_faixas)
        if not faixas:
            detalhes['aviso'] = 'Nenhuma faixa de escalonamento encontrada'
            faixas = [{'min': 0, 'max': np.inf, 'taxa': regra['Valor']}]
        
        df_cli['provisao_diaria'] = df_cli['Patrimonio'].apply(
            lambda x: calcular_provisao_escalonada_progressiva(x, faixas)
        )
    elif regra['EscalonamentoTipo'] == 'Simples':
        faixas = obter_faixas_vigentes(cliente_codigo, data_inicio, df_faixas)
        if not faixas:
            detalhes['aviso'] = 'Nenhuma faixa de escalonamento encontrada'
            faixas = [{'min': 0, 'max': np.inf, 'taxa': regra['Valor']}]
        
        df_cli['provisao_diaria'] = df_cli['Patrimonio'].apply(
            lambda x: calcular_provisao_escalonada_simples(x, faixas)
        )
    else:
        if regra['TipoTaxa'] == 'Percentual':
            df_cli['provisao_diaria'] = df_cli['Patrimonio'].apply(
                lambda x: calcular_provisao_diaria(x, regra['Valor'])
            )
        else:
            df_cli['provisao_diaria'] = 0
    
    # Mapear periodicidade
    periodo_map = {
        'Mensal': 'M',
        'Bimestral': 'B',
        'Trimestral': 'Q',
        'Semestral': 'S',
        'Anual': 'A'
    }
    periodo = periodo_map.get(regra['Periodicidade'], 'M')
    
    # Determinar grupos de períodos
    if periodo == 'M':
        df_cli['periodo_grupo'] = df_cli['Data'].dt.to_period('M')
    elif periodo == 'B':
        bimestre = ((df_cli['Data'].dt.month - 1) // 2) + 1
        df_cli['periodo_grupo'] = df_cli['Data'].dt.year.astype(str) + '-B' + bimestre.astype(str)
    elif periodo == 'Q':
        trimestre = ((df_cli['Data'].dt.month - 1) // 3) + 1
        df_cli['periodo_grupo'] = df_cli['Data'].dt.year.astype(str) + '-Q' + trimestre.astype(str)
    elif periodo == 'S':
        semestre = ((df_cli['Data'].dt.month - 1) // 6) + 1
        df_cli['periodo_grupo'] = df_cli['Data'].dt.year.astype(str) + '-S' + semestre.astype(str)
    elif periodo == 'A':
        df_cli['periodo_grupo'] = df_cli['Data'].dt.to_period('Y')
    
    # Agregar por período e calcular pagamentos
    pagamentos_list = []
    
    for periodo_grupo, grupo_df in df_cli.groupby('periodo_grupo'):
        provisao_total = grupo_df['provisao_diaria'].sum()
        
        if regra['TipoTaxa'] == 'Fixo':
            valor_pago = regra['Valor']
        elif regra['TipoTaxa'] == 'Minimo':
            valor_pago = max(provisao_total, regra['Valor'])
        else:
            valor_pago = provisao_total
        
        max_data_periodo = grupo_df['Data'].max()
        prox_dia = max_data_periodo + timedelta(days=1)
        while prox_dia.weekday() >= 5:
            prox_dia += timedelta(days=1)
        
        if valor_pago > 0:
            pagamentos_list.append({
                'data_pagamento': prox_dia,
                'competencia': str(periodo_grupo),
                'valor_pago': valor_pago,
                'provisao_acumulada': provisao_total
            })
    
    df_resultado = pd.DataFrame(pagamentos_list)
    
    return df_resultado, detalhes

# ============================================================================
# SEÇÃO 3: LÓGICA DE AUTENTICAÇÃO
# ============================================================================

# Verificar autenticação
if 'autenticado' not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🔐 Acesso Restrito")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        usuario_input = st.text_input("Usuário", key="usuario_login")
        senha_input = st.text_input("Senha", type="password", key="senha_login")
        
        if st.button("Entrar"):
            if usuario_input == USUARIO_CORRETO and senha_input == SENHA_CORRETA:
                st.session_state.autenticado = True
                st.session_state.usuario = usuario_input
                st.success("✅ Acesso concedido!")
                st.rerun()
            else:
                st.error("❌ Usuário ou senha incorretos")
else:
    # ========================================================================
    # SEÇÃO 4: MENU LATERAL E NAVEGAÇÃO
    # ========================================================================
    
    st.sidebar.title(f"👤 {st.session_state.usuario}")
    
    if st.sidebar.button("🚪 Sair"):
        st.session_state.autenticado = False
        st.session_state.clear()
        st.rerun()
    
    pagina = st.sidebar.radio(
        "Navegação",
        ["📊 Cálculo de Pagamentos", "✅ Validação de Contas", "🧪 Teste de Regras"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Upload de Dados")
    uploaded_file = st.sidebar.file_uploader("Selecione o arquivo Excel", type=['xlsx'], key="main_uploader")
    
    # ========================================================================
    # SEÇÃO 5: PÁGINAS DO APLICATIVO
    # ========================================================================
    
    if uploaded_file:
        # Carregar dados uma única vez
        try:
            df_dados = pd.read_excel(uploaded_file, sheet_name='Dados_Diarios')
            df_regras = pd.read_excel(uploaded_file, sheet_name='Regras_Taxa')
            df_faixas = pd.read_excel(uploaded_file, sheet_name='Faixas_Escalonamento')
            
            # ================================================================
            # PÁGINA 1: CÁLCULO DE PAGAMENTOS
            # ================================================================
            if pagina == "📊 Cálculo de Pagamentos":
                st.title("📊 Cálculo de Pagamentos de Taxa")
                
                clientes = ['Todos os clientes'] + df_dados['CodigoCliente'].astype(str).unique().tolist()
                cliente_sel = st.selectbox('Selecione o cliente', clientes, key="pag1_cliente")
                
                if st.button('Calcular Provisões', key="pag1_calcular"):
                    if cliente_sel == 'Todos os clientes':
                        todos_resultados = []
                        for cliente in clientes[1:]:
                            df_resultado, detalhes = calcular_provisoes_periodo(
                                cliente, df_dados, df_regras, df_faixas
                            )
                            if not df_resultado.empty:
                                df_resultado['CodigoCliente'] = cliente
                                todos_resultados.append(df_resultado)
                        
                        if todos_resultados:
                            df_pagamentos = pd.concat(todos_resultados, ignore_index=True)
                        else:
                            df_pagamentos = pd.DataFrame()
                    else:
                        df_pagamentos, detalhes = calcular_provisoes_periodo(
                            cliente_sel, df_dados, df_regras, df_faixas
                        )
                    
                    if not df_pagamentos.empty:
                        st.success(f"✅ Cálculo realizado para {cliente_sel}!")
                        
                        df_exibicao = df_pagamentos.copy()
                        df_exibicao['data_pagamento'] = pd.to_datetime(df_exibicao['data_pagamento']).dt.strftime('%d/%m/%Y')
                        df_exibicao['valor_pago'] = df_exibicao['valor_pago'].apply(
                            lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                        )
                        
                        if 'provisao_acumulada' in df_exibicao.columns:
                            df_exibicao['provisao_acumulada'] = df_exibicao['provisao_acumulada'].apply(
                                lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                            )
                        
                        if 'CodigoCliente' in df_exibicao.columns:
                            df_exibicao = df_exibicao[['CodigoCliente', 'data_pagamento', 'competencia', 'provisao_acumulada', 'valor_pago']]
                            df_exibicao = df_exibicao.rename(columns={
                                'CodigoCliente': 'Cliente',
                                'data_pagamento': 'Data do Pagamento',
                                'competencia': 'Competência',
                                'provisao_acumulada': 'Provisão Acumulada',
                                'valor_pago': 'Valor Pago'
                            })
                        else:
                            df_exibicao = df_exibicao.rename(columns={
                                'data_pagamento': 'Data do Pagamento',
                                'competencia': 'Competência',
                                'provisao_acumulada': 'Provisão Acumulada',
                                'valor_pago': 'Valor Pago'
                            })
                        
                        st.dataframe(df_exibicao, use_container_width=True)
                        
                        # Exportar
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_exibicao.to_excel(writer, sheet_name='Pagamentos', index=False)
                        output.seek(0)
                        
                        st.download_button(
                            label="📥 Baixar Excel",
                            data=output.getvalue(),
                            file_name=f"pagamentos_{cliente_sel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    else:
                        st.error("❌ Nenhum resultado encontrado.")
            
            # ================================================================
            # PÁGINA 2: VALIDAÇÃO DE CONTAS
            # ================================================================
            elif pagina == "✅ Validação de Contas":
                st.title("✅ Validação de Contas - Provisionamento Diário")
                
                clientes = df_dados['CodigoCliente'].astype(str).unique().tolist()
                cliente_val = st.selectbox('Selecione o cliente para validação', clientes, key="pag2_cliente")
                
                # Filtrar dados do cliente
                df_cli_val = df_dados[df_dados['CodigoCliente'] == cliente_val].copy()
                df_cli_val['Data'] = pd.to_datetime(df_cli_val['Data'], format='%d/%m/%Y')
                df_cli_val = df_cli_val.sort_values('Data').reset_index(drop=True)
                
                # Obter regra vigente
                regra_val = obter_regra_vigente(cliente_val, df_cli_val['Data'].min(), df_regras)
                
                if regra_val is not None:
                    # Exibir informações da taxa
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("ID da Taxa", regra_val.get('IDTaxa', 'N/A'))
                    with col2:
                        st.metric("Nome da Taxa", regra_val.get('NomeTaxa', 'N/A')[:25])
                    with col3:
                        st.metric("Tipo", regra_val['TipoTaxa'])
                    with col4:
                        st.metric("Periodicidade", regra_val['Periodicidade'])
                    
                    st.markdown("---")
                    
                    # Calcular provisão diária
                    if regra_val['EscalonamentoTipo'] == 'Progressiva':
                        faixas = obter_faixas_vigentes(cliente_val, df_cli_val['Data'].min(), df_faixas)
                        if not faixas:
                            faixas = [{'min': 0, 'max': np.inf, 'taxa': regra_val['Valor']}]
                        
                        df_cli_val['provisao_diaria'] = df_cli_val['Patrimonio'].apply(
                            lambda x: calcular_provisao_escalonada_progressiva(x, faixas)
                        )
                    elif regra_val['EscalonamentoTipo'] == 'Simples':
                        faixas = obter_faixas_vigentes(cliente_val, df_cli_val['Data'].min(), df_faixas)
                        if not faixas:
                            faixas = [{'min': 0, 'max': np.inf, 'taxa': regra_val['Valor']}]
                        
                        df_cli_val['provisao_diaria'] = df_cli_val['Patrimonio'].apply(
                            lambda x: calcular_provisao_escalonada_simples(x, faixas)
                        )
                    else:
                        if regra_val['TipoTaxa'] == 'Percentual':
                            df_cli_val['provisao_diaria'] = df_cli_val['Patrimonio'].apply(
                                lambda x: calcular_provisao_diaria(x, regra_val['Valor'])
                            )
                        else:
                            df_cli_val['provisao_diaria'] = 0
                    
                    # Adicionar ID e Nome da taxa
                    df_cli_val['IDTaxa'] = regra_val.get('IDTaxa', 'N/A')
                    df_cli_val['NomeTaxa'] = regra_val.get('NomeTaxa', 'N/A')
                    
                    # Exibir tabela de validação
                    df_val_exibicao = df_cli_val[['Data', 'Patrimonio', 'provisao_diaria', 'IDTaxa', 'NomeTaxa']].copy()
                    df_val_exibicao['Data'] = df_val_exibicao['Data'].dt.strftime('%d/%m/%Y')
                    df_val_exibicao['Patrimonio'] = df_val_exibicao['Patrimonio'].apply(
                        lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                    )
                    df_val_exibicao['provisao_diaria'] = df_val_exibicao['provisao_diaria'].apply(
                        lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                    )
                    
                    df_val_exibicao = df_val_exibicao.rename(columns={
                        'Data': 'Data',
                        'Patrimonio': 'Patrimônio',
                        'provisao_diaria': 'Provisão Diária',
                        'IDTaxa': 'ID da Taxa',
                        'NomeTaxa': 'Nome da Taxa'
                    })
                    
                    st.subheader("📋 Provisionamento Diário")
                    st.dataframe(df_val_exibicao, use_container_width=True)
                    
                    # Estatísticas
                    st.subheader("📈 Estatísticas")
                    col1, col2, col3 = st.columns(3)
                    
                    provisao_total = df_cli_val['provisao_diaria'].sum()
                    patrimonio_medio = df_cli_val['Patrimonio'].mean()
                    
                    with col1:
                        st.metric("Patrimônio Mínimo", f"R$ {df_cli_val['Patrimonio'].min():,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                    with col2:
                        st.metric("Patrimônio Médio", f"R$ {patrimonio_medio:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                    with col3:
                        st.metric("Patrimônio Máximo", f"R$ {df_cli_val['Patrimonio'].max():,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Provisão Total", f"R$ {provisao_total:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                    with col2:
                        st.metric("Dias Processados", len(df_cli_val))
                    
                    # Exportar validação
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_val_exibicao.to_excel(writer, sheet_name='Validação', index=False)
                    output.seek(0)
                    
                    st.download_button(
                        label="📥 Baixar Validação em Excel",
                        data=output.getvalue(),
                        file_name=f"validacao_{cliente_val}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                else:
                    st.error(f"❌ Nenhuma regra de taxa vigente encontrada para {cliente_val}")
            
            # ================================================================
            # PÁGINA 3: TESTE DE REGRAS
            # ================================================================
            elif pagina == "🧪 Teste de Regras":
                st.title("🧪 Teste de Regras - Sandbox")
                st.markdown("Teste diferentes configurações de taxa sem alterar o sistema oficial.")
                
                clientes = df_dados['CodigoCliente'].astype(str).unique().tolist()
                cliente_teste = st.selectbox('Selecione o cliente para teste', clientes, key="pag3_cliente")
                
                # Filtrar dados do cliente
                df_cli_teste = df_dados[df_dados['CodigoCliente'] == cliente_teste].copy()
                df_cli_teste['Data'] = pd.to_datetime(df_cli_teste['Data'], format='%d/%m/%Y')
                df_cli_teste = df_cli_teste.sort_values('Data').reset_index(drop=True)
                
                st.subheader("⚙️ Configuração de Teste")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    id_taxa_teste = st.text_input("ID da Taxa (teste)", value="TAX_TESTE_001")
                    nome_taxa_teste = st.text_input("Nome da Taxa (teste)", value="Taxa de Teste")
                    tipo_taxa_teste = st.selectbox("Tipo de Taxa", ["Percentual", "Fixo", "Minimo"])
                    valor_taxa_teste = st.number_input("Valor da Taxa", min_value=0.0, step=0.01)
                
                with col2:
                    periodicidade_teste = st.selectbox("Periodicidade", ["Mensal", "Bimestral", "Trimestral", "Semestral", "Anual"])
                    escalonamento_teste = st.selectbox("Tipo de Escalonamento", ["Nenhum", "Progressiva", "Simples"])
                
                # Se houver escalonamento, permitir configuração de faixas
                faixas_teste = []
                if escalonamento_teste != "Nenhum":
                    st.markdown("---")
                    st.subheader("📊 Configuração de Faixas")
                    
                    num_faixas = st.number_input("Número de faixas", min_value=1, max_value=5, value=1)
                    
                    for i in range(num_faixas):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            min_patrimonio = st.number_input(f"Faixa {i+1} - Min Patrimônio", min_value=0.0, key=f"min_{i}")
                        with col2:
                            max_patrimonio = st.number_input(f"Faixa {i+1} - Max Patrimônio", min_value=0.0, key=f"max_{i}")
                        with col3:
                            taxa_faixa = st.number_input(f"Faixa {i+1} - Taxa (%)", min_value=0.0, step=0.01, key=f"taxa_faixa_{i}")
                        
                        faixas_teste.append({
                            'min': min_patrimonio,
                            'max': max_patrimonio,
                            'taxa': taxa_faixa
                        })
                
                # Botão para calcular teste
                if st.button("🧪 Executar Teste", key="pag3_calcular"):
                    st.markdown("---")
                    st.subheader("📊 Resultado do Teste")
                    
                    # Calcular provisão diária com regras de teste
                    df_cli_teste_copia = df_cli_teste.copy()
                    
                    if escalonamento_teste == "Progressiva":
                        df_cli_teste_copia['provisao_diaria'] = df_cli_teste_copia['Patrimonio'].apply(
                            lambda x: calcular_provisao_escalonada_progressiva(x, faixas_teste)
                        )
                    elif escalonamento_teste == "Simples":
                        df_cli_teste_copia['provisao_diaria'] = df_cli_teste_copia['Patrimonio'].apply(
                            lambda x: calcular_provisao_escalonada_simples(x, faixas_teste)
                        )
                    else:
                        if tipo_taxa_teste == "Percentual":
                            df_cli_teste_copia['provisao_diaria'] = df_cli_teste_copia['Patrimonio'].apply(
                                lambda x: calcular_provisao_diaria(x, valor_taxa_teste)
                            )
                        else:
                            df_cli_teste_copia['provisao_diaria'] = 0
                    
                    # Adicionar informações do teste
                    df_cli_teste_copia['IDTaxa'] = id_taxa_teste
                    df_cli_teste_copia['NomeTaxa'] = nome_taxa_teste
                    
                    # Exibir tabela
                    df_teste_exibicao = df_cli_teste_copia[['Data', 'Patrimonio', 'provisao_diaria', 'IDTaxa', 'NomeTaxa']].copy()
                    df_teste_exibicao['Data'] = df_teste_exibicao['Data'].dt.strftime('%d/%m/%Y')
                    df_teste_exibicao['Patrimonio'] = df_teste_exibicao['Patrimonio'].apply(
                        lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                    )
                    df_teste_exibicao['provisao_diaria'] = df_teste_exibicao['provisao_diaria'].apply(
                        lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                    )
                    
                    df_teste_exibicao = df_teste_exibicao.rename(columns={
                        'Data': 'Data',
                        'Patrimonio': 'Patrimônio',
                        'provisao_diaria': 'Provisão Diária (TESTE)',
                        'IDTaxa': 'ID da Taxa',
                        'NomeTaxa': 'Nome da Taxa'
                    })
                    
                    st.dataframe(df_teste_exibicao, use_container_width=True)
                    
                    # Estatísticas do teste
                    st.subheader("📈 Estatísticas do Teste")
                    col1, col2, col3 = st.columns(3)
                    
                    provisao_total_teste = df_cli_teste_copia['provisao_diaria'].sum()
                    
                    with col1:
                        st.metric("Provisão Total", f"R$ {provisao_total_teste:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                    with col2:
                        st.metric("Provisão Média Diária", f"R$ {df_cli_teste_copia['provisao_diaria'].mean():,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                    with col3:
                        st.metric("Dias Processados", len(df_cli_teste_copia))
                    
                    # Exportar teste
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_teste_exibicao.to_excel(writer, sheet_name='Teste', index=False)
                    output.seek(0)
                    
                    st.download_button(
                        label="📥 Baixar Resultado do Teste em Excel",
                        data=output.getvalue(),
                        file_name=f"teste_taxa_{cliente_teste}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
    else:
        st.info("📁 Por favor, carregue um arquivo Excel usando o upload no menu lateral.")
