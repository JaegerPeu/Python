import numpy as np
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import calendar

API_URL = "https://api.comdinheiro.com.br/v1/ep1/import-data"
HEADERS = {'Content-Type': 'application/x-www-form-urlencoded'}

GRANDES_CLASSES = {
    "Equities": ["Equities US", "Equities Global", "Equities EM"],
    "Fixed Income": ["Money Markets", "Investment Grade (3-10)", "High Yield", "Emerging Markets", "Investment Grade (1-3)"],
    "Alternatives": ["Crypto", "Gold"]
}

ATIVOS_PARA_CLASSE = {
    "US:SPY": "Equities US",
    "US:VT": "Equities Global",
    "US:USFR": "Money Markets",
    "US:SPIB": "Investment Grade (3-10)",
    "US:NYSEARCA:SPTL": "Investment Grade (3-10)",
    "US:NASDAQ:SHY": "Investment Grade (1-3)",
    "US:NYSEARCA:EEM": "Emerging Markets",
    "US:SPHY": "High Yield",
    "US:NASDAQ:EMB": "High Yield",
    "US:GDX": "Gold",
    "US:BITO": "Crypto"
}

@st.cache_data
def fetch_data(payload, tab_name="tab0"):
    response = requests.post(API_URL, data=payload, headers=HEADERS)
    response.raise_for_status()
    json_data = response.json()
    tab = json_data["tables"][tab_name]
    headers = [tab["lin0"][col] for col in sorted(tab["lin0"].keys())]
    rows = []
    for i in range(1, len(tab)):
        linha = tab.get(f"lin{i}", None)
        if linha:
            row = [linha[col].replace(",", ".") if isinstance(linha[col], str) else linha[col] for col in sorted(linha.keys())]
            rows.append(row)
    df = pd.DataFrame(rows, columns=headers)
    for col in df.columns[1:]:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    numeric_cols = df.select_dtypes(include=['number']).columns
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols] / 100
    return df

def main():
    st.set_page_config(layout="wide")
    st.title("Lâmina Comitê SWM")

    # Payload definitions omitted for brevity - keep as in your original code.

    tab_consultas, tab_laminas = st.tabs(["Consultas", "Lâminas"])

    with tab_consultas:
        df1 = fetch_data(payload1, tab_name="tab0")
        df2 = fetch_data(payload2, tab_name="tab0")
        df3 = fetch_data(payload3, tab_name="tab1")
        df3.iloc[:, 1:] = df3.iloc[:, 1:] * 10000

        df4 = fetch_data(payload4, tab_name="tab1")
        if df4.shape[1] > 1:
            df4.iloc[:, 1:] = df4.iloc[:, 1:].apply(pd.to_numeric, errors='coerce') * 100

        # Your Consulta expands...

    with tab_laminas:
        carteira_options = list(df2.columns[1:])
        carteira_selecionada = st.selectbox("Selecione o Portfolio", carteira_options)
        st.header(f"Portfolio: {carteira_selecionada}")

        if ("carteira_selecionada_atual" not in st.session_state or
            st.session_state.carteira_selecionada_atual != carteira_selecionada):
            st.session_state.carteira_selecionada_atual = carteira_selecionada
            st.session_state.alocacoes_atualizadas = list(df2[carteira_selecionada].astype(float))
            # Clear reset flag on portfolio change
            if "reset_flag" in st.session_state:
                del st.session_state.reset_flag
            # Force number input keys to refresh on portfolio change by including portfolio in the key

        df_sb = pd.DataFrame()
        df_sb["Raiz"] = [carteira_selecionada] * len(df2)
        df_sb["Ativo"] = df2["nome_fundo"] if "nome_fundo" in df2.columns else df2.index
        df_sb["Proporcao"] = st.session_state.alocacoes_atualizadas

        df_sb["Classe"] = df_sb["Ativo"].map(ATIVOS_PARA_CLASSE).fillna(df_sb["Ativo"])
        df_sb["Grande Classe"] = df_sb["Classe"].apply(
            lambda x: next((g for g, classes in GRANDES_CLASSES.items() if x in classes), "Outros")
        )
        df_sb = df_sb.dropna(subset=["Raiz", "Grande Classe", "Classe", "Proporcao"])

        COLOR_MAP = {
            carteira_selecionada: "#FFFFFF",
            "Equities": "#896F3D",
            "Fixed Income": "#102134",
            "Alternatives": "#C8BEAA",
            "Outros": "#C8BEAA",
            "Desconhecido": "#C8BEAA"
        }

        with st.expander("Asset Allocation %", expanded=False):
            if st.button("Resetar alocações para valores originais"):
                st.session_state.alocacoes_atualizadas = list(df2[carteira_selecionada].astype(float))
                st.session_state.reset_flag = True  # set flag to identify reset
            cols = st.columns(4)
            nova_alocacao = []
            for i, (idx, row) in enumerate(df_sb.iterrows()):
                col = cols[i % 4]
                # use carteira_selecionada + reset_flag to force key change and refresh input when reset
                key = f"input_{i}_{st.session_state.carteira_selecionada_atual}"
                if st.session_state.get("reset_flag", False):
                    key += "_reset"
                valor_atual_pct = st.session_state.alocacoes_atualizadas[i] * 100
                novo_valor = col.number_input(
                    f'{row["Ativo"]} (%)',
                    min_value=0.0,
                    max_value=100.0,
                    value=valor_atual_pct,
                    step=0.5,
                    key=key
                )
                nova_alocacao.append(novo_valor / 100)
            st.session_state.alocacoes_atualizadas = nova_alocacao

            # clear reset flag after inputs update to avoid repeated resetting keys
            if st.session_state.get("reset_flag", False):
                del st.session_state.reset_flag

            soma_pct = sum(nova_alocacao) * 100
            st.write(f"**Soma das alocações:** {soma_pct:.2f}%")

            df_sb["Proporcao"] = nova_alocacao
            soma = sum(nova_alocacao)
            if abs(soma - 1) > 0.001:
                st.warning("A soma das alocações não é 100%. Valores serão proporcionados.")
                df_sb["Proporcao"] = df_sb["Proporcao"] / soma

        # Restante da aba Laminas...
        # Col1, Col2, Col3 setup com composicao, sunburst, retornos como code above
        # E aba Portfolio Backtest no final como no seu código original

if __name__ == "__main__":
    main()
