import streamlit as st
import pandas as pd
import io
import numpy as np

st.set_page_config(page_title="Ranking por Produto e Gestor", layout="wide")
st.title("🏆 Ranking de Assessores por Produto e Gestor")

with st.expander("📘 Instruções de uso"):
    st.markdown("""
    Para usar o dashboard, você precisa importar:

    - 🟦 **Planilhas de CADASTRO** (Relatório BTG - "Base BTG")  
      *Relatórios > Categoria = Investimento > Relatório = Investimentos (D-1 e D0).*  
      Depois de filtrar, na coluna **Cliente** faça o download em xlsx **"Base BTG"**.

    - 🟨 **Planilhas de POSIÇÃO** (fundos alocados pelos clientes)  
      *Relatórios > Categoria = Investimento > Relatório = Investimentos (D-1 e D0).*  
      Depois de filtrar, na coluna **Painéis** clique em **Fundos** e baixe **somente** a tabela **"Posição - Fundos"**.

    ⚠️ Os arquivos podem ter **várias abas**; o app já lê todas automaticamente.
    ✅ Após o upload, selecione o **Gestor** e o **Produto** para ver o ranking de PL por assessor.
    """)

# -------- Helpers --------
REQ_COLS_CAD = {"Conta": "conta_cliente", "Assessor": "assessor"}
REQ_COLS_POS = {"Conta": "conta_cliente", "Produto": "produto", "Valor Líquido": "valor_liquido", "Gestor": "gestor"}

def normalize_conta(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # remove sufixo .0 de números salvos como float em Excel
    if s.endswith(".0"):
        s = s[:-2]
    # remove separadores estranhos
    s = s.replace(",", "").replace(" ", "")
    return s

def to_number_brl(x):
    """
    Converte textos com vírgula/ponto para float de forma robusta.
    Ex.: '1.234.567,89' -> 1234567.89 ; '123,45' -> 123.45 ; '1234.56' -> 1234.56
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    # Se tem vírgula como decimal, remove pontos de milhar e troca vírgula por ponto
    if "," in s and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        # caso padrão: só tira espaços
        s = s.replace(" ", "")
    try:
        return float(s)
    except:
        return np.nan

def read_all_sheets(file, rename_map, keep_cols):
    """Lê todas as abas; renomeia e mantém apenas as colunas necessárias (caso-insensível)."""
    # Lê todas as sheets como dict
    dfs = pd.read_excel(file, sheet_name=None)
    out = []
    for name, df in dfs.items():
        # normaliza nomes de colunas removendo espaços extras
        df = df.rename(columns=lambda c: str(c).strip())
        # mapeamento tolerante: procura colunas equivalentes ignorando caixa
        col_map = {}
        lower_cols = {c.lower(): c for c in df.columns}
        for orig, new in rename_map.items():
            key = orig.lower()
            if key in lower_cols:
                col_map[lower_cols[key]] = new
        df = df.rename(columns=col_map)

        # checa colunas
        missing = [v for v in rename_map.values() if v not in df.columns]
        if missing:
            # pula aba que não tem o mínimo necessário
            continue

        df = df[list(keep_cols)]
        out.append(df)
    if not out:
        return pd.DataFrame(columns=list(keep_cols))
    return pd.concat(out, ignore_index=True)

def add_medals(ranking_df):
    medalhas = ["🥇", "🥈", "🥉"]
    ranking_df = ranking_df.copy()
    ranking_df.insert(0, "🏅", "")
    for i in range(min(3, len(ranking_df))):
        ranking_df.at[i, "🏅"] = medalhas[i]
    return ranking_df

def fmt_moeda(col):
    return col.apply(lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

def to_excel_bytes(df_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, dfx in df_dict.items():
            dfx.to_excel(writer, sheet_name=sheet[:31], index=False)
    return output.getvalue()

# -------- Upload --------
cadastro_files = st.file_uploader("📄 Upload das planilhas de CADASTRO (XLSX)", type=["xlsx"], accept_multiple_files=True)
posicao_files  = st.file_uploader("📄 Upload das planilhas de POSIÇÃO (XLSX)", type=["xlsx"], accept_multiple_files=True)

if cadastro_files and posicao_files:
    try:
        # CADASTRO
        cadastro_list = []
        for file in cadastro_files:
            dfc = read_all_sheets(file, REQ_COLS_CAD, keep_cols=["conta_cliente", "assessor"])
            cadastro_list.append(dfc)
        cadastro_df = pd.concat(cadastro_list, ignore_index=True) if cadastro_list else pd.DataFrame(columns=["conta_cliente", "assessor"])

        # POSIÇÃO
        posicao_list = []
        for file in posicao_files:
            dfp = read_all_sheets(file, REQ_COLS_POS, keep_cols=["conta_cliente", "produto", "valor_liquido", "gestor"])
            posicao_list.append(dfp)
        posicao_df = pd.concat(posicao_list, ignore_index=True) if posicao_list else pd.DataFrame(columns=["conta_cliente", "produto", "valor_liquido", "gestor"])

        # ---- Saneamento de tipos ----
        cadastro_df["conta_cliente"] = cadastro_df["conta_cliente"].apply(normalize_conta)
        posicao_df["conta_cliente"]  = posicao_df["conta_cliente"].apply(normalize_conta)
        posicao_df["valor_liquido"]  = posicao_df["valor_liquido"].apply(to_number_brl)

        # remove linhas sem chaves essenciais
        cadastro_df = cadastro_df.dropna(subset=["conta_cliente", "assessor"])
        posicao_df  = posicao_df.dropna(subset=["conta_cliente", "produto", "valor_liquido", "gestor"])

        # Se houver múltiplos assessores para a mesma conta em arquivos diferentes, mantemos o último
        cadastro_df = (cadastro_df
                       .sort_index()
                       .drop_duplicates(subset=["conta_cliente"], keep="last"))

        # Merge seguro (ambos são strings)
        df = posicao_df.merge(cadastro_df, on="conta_cliente", how="left")

        # Filtros
        gestores = ["Todos"] + sorted([g for g in df["gestor"].dropna().unique()])
        gestor_sel = st.selectbox("🏢 Selecione o Gestor", gestores, index=0)

        if gestor_sel != "Todos":
            df_filtro_gestor = df[df["gestor"] == gestor_sel]
        else:
            df_filtro_gestor = df.copy()

        # Produtos com base no filtro de gestor
        produtos_unicos = sorted(df_filtro_gestor["produto"].dropna().unique())
        if not produtos_unicos:
            st.warning("Nenhum produto encontrado após o filtro. Verifique se as colunas e dados estão corretos.")
            st.stop()
        
        produtos = ["Todos"] + produtos_unicos
        produto_sel = st.selectbox("📦 Selecione o Produto (ativo)", produtos, index=0)
        
        # Aplica filtro de produto (ou todos)
        if produto_sel != "Todos":
            df_filtrado = df_filtro_gestor[df_filtro_gestor["produto"] == produto_sel].copy()
        else:
            df_filtrado = df_filtro_gestor.copy()
        
        if df_filtrado.empty:
            st.warning("Sem dados para esse filtro.")
            st.stop()
        
        # Ranking por assessor
        ranking_df = (df_filtrado.groupby("assessor", dropna=False)["valor_liquido"]
                      .sum().reset_index()
                      .sort_values(by="valor_liquido", ascending=False)
                      .reset_index(drop=True))
        
        # Medalhas
        ranking_med = add_medals(ranking_df)
        
        # Exibição
        st.subheader("🔎 Ranking de PL por Assessor")
        info_gestor = gestor_sel if gestor_sel != "Todos" else "Todos os Gestores"
        info_produto = produto_sel if produto_sel != "Todos" else "Todos os Produtos"
        st.markdown(f"**Gestor:** {info_gestor}  |  **Produto:** {info_produto}")
        
        # versão formatada para exibir
        show_df = ranking_med.copy()
        show_df["valor_liquido"] = fmt_moeda(show_df["valor_liquido"])
        show_df = show_df.rename(columns={"assessor": "Assessor", "valor_liquido": "PL (Valor Líquido)"})
        st.dataframe(show_df, use_container_width=True)
        
        # Total geral do filtro atual
        total_pl = df_filtrado["valor_liquido"].sum()
        st.markdown(f"### 💰 Total de PL em {info_produto}: {fmt_moeda(pd.Series([total_pl])).iloc[0]}")
        
        st.divider()
        st.subheader("⬇️ Baixar bases tratadas")
        
        # Downloads: ranking e base tratada (pós-merge) do filtro atual
        bytes_ranking = to_excel_bytes({"Ranking": ranking_df})
        st.download_button(
            "Baixar Ranking (xlsx)",
            data=bytes_ranking,
            file_name=f"ranking_{info_gestor}_{info_produto}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # Base tratada (apenas linhas do filtro atual), com colunas bem nomeadas
        base_tratada = df_filtrado[["conta_cliente", "assessor", "gestor", "produto", "valor_liquido"]].copy()
        base_tratada = base_tratada.rename(columns={
            "conta_cliente": "Conta",
            "assessor": "Assessor",
            "gestor": "Gestor",
            "produto": "Produto",
            "valor_liquido": "Valor Líquido"
        })
        bytes_base = to_excel_bytes({"Base Tratada": base_tratada})
        st.download_button(
            "Baixar Base Tratada (xlsx)",
            data=bytes_base,
            file_name=f"base_tratada_{info_gestor}_{info_produto}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


    except Exception as e:
        st.error(f"❌ Erro ao processar os arquivos: {e}")
        st.info("Dica: esse erro costuma acontecer quando 'Conta' vem numérica em um arquivo e texto no outro. O app agora força tudo para texto, mas verifique se os títulos das colunas estão corretos.")
else:
    st.caption("💡 Dica: você pode subir mais de um arquivo em cada campo. O app lê todas as abas de cada XLSX.")
