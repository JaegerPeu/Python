import streamlit as st
import pandas as pd

st.set_page_config(page_title="Ranking por Produto e Gestor", layout="wide")
st.title("🏆 Ranking de Assessores por Produto e Gestor")
with st.expander("📘 Instruções de uso"):
    st.markdown("""
    Para usar o dashboard, você precisa importar:

    - 🟦 **Planilhas de CADASTRO** (Relatório BTG - "Base BTG")
        
    Relatórios >  Categoria = Investimento > Relatório = Investimentos (D-1 e D0).
    Depois de filtrar, na coluna "Cliente" fazer o download em xlsx "Base BTG"

    - 🟨 **Planilhas de POSIÇÃO** (fundos alocados pelos clientes)
        
    Relatórios >  Categoria = Investimento > Relatório = Investimentos (D-1 e D0).
    Depois de filtrar na coluna "Painéis" clicar em "Fundos" e fazer o download somente da tabela "Posição - Fundos"
       
    
    
    ⚠️ As planilhas podem ser geradas a partir do seu sistema interno ou solicitadas à equipe de dados.

    ✅ Após o upload, selecione o **Gestor** e o **Produto** para ver o ranking dos assessores por PL alocado.
    """)

# Upload múltiplo
cadastro_files = st.file_uploader("📄 Upload das planilhas de CADASTRO", type=["xlsx"], accept_multiple_files=True)
posicao_files = st.file_uploader("📄 Upload das planilhas de POSIÇÃO", type=["xlsx"], accept_multiple_files=True)

if cadastro_files and posicao_files:
    try:
        # Leitura e concatenação das planilhas de cadastro
        cadastro_list = []
        for file in cadastro_files:
            df = pd.read_excel(file)
            df = df.rename(columns={"Conta": "conta_cliente", "Assessor": "assessor"})
            cadastro_list.append(df[["conta_cliente", "assessor"]])
        cadastro_df = pd.concat(cadastro_list, ignore_index=True)

        # Leitura e concatenação das planilhas de posição
        posicao_list = []
        for file in posicao_files:
            df = pd.read_excel(file)
            df = df.rename(columns={
                "Conta": "conta_cliente",
                "Produto": "produto",
                "Valor Líquido": "valor_liquido",
                "Gestor": "gestor"
            })
            posicao_list.append(df[["conta_cliente", "produto", "valor_liquido", "gestor"]])
        posicao_df = pd.concat(posicao_list, ignore_index=True)

        # Merge
        df = posicao_df.merge(cadastro_df, on="conta_cliente", how="left")
        df = df.dropna(subset=["assessor", "produto", "valor_liquido", "gestor"])

        # Filtro de Gestor
        gestores = sorted(df["gestor"].dropna().unique())
        gestor_sel = st.selectbox("🏢 Selecione o Gestor", gestores)

        df_gestor = df[df["gestor"] == gestor_sel]

        # Filtro de Produto (dentro do gestor selecionado)
        produtos = sorted(df_gestor["produto"].dropna().unique())
        produto_sel = st.selectbox("📦 Selecione o Produto (ativo)", produtos)

        df_filtrado = df_gestor[df_gestor["produto"] == produto_sel]

                # Agrupamento por assessor
        ranking_df = df_filtrado.groupby("assessor")["valor_liquido"].sum().reset_index()
        ranking_df = ranking_df.sort_values(by="valor_liquido", ascending=False).reset_index(drop=True)

        # Medalhas 🥇🥈🥉
        medalhas = ["🥇", "🥈", "🥉"]
        ranking_df.insert(0, "🏅", "")
        for i in range(min(3, len(ranking_df))):
            ranking_df.at[i, "🏅"] = medalhas[i]

        # Formatação de moeda no padrão brasileiro
        ranking_df["valor_liquido"] = ranking_df["valor_liquido"].apply(
            lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )

        # Exibição
        st.subheader("🔎 Ranking de PL por Assessor")
        st.markdown(f"**Gestor:** {gestor_sel}  |  **Produto:** {produto_sel}")
        st.dataframe(ranking_df.rename(columns={"assessor": "Assessor", "valor_liquido": "PL (Valor Líquido)"}),
                     use_container_width=True)

        # Total geral
        total_pl = df_filtrado["valor_liquido"].sum()
        total_pl_fmt = f"R$ {total_pl:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        st.markdown(f"### 💰 Total de PL nesse produto: {total_pl_fmt}")


    except Exception as e:
        st.error(f"❌ Erro ao processar os arquivos: {e}")
        



