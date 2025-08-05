
import pandas as pd
import streamlit.components.v1 as components



def formatar(valor):
    if valor is None or pd.isna(valor):
        return "N/A"
    try:
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(valor)

def formatar_off(valor):
    if valor is None or pd.isna(valor):
        return "N/A"
    try:
        return f"US$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(valor)

def formatar_reais_str(valor):
    if valor is None or pd.isna(valor):
        return "N/A"
    try:
        if isinstance(valor, str):
            valor = valor.replace("R$", "").replace(".", "").replace(",", ".")
        return formatar(float(valor))
    except:
        return str(valor)


def mostrar_veiculos(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo, sheet_name="Veiculos")

    html_style = """
    <style>
    .card-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
    }
    .card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        font-family: Arial, sans-serif;
        width: 310px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    }
    .header {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 5px;
        text-align: center;
    }
    .meta {
        font-size: 13px;
        color: #333;
        margin-bottom: 4px;
        text-align: center;
    }
    </style>
    """

    html_cards = '<div class="card-container">'

    for _, row in df.iterrows():
        fundo = row['Fundo']
        grande_classe = row['Grande Classe']
        cor_fundo = "#e8f5e9" if grande_classe == "FI" else "#e3f2fd"

        try: aum_percent = f"{float(row['%AUM']) * 100:.2f}%"
        except: aum_percent = str(row['%AUM'])

        try: taxa_adm = f"{float(row['Taxa Adm']) * 100:.2f}%"
        except: taxa_adm = str(row['Taxa Adm'])

        patrimonio = formatar_reais_str(row['Patrimônio'])
        receita_total = formatar_reais_str(row['Receita Total Estimada'])
        receita_asset = formatar_reais_str(row['Receita Asset Estimada'])

        html_cards += f"""
        <div class="card" style="background-color:{cor_fundo}; border-left: 5px solid #666;">
            <div class="header">📌 {fundo}</div>
            <div class="meta">🏦 Classe: {grande_classe} | 💼 {row['Custodiante']}</div>

            <br>
            <div style="display: flex; justify-content: space-between;">
                <div style="text-align:center;">
                    <div style="font-size: 20px; font-weight: bold;">{patrimonio}</div>
                    <div style="font-size: 12px; color: #666;">Patrimônio</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size: 20px; font-weight: bold;">{aum_percent}</div>
                    <div style="font-size: 12px; color: #666;">% AUM</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size: 20px; font-weight: bold;">{taxa_adm}</div>
                    <div style="font-size: 12px; color: #666;">Taxa Adm</div>
                </div>
            </div>
            <br>
            <hr style="margin: 10px 0;">
            <div class="meta">👤 Gestor: {row['Gestor']}</div>
            <div class="meta">💸 Receita Total Estimada: {receita_total}</div>
            <div class="meta">🏦 Receita Asset Estimada: {receita_asset}</div>
        </div>
        """

    html_cards += '</div>'
    altura_estimativa = max(600, len(df) * 420)
    components.html(html_style + html_cards, height=altura_estimativa, scrolling=False)
