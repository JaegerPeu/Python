import streamlit as st
import smtplib
from email.mime.text import MIMEText
from datetime import date

# Lê configurações do SMTP (Outlook) de st.secrets
SMTP_SERVER = st.secrets["smtp_server"]      # ex: "smtp.office365.com"
SMTP_PORT = int(st.secrets["smtp_port"])     # ex: 587
SMTP_USER = st.secrets["smtp_user"]          # ex: "seu.email@seu-dominio.com"
SMTP_PASSWORD = st.secrets["smtp_password"]  # senha / senha de app
EMAIL_DESTINO = st.secrets["email_destino"]  # para onde vai o pedido de TED


def enviar_email(assunto: str, corpo: str):
    msg = MIMEText(corpo, "plain")
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_DESTINO
    msg["Subject"] = assunto

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


st.title("Boletador de TED (simples)")

# --- Dados do cliente / favorecido
cliente = st.text_input("Nome do cliente (origem)")

usar_conta_existente = st.checkbox("Usar conta já cadastrada?", value=False)

if usar_conta_existente:
    favorecido = st.text_input("Nome do favorecido (cadastrado)")
    banco = st.text_input("Banco (ex: 341 - ITAÚ)")
    agencia = st.text_input("Agência")
    conta = st.text_input("Conta")
    cpf_cnpj = st.text_input("CPF/CNPJ do favorecido")
else:
    st.subheader("Novo favorecido")
    favorecido = st.text_input("Nome do favorecido")
    cpf_cnpj = st.text_input("CPF/CNPJ do favorecido")
    banco = st.text_input("Banco (ex: 341 - ITAÚ)")
    agencia = st.text_input("Agência")
    conta = st.text_input("Conta")

# --- Dados da TED
valor = st.number_input("Valor da TED (R$)", min_value=0.01, step=0.01, format="%.2f")
data_ted = st.date_input("Data da TED", value=date.today())
observacoes = st.text_area("Observações (opcional)")

if st.button("Enviar instrução de TED"):
    erros = []
    if not cliente:
        erros.append("Informe o nome do cliente.")
    if not favorecido:
        erros.append("Informe o nome do favorecido.")
    if not cpf_cnpj:
        erros.append("Informe CPF/CNPJ do favorecido.")
    if not banco:
        erros.append("Informe o banco.")
    if not agencia:
        erros.append("Informe a agência.")
    if not conta:
        erros.append("Informe a conta.")
    if valor <= 0:
        erros.append("Valor deve ser maior que zero.")

    if erros:
        st.error("Erros:\n- " + "\n- ".join(erros))
    else:
        corpo = f"""
Instrução de TED

Cliente origem: {cliente}

Favorecido: {favorecido}
CPF/CNPJ: {cpf_cnpj}

Banco: {banco}
Agência: {agencia}
Conta: {conta}

Valor: R$ {valor:,.2f}
Data da TED: {data_ted.strftime('%d/%m/%Y')}

Observações:
{observacoes or '-'}

Gerado via Boletador de TED (Streamlit simples).
"""
        try:
            enviar_email(
                assunto=f"[TED] {cliente} -> {favorecido} R$ {valor:,.2f}",
                corpo=corpo,
            )
            st.success("Instrução de TED enviada por e-mail.")
            st.code(corpo)
        except Exception as e:
            st.error(f"Falha ao enviar e-mail: {e}")
