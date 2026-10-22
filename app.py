"""Réplica via API do Gem de manutenção da Metalúrgica Mezzalira (RAG simples).

Arquivos esperados na mesma pasta:
  mezzalira_medalhao.xlsx -> planilha da Etapa 1 (abas Prata e Log alimentam a aba Visão)
  manual.pdf            -> o manual de procedimentos (mesmo arquivo anexado ao Gem)
  instrucoes.md         -> as instruções finais do Gem (copie e cole, sem alterar)
  perguntas_teste.txt   -> as 8 perguntas de teste, uma por linha (opcional)
Chave: GEMINI_API_KEY em .streamlit/secrets.toml (ou variável de ambiente).
"""
import os
import re
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pypdf import PdfReader

PASTA = Path(__file__).parent
MODELO_RESPOSTA = os.getenv("MODELO_RESPOSTA", "gemini-2.5-flash")      # confirme o nome vigente
MODELO_EMBEDDING = os.getenv("MODELO_EMBEDDING", "gemini-embedding-001")  # idem
TAMANHO_TRECHO = 700      # caracteres por trecho
SOBREPOSICAO = 120
QTD_TRECHOS = 3
MAX_PERGUNTAS_POR_SESSAO = 30  # protege a cota gratuita da chave


def pegar_chave():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


@st.cache_resource
def pegar_cliente():
    chave = pegar_chave()
    if not chave:
        st.error("Configure GEMINI_API_KEY (secrets.toml ou variável de ambiente) para usar o chat.")
        st.stop()
    return genai.Client(api_key=chave)


def quebrar_em_trechos(texto):
    texto = re.sub(r"[ \t]+", " ", texto)
    paragrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    trechos, atual = [], ""
    for p in paragrafos:
        if len(atual) + len(p) <= TAMANHO_TRECHO:
            atual = f"{atual}\n\n{p}".strip()
        else:
            if atual:
                trechos.append(atual)
            # parágrafo gigante: fatia com sobreposição
            while len(p) > TAMANHO_TRECHO:
                trechos.append(p[:TAMANHO_TRECHO])
                p = p[TAMANHO_TRECHO - SOBREPOSICAO:]
            atual = p
    if atual:
        trechos.append(atual)
    return trechos


def gerar_embeddings(textos, tipo_tarefa):
    resposta = pegar_cliente().models.embed_content(
        model=MODELO_EMBEDDING,
        contents=textos,
        config=types.EmbedContentConfig(task_type=tipo_tarefa),
    )
    matriz = np.array([e.values for e in resposta.embeddings], dtype=np.float32)
    return matriz / np.linalg.norm(matriz, axis=1, keepdims=True)


@st.cache_resource(show_spinner="Indexando o manual...")
def montar_indice():
    leitor = PdfReader(PASTA / "manual.pdf")
    texto = "\n\n".join((pagina.extract_text() or "") for pagina in leitor.pages)
    trechos = quebrar_em_trechos(texto)
    return trechos, gerar_embeddings(trechos, "RETRIEVAL_DOCUMENT")


def buscar_trechos(pergunta):
    trechos, matriz = montar_indice()
    vetor = gerar_embeddings([pergunta], "RETRIEVAL_QUERY")[0]
    similaridades = matriz @ vetor
    melhores = np.argsort(similaridades)[::-1][:QTD_TRECHOS]
    return [(trechos[i], float(similaridades[i])) for i in melhores]


def responder(pergunta):
    achados = buscar_trechos(pergunta)
    contexto = "\n\n---\n\n".join(t for t, _ in achados)
    instrucoes = (PASTA / "instrucoes.md").read_text(encoding="utf-8")
    conteudo = (
        "TRECHOS DO MANUAL (única fonte permitida):\n"
        f"{contexto}\n\nPERGUNTA DO OPERADOR:\n{pergunta}"
    )
    max_tentativas = 3
    for tentativa in range(max_tentativas):
        try:
            saida = pegar_cliente().models.generate_content(
                model=MODELO_RESPOSTA,
                contents=conteudo,
                config=types.GenerateContentConfig(
                    system_instruction=instrucoes, temperature=0.2
                ),
            )
            return saida.text, achados
        except APIError as e:
            if tentativa < max_tentativas - 1:
                tempo_espera = (tentativa + 1) * 5
                st.warning(
                    f"Erro na API ({e.code}). Tentando novamente em {tempo_espera}s..."
                )
                time.sleep(tempo_espera)
            else:
                return f"⚠️ Erro ao consultar a API do Gemini: {e.message}", achados

def mostrar_trechos(achados):
    with st.expander("Trechos do manual usados nesta resposta"):
        for texto, nota in achados:
            st.caption(f"similaridade {nota:.2f}")
            st.text(texto)


@st.cache_data
def carregar_planilha():
    caminho = PASTA / "mezzalira_medalhao.xlsx"
    prata = pd.read_excel(caminho, sheet_name="Prata")
    prata["Data"] = pd.to_datetime(prata["Data"])
    log = pd.read_excel(caminho, sheet_name="Log")
    bronze_linhas = len(pd.read_excel(caminho, sheet_name="Bronze"))
    return prata, log, bronze_linhas


def real(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def desenhar_visao():
    prata, log, bronze_linhas = carregar_planilha()
    todas_cidades = sorted(prata["Cidade"].unique())
    todos_meses = sorted(prata["Mês"].unique())
    col1, col2 = st.columns(2)
    cidades = col1.multiselect("Cidades", todas_cidades, default=todas_cidades)
    meses = col2.multiselect("Meses", todos_meses, default=todos_meses)
    filtrada = prata[prata["Cidade"].isin(cidades) & prata["Mês"].isin(meses)]
    if filtrada.empty:
        st.info("Nenhum pedido com esses filtros.")
        return

    por_cidade = filtrada.groupby("Cidade")["Valor (R$)"].sum().sort_values(ascending=False)
    por_mes = filtrada.groupby("Mês")["Valor (R$)"].sum()
    cidade_lider = por_cidade.index[0]
    mes_da_lider = (filtrada[filtrada["Cidade"] == cidade_lider]
                    .groupby("Mês")["Valor (R$)"].sum().idxmax())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Faturamento", real(filtrada["Valor (R$)"].sum()))
    k2.metric("Pedidos", len(filtrada))
    k3.metric("Cidade líder", cidade_lider)
    k4.metric(f"Mês de pico de {cidade_lider}", mes_da_lider)

    cruzado = filtrada.groupby(["Cidade", "Mês"], as_index=False)["Valor (R$)"].sum()
    mapa = alt.Chart(cruzado).mark_rect().encode(
        x=alt.X("Mês:O"), y=alt.Y("Cidade:N", sort="-x"),
        color=alt.Color("Valor (R$):Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Cidade", "Mês", alt.Tooltip("Valor (R$):Q", format=",.0f")],
    ).properties(title="Faturamento por cidade e mês (Ouro)", height=220)
    texto = alt.Chart(cruzado).mark_text(fontSize=11).encode(
        x="Mês:O", y=alt.Y("Cidade:N", sort="-x"),
        text=alt.Text("Valor (R$):Q", format=",.0f"),
        color=alt.condition(alt.datum["Valor (R$)"] > cruzado["Valor (R$)"].max() * 0.55,
                            alt.value("white"), alt.value("black")))
    st.altair_chart(mapa + texto, width='stretch')

    g1, g2 = st.columns(2)
    g1.subheader("Por cidade")
    g1.bar_chart(por_cidade)
    g2.subheader("Por mês")
    g2.bar_chart(por_mes)

    st.subheader("Tabela do Ouro")
    ouro = filtrada.pivot_table(index="Cidade", columns="Mês", values="Valor (R$)",
                                aggfunc="sum", fill_value=0, margins=True, margins_name="Total")
    st.dataframe(ouro.style.format("R$ {:,.0f}"), width='stretch')

    st.caption(f"Dados da Prata: {bronze_linhas} linhas na Bronze viram {len(prata)} após a limpeza. "
               "O resultado depende de ler 'REC' como Recife (ver Log).")
    with st.expander("Pedidos (Prata)"):
        st.dataframe(filtrada.drop(columns=["Mês"]), width='stretch', hide_index=True)
    with st.expander("Log de correções"):
        st.dataframe(log, width='stretch', hide_index=True)


def dentro_do_limite():
    if st.session_state.perguntas_feitas >= MAX_PERGUNTAS_POR_SESSAO:
        st.warning("Limite de perguntas desta sessão atingido.")
        return False
    st.session_state.perguntas_feitas += 1
    return True


def desenhar_chat():
    for papel, texto, achados in st.session_state.historico:
        with st.chat_message(papel):
            st.markdown(texto)
            if achados:
                mostrar_trechos(achados)

    arquivo = PASTA / "perguntas_teste.txt"
    if arquivo.exists():
        lista = [l.strip() for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]
        with st.expander(f"Bateria de testes ({len(lista)} perguntas)"):
            if st.button("Rodar todas"):
                for n, p in enumerate(lista, 1):
                    if not dentro_do_limite():
                        break
                    st.markdown(f"**{n}. {p}**")
                    texto, achados = responder(p)
                    st.markdown(texto)
                    mostrar_trechos(achados)
                    time.sleep(9)  # folga para limites por minuto da camada gratuita

    pergunta = st.chat_input("Pergunte sobre os procedimentos de manutenção")
    if pergunta and dentro_do_limite():
        st.session_state.historico.append(("user", pergunta, None))
        with st.chat_message("user"):
            st.markdown(pergunta)
        with st.chat_message("assistant"):
            texto, achados = responder(pergunta)
            st.markdown(texto)
            mostrar_trechos(achados)
        st.session_state.historico.append(("assistant", texto, achados))


st.set_page_config(page_title="Mezzalira · Dados e Manutenção", page_icon="🔧", layout="wide")
st.title("Metalúrgica Mezzalira")
st.session_state.setdefault("perguntas_feitas", 0)
st.session_state.setdefault("historico", [])

aba_visao, aba_chat = st.tabs(["📊 Visão dos pedidos", "🔧 Assistente de manutenção"])
with aba_visao:
    desenhar_visao()
with aba_chat:
    st.caption("Versão API (RAG) do Gem. Responde só com base no manual de manutenção.")
    desenhar_chat()
