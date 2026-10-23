"""Painel da Metalúrgica Mezzalira: visão dos pedidos e assistente de manutenção.

Arquivos esperados na mesma pasta:
  mezzalira_medalhao.xlsx  planilha de pedidos tratada
  manual.pdf               manual de procedimentos de manutenção
  instrucoes.md            instruções do assistente
Chave: GEMINI_API_KEY em .streamlit/secrets.toml (ou variável de ambiente).
"""
import os
import re
import sys
import time
import traceback
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

PASTA = Path(__file__).parent
MODELO_RESPOSTA = os.getenv("MODELO_RESPOSTA", "gemini-2.5-flash")
MODELO_EMBEDDING = os.getenv("MODELO_EMBEDDING", "gemini-embedding-001")
TAMANHO_TRECHO = 700
SOBREPOSICAO = 120
QTD_TRECHOS = 3
MAX_PERGUNTAS_POR_SESSAO = 40
MENSAGEM_INDISPONIVEL = "Não consegui responder agora. Tente novamente em instantes."

ACENTO = "#2563EB"
NEUTRO = "#94A3B8"
CORES_CIDADE = {
    "Recife": "#2563EB",
    "Caruaru": "#F59E0B",
    "Olinda": "#10B981",
    "Jaboatão dos Guararapes": "#8B5CF6",
    "Petrolina": "#EC4899",
}
MESES_ABREV = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun",
               7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}


# ---------------------------------------------------------------- assistente
def pegar_chave():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


@st.cache_resource
def pegar_cliente():
    chave = pegar_chave()
    if not chave:
        raise RuntimeError("GEMINI_API_KEY não configurada")
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


@st.cache_resource(show_spinner="Carregando o assistente...")
def montar_indice():
    leitor = PdfReader(PASTA / "manual.pdf")
    texto = "\n\n".join((pagina.extract_text() or "") for pagina in leitor.pages)
    trechos = quebrar_em_trechos(texto)
    return trechos, gerar_embeddings(trechos, "RETRIEVAL_DOCUMENT")


def buscar_trechos(pergunta):
    trechos, matriz = montar_indice()
    vetor = gerar_embeddings([pergunta], "RETRIEVAL_QUERY")[0]
    melhores = np.argsort(matriz @ vetor)[::-1][:QTD_TRECHOS]
    return [trechos[i] for i in melhores]


def responder(pergunta):
    trechos = buscar_trechos(pergunta)
    contexto = "\n\n---\n\n".join(trechos)
    instrucoes = (PASTA / "instrucoes.md").read_text(encoding="utf-8")
    conteudo = (
        "TRECHOS DO MANUAL (única fonte permitida):\n"
        f"{contexto}\n\nPERGUNTA DO OPERADOR:\n{pergunta}"
    )
    saida = pegar_cliente().models.generate_content(
        model=MODELO_RESPOSTA,
        contents=conteudo,
        config=types.GenerateContentConfig(system_instruction=instrucoes, temperature=0.2),
    )
    return saida.text, trechos


def responder_com_seguranca(pergunta, tentativas=4):
    for tentativa in range(1, tentativas + 1):
        try:
            return responder(pergunta)
        except Exception:
            traceback.print_exc(file=sys.stderr)  # detalhe técnico vai só para o log do servidor
            if tentativa < tentativas:
                time.sleep(2 * tentativa)  # espera 2s, 4s, 6s antes de tentar de novo
    return MENSAGEM_INDISPONIVEL, []


def mostrar_trechos(trechos):
    if not trechos:
        return
    with st.expander("Trechos do manual consultados"):
        for trecho in trechos:
            st.text(trecho)


def dentro_do_limite():
    if st.session_state.perguntas_feitas >= MAX_PERGUNTAS_POR_SESSAO:
        st.warning("Você atingiu o limite de perguntas desta sessão. Atualize a página para recomeçar.")
        return False
    st.session_state.perguntas_feitas += 1
    return True


def desenhar_chat():
    st.caption("As respostas vêm somente do manual de manutenção da Mezzalira.")

    arquivo = PASTA / "perguntas_teste.txt"
    if arquivo.exists():
        lista = [l.strip() for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]
        with st.expander(f"Perguntas de teste ({len(lista)})"):
            if st.button(f"Rodar as {len(lista)} perguntas"):
                for n, pergunta in enumerate(lista, 1):
                    if not dentro_do_limite():
                        break
                    st.markdown(f"**{n}. {pergunta}**")
                    texto, trechos = responder_com_seguranca(pergunta)
                    st.markdown(texto)
                    mostrar_trechos(trechos)
                    time.sleep(2)  # folga para limites por minuto da camada gratuita

    for papel, texto, trechos in st.session_state.historico:
        with st.chat_message(papel):
            st.markdown(texto)
            mostrar_trechos(trechos)

    pergunta = st.chat_input("Pergunte sobre os procedimentos de manutenção")
    if pergunta and dentro_do_limite():
        st.session_state.historico.append(("user", pergunta, None))
        with st.chat_message("user"):
            st.markdown(pergunta)
        with st.chat_message("assistant"):
            with st.spinner("Consultando o manual..."):
                texto, trechos = responder_com_seguranca(pergunta)
            st.markdown(texto)
            mostrar_trechos(trechos)
        st.session_state.historico.append(("assistant", texto, trechos))


# ------------------------------------------------------------------ pedidos
def rotulo_mes(mes_ano):
    """'03/2026' (MM/AAAA, como na planilha) -> 'mar/26'."""
    mes, ano = mes_ano.split("/")
    return f"{MESES_ABREV[int(mes)]}/{ano[2:]}"


def chave_mes(mes_ano):
    mes, ano = mes_ano.split("/")
    return (int(ano), int(mes))


def real(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def real_md(valor):
    # st.markdown trata dois "$" como fórmula; o cifrão precisa de barra
    return real(valor).replace("$", "\\$")


def real_inteiro(valor):
    return f"{valor:,.0f}".replace(",", ".")


def compacto(valor):
    return f"R$ {valor / 1000:.1f} mil".replace(".", ",")


@st.cache_data
def carregar_pedidos():
    pedidos = pd.read_excel(PASTA / "mezzalira_metalurgica.xlsx", sheet_name="Prata")
    pedidos["Data"] = pd.to_datetime(pedidos["Data"])
    pedidos["Mês rótulo"] = pedidos["Mês"].map(rotulo_mes)
    return pedidos


def grafico_ranking(por_cidade):
    tabela = por_cidade.reset_index()
    tabela.columns = ["Cidade", "Valor"]
    tabela["rotulo"] = tabela["Valor"].map(compacto)
    tabela["lider"] = tabela.index == 0
    base = alt.Chart(tabela).encode(
        y=alt.Y("Cidade:N", sort=alt.EncodingSortField("Valor", order="descending"),
                title=None, axis=alt.Axis(labelLimit=220, labelFontSize=13)))
    barras = base.mark_bar(cornerRadiusEnd=4, size=26).encode(
        x=alt.X("Valor:Q", axis=None),
        color=alt.condition("datum.lider", alt.value(ACENTO), alt.value(NEUTRO)),
        tooltip=["Cidade", alt.Tooltip("Valor:Q", format=",.0f", title="R$")])
    textos = base.mark_text(align="left", dx=6, fontSize=13, fontWeight="bold").encode(
        x="Valor:Q", text="rotulo:N")
    return (barras + textos).properties(height=44 * len(tabela)).configure_view(stroke=None)


def grafico_mensal(filtrada, ordem_meses):
    por_cidade_mes = filtrada.groupby(["Mês rótulo", "Cidade"], as_index=False)["Valor (R$)"].sum()
    totais = filtrada.groupby("Mês rótulo", as_index=False)["Valor (R$)"].sum()
    totais["rotulo"] = totais["Valor (R$)"].map(compacto)
    cidades = [c for c in CORES_CIDADE if c in set(por_cidade_mes["Cidade"])]
    cores = alt.Scale(domain=cidades, range=[CORES_CIDADE[c] for c in cidades])
    eixo_x = alt.X("Mês rótulo:N", sort=ordem_meses, title=None,
                   axis=alt.Axis(labelAngle=0, labelFontSize=13))
    colunas = alt.Chart(por_cidade_mes).mark_bar(size=54).encode(
        x=eixo_x,
        y=alt.Y("Valor (R$):Q", title=None,
                axis=alt.Axis(labelExpr="'R$ ' + datum.value / 1000 + ' mil'", tickCount=5)),
        color=alt.Color("Cidade:N", scale=cores,
                        legend=alt.Legend(orient="bottom", title=None, columns=3)),
        tooltip=["Mês rótulo", "Cidade", alt.Tooltip("Valor (R$):Q", format=",.0f", title="R$")])
    rotulos = alt.Chart(totais).mark_text(dy=-8, fontSize=13, fontWeight="bold").encode(
        x=eixo_x, y="Valor (R$):Q", text="rotulo:N")
    return (colunas + rotulos).properties(height=340).configure_view(stroke=None)


def grafico_mapa_calor(filtrada, ordem_cidades, ordem_meses):
    cruzado = filtrada.groupby(["Cidade", "Mês rótulo"], as_index=False)["Valor (R$)"].sum()
    grade = pd.MultiIndex.from_product([ordem_cidades, ordem_meses], names=["Cidade", "Mês rótulo"])
    cruzado = cruzado.set_index(["Cidade", "Mês rótulo"]).reindex(grade, fill_value=0).reset_index()
    maximo = cruzado["Valor (R$)"].max()
    cruzado["texto"] = cruzado["Valor (R$)"].map(lambda v: real_inteiro(v) if v else "—")
    cruzado["claro"] = cruzado["Valor (R$)"] > maximo * 0.55
    eixo_y = alt.Y("Cidade:N", sort=ordem_cidades, title=None)
    eixo_x = alt.X("Mês rótulo:N", sort=ordem_meses, title=None)
    celulas = alt.Chart(cruzado).mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
        x=eixo_x, y=eixo_y,
        color=alt.Color("Valor (R$):Q", scale=alt.Scale(scheme="blues", domain=[0, maximo]), legend=None),
        tooltip=["Cidade", "Mês rótulo", alt.Tooltip("Valor (R$):Q", format=",.0f", title="R$")])
    numeros = alt.Chart(cruzado).mark_text(fontSize=13).encode(
        x=eixo_x, y=eixo_y, text="texto:N",
        color=alt.condition("datum.claro", alt.value("white"), alt.value("#1E293B")))
    pico = cruzado.loc[[cruzado["Valor (R$)"].idxmax()]]
    destaque = alt.Chart(pico).mark_rect(fill=None, stroke="#F59E0B", strokeWidth=3, cornerRadius=3).encode(
        x=eixo_x, y=eixo_y)
    return ((celulas + numeros + destaque)
            .properties(height=48 * len(ordem_cidades))
            .configure_view(stroke=None)
            .configure_axisX(orient="top", labelAngle=0, labelFontSize=13, domain=False, ticks=False)
            .configure_axisY(labelLimit=220, labelFontSize=13, domain=False, ticks=False))


def desenhar_pedidos():
    pedidos = carregar_pedidos()
    rotulos_ordenados = [rotulo_mes(m) for m in sorted(pedidos["Mês"].unique(), key=chave_mes)]

    st.subheader("Quem mais comprou, e quando?")
    primeiro, ultimo = st.select_slider(
        "Período", options=rotulos_ordenados, value=(rotulos_ordenados[0], rotulos_ordenados[-1]))
    rotulos_ativos = rotulos_ordenados[rotulos_ordenados.index(primeiro):rotulos_ordenados.index(ultimo) + 1]
    filtrada = pedidos[pedidos["Mês rótulo"].isin(rotulos_ativos)]

    por_cidade = filtrada.groupby("Cidade")["Valor (R$)"].sum().sort_values(ascending=False)
    por_mes = filtrada.groupby("Mês rótulo")["Valor (R$)"].sum()
    total = filtrada["Valor (R$)"].sum()
    cidade_lider = por_cidade.index[0]
    mes_da_lider = (filtrada[filtrada["Cidade"] == cidade_lider]
                    .groupby("Mês rótulo")["Valor (R$)"].sum())
    mes_pico_lider = mes_da_lider.idxmax()
    mes_pico_geral = por_mes.idxmax()

    with st.container(border=True):
        st.markdown(
            f"### {cidade_lider} foi a cidade que mais comprou: {real_md(por_cidade.iloc[0])}\n"
            f"Isso representa **{por_cidade.iloc[0] / total:.0%}** do faturamento do período. "
            f"O mês de maior compra dela foi **{mes_pico_lider}** ({real_md(mes_da_lider.max())}). "
            f"No total geral, o mês mais forte foi **{mes_pico_geral}** ({real_md(por_mes.max())}).")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Faturamento", real(total))
    k2.metric("Pedidos", len(filtrada))
    k3.metric("Ticket médio", real(total / len(filtrada)))
    k4.metric("Mês mais forte", mes_pico_geral)

    st.markdown("#### Faturamento por cidade")
    st.caption("Total do período. A barra azul é a cidade que mais comprou.")
    st.altair_chart(grafico_ranking(por_cidade), width="stretch")

    st.markdown("#### Faturamento mês a mês")
    st.caption("Cada coluna é um mês, dividida por cidade. O número no topo é o total do mês.")
    st.altair_chart(grafico_mensal(filtrada, rotulos_ativos), width="stretch")

    st.markdown("#### Faturamento por cidade e mês")
    st.caption("Valores em R$. A moldura laranja marca o maior valor.")
    st.altair_chart(grafico_mapa_calor(filtrada, list(por_cidade.index), rotulos_ativos), width="stretch")

    with st.expander("Ver tabela de faturamento por cidade e mês"):
        tabela = filtrada.pivot_table(index="Cidade", columns="Mês rótulo", values="Valor (R$)",
                                      aggfunc="sum", fill_value=0, margins=True, margins_name="Total")
        tabela = tabela[[m for m in rotulos_ativos if m in tabela.columns] + ["Total"]]
        st.dataframe(tabela.style.format(real_inteiro), width="stretch")
    with st.expander("Ver lista de pedidos"):
        st.dataframe(filtrada.drop(columns=["Mês", "Mês rótulo"]), width="stretch", hide_index=True,
                     column_config={"Data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                                    "Valor (R$)": st.column_config.NumberColumn("Valor (R$)", format="R$ %.2f")})


# ------------------------------------------------------------------- página
st.set_page_config(page_title="Metalúrgica Mezzalira", page_icon="🔧", layout="wide")
st.title("Metalúrgica Mezzalira")
st.session_state.setdefault("perguntas_feitas", 0)
st.session_state.setdefault("historico", [])

SECAO_PEDIDOS = "📊 Pedidos"
SECAO_ASSISTENTE = "🔧 Assistente de manutenção"
secao = st.segmented_control(
    "Seção", [SECAO_PEDIDOS, SECAO_ASSISTENTE], default=SECAO_PEDIDOS,
    key="secao", label_visibility="collapsed")
if secao == SECAO_ASSISTENTE:
    desenhar_chat()
else:
    desenhar_pedidos()