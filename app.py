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


ACENTO = "#2563EB"
NEUTRO = "#94A3B8"
COR_REC = "#CBD5E1"
CORES_CIDADE = {
    "Recife": "#2563EB",
    "Caruaru": "#F59E0B",
    "Olinda": "#10B981",
    "Jaboatão dos Guararapes": "#8B5CF6",
    "Petrolina": "#EC4899",
    "REC (sem cidade confirmada)": COR_REC,
}
MESES_ABREV = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun",
               7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}
NOME_REC_SOLTO = "REC (sem cidade confirmada)"


def rotulo_mes(ano_mes):
    ano, mes = ano_mes.split("-")
    return f"{MESES_ABREV[int(mes)]}/{ano[2:]}"


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
def carregar_planilha():
    caminho = PASTA / "mezzalira_medalhao.xlsx"
    bronze = pd.read_excel(caminho, sheet_name="Bronze", dtype=object)
    prata = pd.read_excel(caminho, sheet_name="Prata")
    log = pd.read_excel(caminho, sheet_name="Log")
    prata["Data"] = pd.to_datetime(prata["Data"])

    # grafia original da cidade, para o botão "tratar REC como Recife"
    original = bronze.dropna(subset=["Pedido"]).drop_duplicates("Pedido")[["Pedido", "Cidade"]]
    original["Pedido"] = original["Pedido"].astype(int)
    prata = prata.merge(original.rename(columns={"Cidade": "Cidade original"}),
                        on="Pedido", how="left")
    prata["Mês rótulo"] = prata["Mês"].map(rotulo_mes)

    # números do tratamento, calculados direto da Bronze
    sem_vazias = bronze.dropna(how="all")
    sem_duplicadas = sem_vazias.drop_duplicates()
    resumo = {
        "bronze": len(bronze),
        "vazias": len(bronze) - len(sem_vazias),
        "duplicadas": len(sem_vazias) - len(sem_duplicadas),
        "prata": len(sem_duplicadas),
        "datas_texto": int(sem_duplicadas["Data"].map(lambda x: isinstance(x, str)).sum()),
        "cidades_variantes": int(sem_duplicadas["Cidade"].isin(["recife", "REC"]).sum()),
        "valores_texto": int(sem_duplicadas["Valor (R$)"].map(lambda x: isinstance(x, str)).sum()),
    }
    return prata, log, resumo


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
    eixo_x = alt.X("Mês rótulo:N", sort=ordem_meses, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13))
    colunas = alt.Chart(por_cidade_mes).mark_bar(size=54).encode(
        x=eixo_x,
        y=alt.Y("Valor (R$):Q", title=None,
                axis=alt.Axis(labelExpr="'R$ ' + datum.value / 1000 + ' mil'", tickCount=5)),
        color=alt.Color("Cidade:N", scale=cores, legend=alt.Legend(orient="bottom", title=None, columns=3)),
        tooltip=["Mês rótulo", "Cidade", alt.Tooltip("Valor (R$):Q", format=",.0f", title="R$")])
    rotulos = alt.Chart(totais).mark_text(dy=-8, fontSize=13, fontWeight="bold").encode(
        x=eixo_x, y="Valor (R$):Q", text="rotulo:N")
    return (colunas + rotulos).properties(height=340).configure_view(stroke=None)


def grafico_mapa_calor(filtrada, ordem_cidades, ordem_meses):
    cruzado = filtrada.groupby(["Cidade", "Mês rótulo"], as_index=False)["Valor (R$)"].sum()
    grade = pd.MultiIndex.from_product([ordem_cidades, ordem_meses], names=["Cidade", "Mês rótulo"])
    cruzado = (cruzado.set_index(["Cidade", "Mês rótulo"]).reindex(grade, fill_value=0).reset_index())
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


def desenhar_visao():
    prata, log, resumo = carregar_planilha()
    meses_ordenados = sorted(prata["Mês"].unique())
    rotulos_ordenados = [rotulo_mes(m) for m in meses_ordenados]

    st.subheader("Quem mais comprou da Metalúrgica Mezzalira, e quando?")
    col_periodo, col_rec = st.columns([3, 2], vertical_alignment="bottom")
    primeiro, ultimo = col_periodo.select_slider(
        "Período", options=rotulos_ordenados, value=(rotulos_ordenados[0], rotulos_ordenados[-1]))
    tratar_rec = col_rec.toggle("Tratar 'REC' como Recife", value=True,
                                help="Decisão registrada no Log. Desligue para ver como a resposta muda.")

    dados = prata.copy()
    if not tratar_rec:
        dados.loc[dados["Cidade original"] == "REC", "Cidade"] = NOME_REC_SOLTO
    i0, i1 = rotulos_ordenados.index(primeiro), rotulos_ordenados.index(ultimo)
    rotulos_ativos = rotulos_ordenados[i0:i1 + 1]
    filtrada = dados[dados["Mês rótulo"].isin(rotulos_ativos)]

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
            f"### {cidade_lider} mais comprou: {real_md(por_cidade.iloc[0])}\n"
            f"Isso é **{por_cidade.iloc[0] / total:.0%}** do faturamento do período. "
            f"O mês de maior compra dela foi **{mes_pico_lider}** ({real_md(mes_da_lider.max())}). "
            f"No total geral, o mês mais forte foi **{mes_pico_geral}** ({real_md(por_mes.max())}).")
        if not tratar_rec:
            st.warning("Com 'REC' fora de Recife, os 5 pedidos desse código ficam sem cidade "
                       "e a liderança muda. É por isso que a decisão consta no Log como dúvida com impacto.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Faturamento", real(total))
    k2.metric("Pedidos", len(filtrada))
    k3.metric("Ticket médio", real(total / len(filtrada)))
    k4.metric("Mês mais forte", mes_pico_geral)

    st.markdown("#### Ranking de cidades")
    st.caption("Faturamento total no período. A barra azul é a cidade líder.")
    st.altair_chart(grafico_ranking(por_cidade), width="stretch")

    st.markdown("#### Faturamento mês a mês")
    st.caption("Cada coluna é um mês, dividida por cidade. O número em cima é o total do mês.")
    st.altair_chart(grafico_mensal(filtrada, rotulos_ativos), width="stretch")

    st.markdown("#### Cidade × mês (tabela Ouro)")
    st.caption("Valores em R$. A moldura laranja marca o maior valor da tabela.")
    st.altair_chart(grafico_mapa_calor(filtrada, list(por_cidade.index), rotulos_ativos), width="stretch")

    with st.container(border=True):
        st.markdown("#### Do dado bagunçado ao dado útil")
        st.caption("Cada número abaixo é calculado da aba Bronze, que nunca foi alterada.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linhas na Bronze", resumo["bronze"])
        c2.metric("Linhas vazias", f"−{resumo['vazias']}")
        c3.metric("Pedidos duplicados", f"−{resumo['duplicadas']}")
        c4.metric("Linhas na Prata", resumo["prata"])
        st.markdown(
            f"Na Prata também foram padronizadas **{resumo['datas_texto']} datas** escritas como texto, "
            f"**{resumo['cidades_variantes']} grafias de Recife** ('recife' e 'REC') e "
            f"**{resumo['valores_texto']} valores** digitados como texto (ex.: 'R$ 2.240,00').")

    with st.expander("Tabela do Ouro (cidade × mês, em R$)"):
        ouro = filtrada.pivot_table(index="Cidade", columns="Mês rótulo", values="Valor (R$)",
                                    aggfunc="sum", fill_value=0, margins=True, margins_name="Total")
        ouro = ouro[[m for m in rotulos_ativos if m in ouro.columns] + ["Total"]]
        st.dataframe(ouro.style.format(real_inteiro), width="stretch")
    with st.expander("Pedidos da Prata"):
        st.dataframe(filtrada.drop(columns=["Mês", "Mês rótulo", "Cidade original"]),
                     width="stretch", hide_index=True)
    with st.expander("Log de correções"):
        st.dataframe(log, width="stretch", hide_index=True)

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
