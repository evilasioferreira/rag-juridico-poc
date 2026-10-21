"""Interface web da PoC. Rode com: streamlit run app.py"""

import logging
import sys
from pathlib import Path

import streamlit as st

# No Streamlit Community Cloud o pacote não é instalado com `pip install -e .`.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_juridico import quota
from rag_juridico.chain import Answer, ask
from rag_juridico.logbook import summarize_logs
from rag_juridico.sources import load_sources

STATUS = {
    "respondida": ("ok", "Respondida com fundamento no material"),
    "recusada_sem_contexto": ("warn", "Recusada: nenhum trecho relevante no material"),
    "recusada_pelo_modelo": ("warn", "Recusada: o material não sustenta uma resposta"),
    "bloqueada_guardrail": ("block", "Bloqueada: a resposta citava algo fora do material"),
}

EXAMPLES = [
    "Em quanto tempo prescreve a cobrança de honorários?",
    "O advogado pode atuar sem procuração em caso de urgência?",
    "Quais são as sanções disciplinares do Estatuto?",
    "Qual o prazo de contestação no procedimento comum do CPC?",
    "Qual artigo do Estatuto garante 60 dias de férias ao advogado?",
]

CSS = """
<style>
:root { --ok: #1f7a4d; --warn: #9a6700; --block: #b42318; }
.block-container { max-width: 820px; padding-top: 2rem; }
.hero h1 { font-size: 1.9rem; margin-bottom: .2rem; }
.hero p { opacity: .75; margin: 0 0 .8rem 0; }
.pills { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1rem; }
.pill { font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
        background: rgba(127,127,127,.12); }
.badge { display: inline-block; font-size: .75rem; font-weight: 600; padding: .15rem .55rem;
         border-radius: 6px; margin-bottom: .4rem; color: #fff; }
.badge.ok { background: var(--ok); } .badge.warn { background: var(--warn); }
.badge.block { background: var(--block); }
.meta { font-size: .75rem; opacity: .6; margin-top: .3rem; }
</style>
"""

st.set_page_config(page_title="Assistente OAB · PoC", page_icon="⚖️", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []  # [(pergunta, Answer | str)]
    st.session_state.count = 0


@st.cache_resource(show_spinner="Carregando o índice do material (a primeira visita leva cerca de 1 minuto)...")
def warm_up() -> bool:
    from rag_juridico.vectorstore import get_store

    try:
        get_store()
    except FileNotFoundError:
        # Índice ausente (ex.: troca de modelo): gera a partir das fontes públicas.
        from rag_juridico.chunking import build_documents
        from rag_juridico.config import settings
        from rag_juridico.sources import download_all
        from rag_juridico.vectorstore import build_store

        if not settings.raw_dir.exists():
            download_all()
        build_store(build_documents())
        get_store.cache_clear()
    return True


def render_answer(ans: Answer | str) -> None:
    if isinstance(ans, str):  # bloqueio de cota ou erro
        st.info(ans)
        return
    kind, label = STATUS[ans.status]
    st.markdown(f'<span class="badge {kind}">{label}</span>', unsafe_allow_html=True)
    st.write(ans.answer)
    for c in ans.citations:
        with st.expander(f"📖 {c['label']} · {c['source_title']}"):
            st.text(c["text"].split("\n", 1)[-1])
            st.markdown(f"[Ver texto oficial]({c['source_url']})")
    top = ans.retrieved[0]["score"] if ans.retrieved else 0
    st.markdown(
        f'<div class="meta">{ans.latency_ms / 1000:.1f}s · melhor similaridade {top:.3f} · '
        f'{len(ans.retrieved)} trechos consultados</div>',
        unsafe_allow_html=True,
    )


def handle(question: str) -> None:
    question = question.strip()
    if not question:
        return
    blocked = quota.check(question, st.session_state.count)
    if blocked:
        st.session_state.history.append((question, blocked))
        return
    st.session_state.count += 1
    try:
        result: Answer | str = ask(question)
    except Exception:
        logging.exception("falha ao responder: %s", question)
        st.session_state.count -= 1  # falha nossa não consome a cota do visitante
        result = "Não consegui consultar o modelo agora. Tente novamente em instantes."
    st.session_state.history.append((question, result))


# Cabeçalho
st.markdown(
    """
<div class="hero">
  <h1>⚖️ Assistente de dúvidas OAB</h1>
  <p>Prova de conceito: responde só com base no Estatuto da OAB, no Código de Ética e na
  Constituição, cita o artigo de origem e recusa quando o material não sustenta a resposta.</p>
</div>
<div class="pills">
  <span class="pill">📚 Base fechada: 3 normas públicas</span>
  <span class="pill">🔎 Cita o artigo-fonte</span>
  <span class="pill">🛑 Recusa sem base</span>
  <span class="pill">📊 Toda resposta é registrada e medida</span>
</div>
""",
    unsafe_allow_html=True,
)

warm_up()

# Exemplos (só quando a conversa está vazia)
if not st.session_state.history:
    st.caption("Experimente uma pergunta (as duas últimas estão fora do material, de propósito):")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            with st.spinner("Consultando o material..."):
                handle(ex)
            st.rerun()

for question, ans in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant", avatar="⚖️"):
        render_answer(ans)

if prompt := st.chat_input("Digite sua dúvida sobre o Estatuto, o Código de Ética ou a CF",
                           max_chars=quota.MAX_QUESTION_CHARS):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Consultando o material..."):
            handle(prompt)
    st.rerun()  # redesenha pelo histórico (e esconde os exemplos)

# Barra lateral
with st.sidebar:
    st.subheader("Como funciona")
    st.markdown(
        """
1. **Busca** os 5 artigos mais parecidos com a pergunta.
2. **Portão de relevância**: se nada for parecido o bastante, recusa sem chamar a IA.
3. **IA (Claude)** responde usando só esses trechos e cita quais usou.
4. **Verificação automática**: se a resposta citar um trecho ou artigo que não foi
   recuperado, ela é bloqueada.
5. **Registro** de cada interação para medir qualidade.
"""
    )
    st.subheader("Material consultado")
    for src in load_sources():
        st.markdown(f"- [{src['title']}]({src['url']})")
    st.subheader("Uso desta demonstração")
    stats = summarize_logs()
    if stats.get("total"):
        c1, c2 = st.columns(2)
        c1.metric("Perguntas", stats["total"])
        c2.metric("Taxa de recusa", f"{stats['taxa_recusa']:.0%}")
        c1.metric("Latência p50", f"{stats['latencia_p50_ms'] / 1000:.1f}s")
        c2.metric("Bloqueios", stats["por_status"].get("bloqueada_guardrail", 0))
    restante = quota.SESSION_LIMIT - st.session_state.count
    st.caption(f"Você ainda pode fazer {restante} perguntas nesta sessão.")
    if st.session_state.history and st.button("Limpar conversa"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption(
        "Ferramenta de estudo, não substitui orientação jurídica. "
        "[Código no GitHub](https://github.com/evilasioferreira/rag-juridico-poc)"
    )
