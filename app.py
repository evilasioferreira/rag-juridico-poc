"""Interface de demonstração (para o vídeo). Rode com: streamlit run app.py"""

import streamlit as st

from rag_juridico.chain import ask
from rag_juridico.logbook import summarize_logs

STATUS_LABEL = {
    "respondida": "Respondida com fundamento",
    "recusada_sem_contexto": "Recusada: nenhum trecho relevante no material",
    "recusada_pelo_modelo": "Recusada: o material não sustenta uma resposta",
    "bloqueada_guardrail": "Bloqueada: citação sem fundamento detectada",
}

st.set_page_config(page_title="Assistente OAB (PoC)", page_icon="⚖️")
st.title("Assistente de dúvidas OAB")
st.caption("Responde só com base no Estatuto da OAB, Código de Ética e CF. Sem base, recusa.")

question = st.text_input("Sua dúvida", placeholder="Ex.: Em quanto tempo prescreve a cobrança de honorários?")
if question:
    with st.spinner("Consultando o material..."):
        ans = ask(question)
    (st.success if ans.status == "respondida" else st.warning)(STATUS_LABEL[ans.status])
    st.write(ans.answer)
    for c in ans.citations:
        with st.expander(f"Fonte: {c['label']}"):
            st.text(c["text"])
            st.markdown(f"[Texto oficial]({c['source_url']})")
    with st.expander("Bastidores (o que foi recuperado e registrado)"):
        st.json({k: v for k, v in ans.to_dict().items() if k not in ("citations", "raw_model_answer")})

with st.sidebar:
    st.subheader("Monitoramento")
    st.json(summarize_logs())
