# Assistente jurídico com RAG: cita a fonte, recusa sem base e mede alucinação

Prova de conceito de um assistente de dúvidas para estudantes da 1ª fase da OAB. Ele responde
com base no **Estatuto da Advocacia e da OAB**, no **Código de Ética e Disciplina** e em trechos
da **Constituição Federal**, sempre citando o artigo de origem. Quando o material não sustenta
uma resposta, ele **recusa** em vez de inventar.

Em conteúdo jurídico, o risco principal de um chatbot é alucinação: um artigo, prazo ou quórum
inventado, entregue a um aluno na véspera da prova. Por isso o foco desta PoC não é "fazer um
chatbot", é tornar a resposta **verificável, recusável e medida**.

> Usa somente material público (Planalto e OAB). Nenhum conteúdo pago de curso foi utilizado.

## Resultados

Recuperação (30 perguntas, `multilingual-e5-large`, top 5):

| Métrica | Resultado |
|---|---|
| hit@5 (artigo certo entre os 5 trechos) | **25/25 (100%)** |
| hit@1 (artigo certo em 1º lugar) | 21/25 (84%) |
| perguntas fora do escopo barradas pelo portão de relevância, sem chamar o LLM | 3/5 |

Calibração do portão (`MIN_SCORE`): as perguntas fora do material (CPC, Código Penal, taxa do
exame) tiveram similaridade máxima entre 0,819 e 0,830; a menor pergunta respondível teve 0,848.
O limiar foi fixado em 0,835. A margem é pequena e o dataset também, então o portão é tratado
como **filtro de custo**, não como a defesa principal: as duas perguntas restantes (premissa
falsa e súmula fora da base) passam pelo portão e precisam ser recusadas pelo modelo e pelos
guardrails. Com perguntas reais de alunos, esse limiar deve ser recalibrado.

Avaliação completa com LLM (recusa, citação, fidelidade, acerto):
`python -m rag_juridico.cli eval` gera o relatório em `reports/`.

## Como funciona

```
pergunta
   │
   ▼
[1] busca vetorial (top 5 trechos, embeddings locais)
   │
   ├── melhor similaridade < MIN_SCORE ──► recusa sem chamar o LLM (custo zero)
   ▼
[2] LLM (Claude via LangChain) responde em JSON: {answerable, answer, citations[T1..T5]}
   │
   ├── answerable = false ──► recusa
   ▼
[3] guardrails determinísticos
   │   - toda citação precisa ser um dos trechos recuperados
   │   - todo "art. N" no texto precisa existir no contexto
   ├── violação ──► bloqueia e registra
   ▼
[4] resposta + fontes (com link para o texto oficial)
   │
   ▼
[5] log estruturado em logs/interactions.jsonl (pergunta, trechos, scores, status, tokens, latência)
```

## Decisões de desenho

**Chunking por artigo, não por número de caracteres.** Cortar a cada N caracteres separa o caput
dos parágrafos e mistura artigos vizinhos, o que leva o modelo a citar o artigo errado. Aqui cada
chunk é um artigo inteiro (caput, parágrafos e incisos), com o caminho Título > Capítulo > Seção
no cabeçalho. Só artigos muito longos (CF art. 5º, EAOAB art. 7º) são subdivididos, mantendo o
rótulo do artigo em cada parte. O id do chunk é a própria referência (`EAOAB:25`, `CED:35`),
o que torna citação e avaliação triviais de checar.

**Texto revogado é descartado na ingestão.** O Planalto mantém a redação antiga tachada ao lado
da nova. Sem tratar isso, o assistente responderia com lei revogada (ex.: composição antiga da
Diretoria do Conselho Federal, art. 55). O parser remove `<strike>` e as anotações editoriais
("Incluído pela Lei...", "Vide ADIN...").

**Três camadas contra alucinação, da mais barata para a mais cara:**
1. *Portão de relevância*: se nenhum trecho passa da similaridade mínima, recusa sem chamar o LLM.
2. *Instrução + formato*: o modelo responde em JSON e pode declarar `answerable: false`.
3. *Guardrails determinísticos*: não dependem do modelo obedecer. Citação de trecho não recuperado
   ou menção a artigo ausente do contexto bloqueiam a resposta.

**Embeddings locais (fastembed/ONNX, `multilingual-e5-large`).** Sem custo por consulta, sem enviar
o material a um terceiro para vetorização, e com bom desempenho em português. O modelo é
configurável (`EMBEDDING_MODEL`).

**Vector store intercambiável.** A chain só conhece a interface `VectorStore` do LangChain.
`VECTOR_STORE=local` usa um índice em arquivo (zero infraestrutura); `VECTOR_STORE=pinecone`
usa Pinecone serverless. pgvector ou Weaviate entram em `vectorstore.py` sem tocar no resto.

**Avaliação como parte do produto, não como etapa final.** O dataset (`eval/dataset.jsonl`)
tem 30 perguntas: 25 respondíveis e 5 que **devem** ser recusadas, incluindo perguntas que o
modelo sabe responder de memória (prazo de contestação do CPC) e uma premissa falsa ("qual artigo
garante 60 dias de férias?"). O que se mede é se ele resiste a responder fora do material.

## Métricas

| Métrica | O que mede | Como |
|---|---|---|
| hit@k | a busca trouxe o artigo certo | determinístico, roda sem LLM |
| acerto de recusa | recusou quando devia, respondeu quando devia | determinístico |
| precisão de citação | citou o artigo esperado | determinístico |
| fidelidade | toda afirmação está nos trechos citados | LLM como juiz |
| acerto | resposta consistente com a referência | LLM como juiz |

O juiz recebe apenas os trechos **citados**, não todos os recuperados: uma afirmação correta,
mas apoiada num trecho que o aluno não vê como fonte, conta como não fundamentada.

## Como rodar

Requisitos: Python 3.11+ e uma chave da Anthropic (ou `ant auth login`).

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
cp .env.example .env              # preencha ANTHROPIC_API_KEY

python -m rag_juridico.cli download        # baixa as fontes públicas
python -m rag_juridico.cli ingest          # chunking + embeddings + índice
python -m rag_juridico.cli ask "Em quanto tempo prescreve a cobrança de honorários?"
python -m rag_juridico.cli eval --retrieval-only   # mede só a busca, sem custo de LLM
python -m rag_juridico.cli eval                    # avaliação completa
python -m rag_juridico.cli stats                   # resumo dos logs
pytest                                             # 16 testes offline (chunking, guardrails, chain com LLM simulado)
```

Interface de demonstração: `pip install streamlit && streamlit run app.py`.

## Estrutura

```
src/rag_juridico/
  sources.py      download e extração (HTML do Planalto, PDF da OAB), remoção de texto revogado
  chunking.py     um chunk por artigo, com título/capítulo e ids citáveis
  vectorstore.py  embeddings locais + store local ou Pinecone
  chain.py        recuperação, portão de relevância, LLM, montagem da resposta
  guardrails.py   validação determinística de citações e artigos mencionados
  logbook.py      log JSONL de cada interação e resumo de monitoramento
  evaluate.py     avaliação automatizada e relatório em reports/
eval/dataset.jsonl  30 perguntas com referência e artigo esperado
tests/              testes offline
app.py              interface Streamlit
```

## Próximos passos em produção

- Trocar o material público pelo material do curso (apostilas, questões comentadas), mantendo
  o mesmo pipeline de chunking por unidade semântica (questão + comentário, artigo + doutrina).
- Busca híbrida (vetorial + BM25) para perguntas que citam número de artigo ou termo exato.
- Logs em banco com painel: taxa de recusa por tema (mostra lacuna de conteúdo), perguntas
  bloqueadas por guardrail (mostra risco), latência e custo por aluno.
- Feedback do aluno e do monitor em cada resposta, alimentando o dataset de avaliação.
- Pré-correção de peças da 2ª fase comparando a resposta com o espelho, com a mesma lógica de
  fundamentação verificável.
