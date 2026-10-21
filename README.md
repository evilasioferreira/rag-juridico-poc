# Assistente jurídico com RAG: cita a fonte, recusa sem base e mede alucinação

Prova de conceito de um assistente de dúvidas para estudantes da 1ª fase da OAB. Ele responde
com base no **Estatuto da Advocacia e da OAB**, no **Código de Ética e Disciplina** e em trechos
da **Constituição Federal**, sempre citando o artigo de origem. Quando o material não sustenta
uma resposta, ele **recusa** em vez de inventar.

Em conteúdo jurídico, o risco principal de um chatbot é alucinação: um artigo, prazo ou quórum
inventado, entregue a um aluno na véspera da prova. Por isso o foco desta PoC não é "fazer um
chatbot", é tornar a resposta **verificável, recusável e medida**.

**▶ Teste a demo: [juridico-poc.streamlit.app](https://juridico-poc.streamlit.app)**
(se o app estiver hibernando, a primeira abertura leva cerca de 1 minuto)

> Usa somente material público (Planalto e OAB). Nenhum conteúdo pago de curso foi utilizado.

## Resultados

Recuperação nas 25 perguntas respondíveis (top 5), comparando modelos de embeddings e o efeito
da busca híbrida (vetorial + BM25):

| Modelo de embeddings | Tamanho | Busca | hit@5 | hit@1 |
|---|---|---|---|---|
| `potion-multilingual-128M` | 0,5 GB | vetorial | 56% | 7/25 |
| `paraphrase-multilingual-mpnet-base-v2` | 1 GB | vetorial | 88% | 12/25 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0,2 GB | vetorial | 92% | 13/25 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0,2 GB | **híbrida** | **96%** | **19/25** |
| `multilingual-e5-large` | 2,2 GB | vetorial | 100% | 21/25 |
| `multilingual-e5-large` | 2,2 GB | **híbrida** | **100%** | **21/25** |

hit@5 = o artigo certo está entre os 5 trechos entregues ao LLM; hit@1 = está em 1º lugar.

O padrão do repositório é **MiniLM + híbrida**: cabe nos limites de memória do plano gratuito do
Streamlit Community Cloud e fica a 4 pontos do modelo grande. A busca híbrida foi o que fechou a
distância: o BM25 recupera termos exatos ("procuração", "sucumbência", "prescreve") que o modelo
pequeno às vezes perde. Em servidor com 4 GB+ de RAM, `multilingual-e5-large` é a melhor escolha.

**Portão de relevância (`MIN_SCORE`).** Com o e5-large, as perguntas fora do material ficaram
entre 0,819 e 0,830 de similaridade e a menor respondível em 0,848 (limiar 0,835, barra 3 das 5).
Com o MiniLM a separação é pior: limiar 0,53, que barra 2 das 5 (Código Penal e taxa do exame)
sem nenhuma recusa indevida. Em ambos os casos o portão é tratado como **filtro de custo**, não
como a defesa principal: o que passa por ele precisa ser recusado pelo modelo e pelos
guardrails. Com perguntas reais de alunos, o limiar deve ser recalibrado.

**Verificado na demo publicada:** respostas com o artigo-fonte citado, e recusa correta nas duas
perguntas que passam pelo portão e dependem do modelo: uma de conhecimento geral fora da base
(prazo de contestação do CPC, que o modelo sabe de memória) e uma premissa falsa ("qual artigo
garante 60 dias de férias ao advogado?").

**Auditoria dos guardrails** (`python -m rag_juridico.cli audit-guardrails`, offline, com os
trechos reais do índice). As 25 respostas de referência são checadas citando os trechos
esperados e depois adulteradas de três formas que um revisor técnico testaria primeiro:

| Caso | Resultado |
|---|---|
| Respostas corretas aprovadas (sem falso bloqueio) | **25/25** |
| Norma trocada ("art. 25 do Estatuto" → "do Código de Ética") | **25/25 bloqueadas** |
| Prazo ou quórum alterado ("cinco anos" → "seis anos") | **9/9 bloqueadas** |
| Artigo inexistente acrescentado ("art. 99 do Estatuto") | **25/25 bloqueadas** |

Só 9 das 25 respostas de referência contêm prazo ou quórum, por isso o total menor na
segunda linha. A auditoria roda como teste automatizado (`tests/test_audit.py`).

Avaliação completa com LLM (recusa, citação, fidelidade, acerto) nas 30 perguntas:
`python -m rag_juridico.cli eval` gera o relatório em `reports/`.

## Como funciona

```
pergunta
   │
   ▼
[1] busca híbrida: vetorial (embeddings locais) + BM25, fundidas por RRF → top 5 trechos
   │
   ├── melhor similaridade < MIN_SCORE ──► recusa sem chamar o LLM (custo zero)
   ▼
[2] LLM (Claude via LangChain) responde em JSON: {answerable, answer, citations[T1..T5]}
   │
   ├── answerable = false ──► recusa
   ▼
[3] guardrails determinísticos
   │   - toda citação precisa ser um dos trechos recuperados
   │   - todo "art. N da norma X" precisa existir no contexto (norma + artigo)
   │   - todo prazo/quórum (dias, meses, anos, terços, %) precisa estar nos trechos citados
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
3. *Guardrails determinísticos*: não dependem do modelo obedecer. Bloqueiam a resposta quando:
   - cita um trecho que não foi recuperado, ou não cita nenhum;
   - menciona um artigo que não está no contexto **com aquela norma**. O número sozinho não basta:
     "art. 25 do Código de Ética" é bloqueado quando o contexto só tem o art. 25 do Estatuto.
     Referências internas dos trechos ("art. 34 desta Lei") são resolvidas para a norma de origem;
   - afirma um prazo ou quórum que não aparece nos **trechos citados**. "15 dias" e "quinze dias"
     são normalizados para o mesmo valor; "três anos" é bloqueado se o trecho diz "cinco anos".

   Limite conhecido: a checagem cobre artigos, normas, prazos e quóruns, que são os erros de
   maior dano em prova. Uma afirmação errada sem número nem artigo (ex.: trocar "censura" por
   "advertência") não é pega por regra; para isso existe a avaliação de fidelidade com LLM juiz.

**Embeddings locais (fastembed/ONNX).** Sem custo por consulta e sem enviar o material a um
terceiro para vetorização. O modelo é configurável (`EMBEDDING_MODEL`) e cada modelo tem seu
próprio arquivo de índice, então trocar de modelo nunca mistura vetores. A escolha entre eles
foi feita medindo, não por reputação (tabela em Resultados).

**Busca híbrida (vetorial + BM25, Reciprocal Rank Fusion).** Texto de lei tem termos exatos que
decidem a resposta. A fusão por posição (RRF) dispensa calibrar pesos entre dois scores de
escalas diferentes. O score devolvido continua sendo a similaridade de cosseno, usada pelo portão.

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
python -m rag_juridico.cli audit-guardrails        # guardrails vs. respostas adulteradas, sem LLM
pytest                                             # 27 testes offline (chunking, busca híbrida, guardrails, auditoria, chain com LLM simulado)
```

Interface web: `streamlit run app.py`.

## Demo pública

A interface (`app.py`) foi feita para ficar aberta a quem quiser testar: chat com perguntas de
exemplo (inclusive duas fora do material, para mostrar a recusa), fontes clicáveis com o texto
do artigo e um painel de uso. Para proteger a chave da API, há três limites configuráveis por
variável de ambiente:

| Variável | Padrão | Efeito |
|---|---|---|
| `MAX_QUESTION_CHARS` | 400 | tamanho máximo da pergunta |
| `SESSION_LIMIT` | 15 | perguntas por sessão de navegador |
| `DAILY_LLM_LIMIT` | 200 | chamadas ao LLM por dia, somando todos os visitantes |

Perguntas barradas pelo portão de relevância não chamam o LLM e não contam no limite diário.

**Publicação gratuita no Streamlit Community Cloud** (usa o modelo leve e o índice que já vem
no repositório, então o app não depende do site do Planalto para subir):

1. Em share.streamlit.io, entre com o GitHub e clique em **Create app > Deploy a public app from GitHub**.
2. Repositório `evilasioferreira/rag-juridico-poc`, branch `main`, arquivo `app.py`.
3. Em *Advanced settings*, escolha Python **3.12** e cole em *Secrets*:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. **Deploy.** A primeira visita leva cerca de 1 minuto (download do modelo de 220 MB).

Os logs são efêmeros (somem quando o app reinicia) e o app hiberna após alguns dias sem acesso.
Para uma demonstração, basta.

**Alternativa com Docker** (qualquer host, ou Hugging Face Spaces no plano pago): o `Dockerfile`
gera fontes e índice no build. Para o Space, `bash deploy/push_space.sh <usuario>/<space>`
depois de `hf auth login`.

## Estrutura

```
src/rag_juridico/
  sources.py      download e extração (HTML do Planalto, PDF da OAB), remoção de texto revogado
  chunking.py     um chunk por artigo, com título/capítulo e ids citáveis
  vectorstore.py  embeddings locais + store local ou Pinecone + busca híbrida
  hybrid.py       BM25 e Reciprocal Rank Fusion
  quota.py        limites de uso da demo pública
  chain.py        recuperação, portão de relevância, LLM, montagem da resposta
  guardrails.py   validação determinística de citações, norma + artigo, prazos e quóruns
  audit.py        auditoria dos guardrails com respostas adulteradas
  logbook.py      log JSONL de cada interação e resumo de monitoramento
  evaluate.py     avaliação automatizada e relatório em reports/
eval/dataset.jsonl  30 perguntas com referência e artigo esperado
tests/              testes offline
app.py              interface Streamlit
```

## Próximos passos em produção

- Trocar o material público pelo material do curso (apostilas, questões comentadas), mantendo
  o mesmo pipeline de chunking por unidade semântica (questão + comentário, artigo + doutrina).
- Reranker (cross-encoder) sobre os 20 candidatos da busca híbrida, para subir o hit@1.
- Logs em banco com painel: taxa de recusa por tema (mostra lacuna de conteúdo), perguntas
  bloqueadas por guardrail (mostra risco), latência e custo por aluno.
- Feedback do aluno e do monitor em cada resposta, alimentando o dataset de avaliação.
- Pré-correção de peças da 2ª fase comparando a resposta com o espelho, com a mesma lógica de
  fundamentação verificável.
