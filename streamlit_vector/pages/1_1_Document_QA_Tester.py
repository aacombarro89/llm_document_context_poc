import json
import streamlit as st
from utils import strtobool
from common import generate_index, query_index

from dotenv import load_dotenv
import langchain

# Related to: https://github.com/hwchase17/langchain/issues/4164
langchain.verbose = False

load_dotenv()  # take environment variables from .env.

nodes_retrieved = []

query_params = st.experimental_get_query_params()

# Stored current state in session state
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = query_params.get(
        'embedding_model', ['text-ada-002'])[0]

if 'llm_model' not in st.session_state:
    st.session_state.llm_model = query_params.get(
        'llm_model', ['gpt-3.5-turbo'])[0]

if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = int(query_params.get(
        'chunk_size', [1000])[0])

if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = int(query_params.get(
        'chunk_overlap', [100])[0])

if 'text_splitter' not in st.session_state:
    st.session_state.text_splitter = query_params.get(
        'text_splitter', ['TokenTextSplitter'])[0]

if 'use_title_extractor' not in st.session_state:
    st.session_state.use_title_extractor = strtobool(query_params.get(
        'use_title_extractor', ['False'])[0])

if 'nodes_for_title_extraction' not in st.session_state:
    st.session_state.nodes_for_title_extraction = int(query_params.get(
        'nodes_for_title_extraction', [5])[0])

if 'use_summary_extractor' not in st.session_state:
    st.session_state.use_summary_extractor = strtobool(query_params.get(
        'use_summary_extractor', ['False'])[0])

if 'summary_nodes' not in st.session_state:
    st.session_state.summary_nodes = query_params.get(
        'summary_nodes', ['self'])

if 'use_questions_answered_extractor' not in st.session_state:
    st.session_state.use_questions_answered_extractor = strtobool(query_params.get(
        'use_questions_answered_extractor', ['False'])[0])

if 'questions_answered_number' not in st.session_state:
    st.session_state.questions_answered_number = int(query_params.get(
        'questions_answered_number', [5])[0])

if 'use_keyword_extractor' not in st.session_state:
    st.session_state.use_keyword_extractor = strtobool(query_params.get(
        'use_keyword_extractor', ['False'])[0])

if 'keyword_number' not in st.session_state:
    st.session_state.keyword_number = int(query_params.get(
        'keyword_number', [10])[0])

if 'retriever' not in st.session_state:
    st.session_state.retriever = query_params.get(
        'retriever', ['default'])[0]

if 'chunk_to_retrieve' not in st.session_state:
    st.session_state.chunk_to_retrieve = int(query_params.get(
        'chunk_to_retrieve', [3])[0])

# Sidebar

st.sidebar.title("Parameters")

with st.sidebar.expander("Build Index Params"):
    selected_embedding_model = st.selectbox(
        'Which embeddings should we use?',
        ('text-ada-002', 'intfloat/e5-large-v2',
         'sentence-transformers/all-MiniLM-L6-v2'), key='embedding_model'
    )

    chunk_size = st.slider(
        'Nodes Chunk size',
        500, 2000, step=100, key='chunk_size')

    chunk_overlap = st.slider(
        'Nodes Chunk Overlap',
        0, 150, step=10, key='chunk_overlap')

    text_splitter = st.selectbox(
        'Split method',
        ('TokenTextSplitter', 'SentenceTextSplitter'), key='text_splitter'
    )

    st.subheader("Document metadata extraction")

    use_title_extractor = st.checkbox(
        'Use TitleExtractor', key='use_title_extractor')
    nodes_for_title_extraction = st.slider(
        'Number of nodes to use for title extraction', 2, 10, disabled=not use_title_extractor, key='nodes_for_title_extraction')

    use_summary_extractor = st.checkbox(
        'Use SummaryExtractor', key='use_summary_extractor')
    summary_nodes = st.multiselect(
        'Which nodes should we use for each node summary extraction?',
        ['prev', 'self', 'next'], ['prev', 'self'], disabled=not use_summary_extractor, key='summary_nodes')

    use_questions_answered_extractor = st.checkbox(
        'Use QuestionsAnsweredExtractor', key='use_questions_answered_extractor')
    questions_answered_number = st.slider(
        'Number of questions to answer', 1, 10, disabled=not use_questions_answered_extractor, key='questions_answered_number')
    use_keyword_extractor = st.checkbox(
        'Use KeywordExtractor', key='use_keyword_extractor')
    keyword_number = st.slider(
        'Number of keywords to extract', 1, 20, disabled=not use_keyword_extractor, key='keyword_number')

with st.sidebar.expander("Retriever Params"):
    selected_llm = st.selectbox(
        'Which LLM should we use?',
        ('gpt-3.5-turbo', 'gpt-4', 'claude-2', 'claude-instant-1'), key='llm_model'
    )

    retriever = st.selectbox(
        'Retriever',
        ('default', 'sparse', 'hybrid', 'text_search', 'svm',
         'logistic_regression', 'linear_regression', 'mmr'), key='retriever'
    )

    chunk_to_retrieve = st.slider(
        'Number of chunks to retrieve',
        2, 5, 3, key='chunk_to_retrieve'
    )

st.title("🦙 Llama Index - Langchain - Vector DB Demo 🦙")
st.header("Welcome to the Llama Index Streamlit Demo")
st.write(
    "Enter a query about Paul Graham's essays. You can check out the original essay [here](https://raw.githubusercontent.com/jerryjliu/llama_index/main/examples/paul_graham_essay/data/paul_graham_essay.txt). Your query will be answered using the essay as context, using embeddings from different models and LLM completions from gpt-3.5-turbo or Anthropic. You can read more about Llama Index and how this works in [our docs!](https://gpt-index.readthedocs.io/en/latest/index.html)"
)

metadata_extractors = {}

if use_title_extractor:
    metadata_extractors['title'] = {
        'nodes_for_title_extraction': nodes_for_title_extraction}

if use_summary_extractor:
    metadata_extractors['summary'] = {'summary_nodes': summary_nodes}

if use_questions_answered_extractor:
    metadata_extractors['questions_answered'] = {
        'questions_answered_number': questions_answered_number}

if use_keyword_extractor:
    metadata_extractors['keywords'] = {'keyword_number': keyword_number}

st.experimental_set_query_params(
    embedding_model=selected_embedding_model,
    llm_model=selected_llm,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    text_splitter=text_splitter,
    use_title_extractor=use_title_extractor,
    nodes_for_title_extraction=nodes_for_title_extraction,
    use_summary_extractor=use_summary_extractor,
    summary_nodes=summary_nodes,
    use_questions_answered_extractor=use_questions_answered_extractor,
    questions_answered_number=questions_answered_number,
    use_keyword_extractor=use_keyword_extractor,
    keyword_number=keyword_number
)

index = None
index = generate_index(selected_embedding_model, selected_llm,
                       chunk_size, chunk_overlap, text_splitter, metadata_extractors)


if index is None:
    st.warning("Please enter your api key first.")

text = st.text_input("Query text:", value="What did the author do growing up?")

if st.button("Run Query") and text is not None:
    response, nodes_retrieved = query_index(
        index, text, retriever, chunk_to_retrieve)
    st.markdown(response)

    st.header("Nodes Retrieved:")
    for node in nodes_retrieved:
        st.markdown(json.dumps(
            {"node_id": node.node.node_id, "score": node.score}, indent=2))
        with st.expander("See Text"):
            st.markdown(node.node.text)

    llm_col, embed_col = st.columns(2)
    # with llm_col:
    #     st.markdown(
    #         f"LLM Tokens Used: {index.service_context._last_token_usage}"
    #     )

    # with embed_col:
    #     st.markdown(
    #         f"Embedding Tokens Used: {index.service_context.embed_model._last_token_usage}"
    #     )
