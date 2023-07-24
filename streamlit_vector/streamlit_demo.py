import hashlib
import json
import os
import streamlit as st
from utils import strtobool
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter, SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings import OpenAIEmbedding, LangchainEmbedding
from llama_index.retrievers import VectorIndexRetriever
from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index.logger import LlamaLogger
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import langchain

# Related to: https://github.com/hwchase17/langchain/issues/4164
langchain.verbose = False

load_dotenv()  # take environment variables from .env.

base_index_name = "./saved_index"
documents_folder = "./documents"

service_context = None
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


def get_text_splitter(text_splitter, chunk_size, chunk_overlap):
    if text_splitter == 'TokenTextSplitter':
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif text_splitter == 'SentenceTextSplitter':
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        return None


def get_metadata_extractors(metadata_extractors={}):
    extractors = []
    if 'title' in metadata_extractors:
        extractors.append(TitleExtractor(
            nodes=metadata_extractors['title']['nodes_for_title_extraction']))
    if 'summary' in metadata_extractors:
        extractors.append(SummaryExtractor(
            summaries=metadata_extractors['summary']['summary_nodes']))
    if 'questions_answered' in metadata_extractors:
        extractors.append(QuestionsAnsweredExtractor(
            questions=metadata_extractors['questions_answered']['questions_answered_number']))
    if 'keywords' in metadata_extractors:
        extractors.append(KeywordExtractor(
            keywords=metadata_extractors['keywords']['keyword_number']))

    return MetadataExtractor(extractors=extractors) if len(extractors) > 0 else None


@st.cache_resource
def initialize_index(base_index_folder,
                     documents_folder,
                     selected_embedding_model,
                     selected_llm_model,
                     chunk_size,
                     chunk_overlap,
                     selected_text_splitter,
                     selected_metadata_extractors={},
                     ):
    global service_context

    metadata_extractors = get_metadata_extractors(selected_metadata_extractors)
    text_splitter = get_text_splitter(
        selected_text_splitter, chunk_size, chunk_overlap)
    node_parser = SimpleNodeParser(
        text_splitter=text_splitter, metadata_extractor=metadata_extractors)

    llm_model = None

    if selected_llm_model == 'gpt-3.5-turbo':
        llm_model = ChatOpenAI(temperature=0, model_name=selected_llm_model)
    elif selected_llm_model == 'anthropic':
        llm_model = ChatAnthropic()
    embed_model = None

    if selected_embedding_model == 'text-ada-002':
        embed_model = OpenAIEmbedding()
    else:
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=selected_embedding_model))

    print("Using LLM model: ", selected_llm_model,
          "Using Embedding model: ", selected_embedding_model)

    llama_logger = LlamaLogger()

    service_context = ServiceContext.from_defaults(
        llm=llm_model, embed_model=embed_model, node_parser=node_parser, llama_logger=llama_logger)

    index_full_name = f"{selected_embedding_model}_{selected_llm_model}_{chunk_size}_{chunk_overlap}_{selected_text_splitter}_{repr(selected_metadata_extractors)}"
    folder_name = hashlib.sha256(index_full_name.encode('UTF-8')).hexdigest()

    full_folder_path = os.path.join(
        base_index_folder, selected_llm_model, selected_embedding_model, folder_name)

    print("Index full name: ", index_full_name,
          "folder path: ", full_folder_path)

    if os.path.exists(full_folder_path):
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=full_folder_path),
            service_context=service_context,
        )
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.storage_context.persist(persist_dir=full_folder_path)

        params_filepath = os.path.join(
            full_folder_path, "index_params.json")

        with open(params_filepath, 'w') as f:
            json.dump({
                'llm_model': selected_llm_model,
                'embedding_model': selected_embedding_model,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'text_splitter': selected_text_splitter,
                'metadata_extractors': selected_metadata_extractors
            }, f, indent=2)

    return index


# @st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text, retriever_mode, chunks_to_retrieve):
    global service_context, nodes_retrieved

    if _index is None:
        return "Please initialize the index!"
    response = _index.as_query_engine(vector_store_query_mode=VectorStoreQueryMode(
        retriever_mode), similarity_top_k=chunks_to_retrieve).query(query_text)

    nodes_retrieved = response.source_nodes

    return str(response)


# Sidebar

st.sidebar.title("Parameters")

# questions_number = st.sidebar.slider(
#     'Number of eval questions',
#     0, 15, 5)


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
        0, 150, key='chunk_overlap')

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
        ('gpt-3.5-turbo', 'anthropic')
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

# grading_prompt = st.sidebar.selectbox(
#     'Grading prompt style',
#     ('Descriptive', 'Descriptive w/ bias check', 'OpenAI grading prompt')
# )

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
index = initialize_index(base_index_name, documents_folder, selected_embedding_model,
                         selected_llm, chunk_size, chunk_overlap, text_splitter, metadata_extractors)


if index is None:
    st.warning("Please enter your api key first.")

text = st.text_input("Query text:", value="What did the author do growing up?")

if st.button("Run Query") and text is not None:
    response = query_index(index, text, retriever, chunk_to_retrieve)
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
