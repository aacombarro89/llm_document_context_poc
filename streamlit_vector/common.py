import json
import streamlit as st
import hashlib
import os
import random
import itertools
import mimetypes
import time

import pandas as pd

from langchain.chains import QAGenerationChain
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.embeddings import HuggingFaceEmbeddings

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
from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index.evaluation import DatasetGenerator, QueryResponseEvaluator
from llama_index.logger import LlamaLogger
from llama_index import download_loader
from llama_index.callbacks import CallbackManager, LlamaDebugHandler

from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI, QA_CHAIN_PROMPT, QA_CHAIN_PROMPT_LLAMA
from utils import product_list, flat_product_list

PyMuPDFReader = download_loader("PyMuPDFReader")

base_index_name = "./saved_index"
documents_folder = "./documents"


@st.cache_resource
def make_llm(model_version: str):
    """
    Make LLM from model version
    @param model_version: model_version
    @return: LLN
    """
    if (model_version == "gpt-3.5-turbo") or (model_version == "gpt-4"):
        chosen_model = ChatOpenAI(model_name=model_version, temperature=0)
    elif model_version == "claude-2":
        chosen_model = ChatAnthropic(temperature=0, model='claude-2')
    elif model_version == "claude-instant-1":
        chosen_model = ChatAnthropic(temperature=0, model='claude-instant-1')
    else:
        st.warning(
            "`Model version not recognized. Using gpt-3.5-turbo`", icon="⚠️")
        chosen_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return chosen_model


def get_text_splitter(text_splitter, chunk_size, chunk_overlap):
    if text_splitter == 'TokenTextSplitter':
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif text_splitter == 'SentenceTextSplitter':
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        return None


def get_metadata_extractors(metadata_extractors={}):
    extractors = []
    if 'title' in metadata_extractors and \
            metadata_extractors['title']['nodes_for_title_extraction'] and \
            metadata_extractors['title']['nodes_for_title_extraction'] > 0:
        extractors.append(TitleExtractor(
            nodes=metadata_extractors['title']['nodes_for_title_extraction']))

    if 'summary' in metadata_extractors and metadata_extractors['summary']['summary_nodes'] and len(metadata_extractors['summary']['summary_nodes']) > 0:
        extractors.append(SummaryExtractor(
            summaries=metadata_extractors['summary']['summary_nodes']))

    if 'questions_answered' in metadata_extractors and \
            metadata_extractors['questions_answered']['questions_answered_number'] and \
            metadata_extractors['questions_answered']['questions_answered_number'] > 0:
        extractors.append(QuestionsAnsweredExtractor(
            questions=metadata_extractors['questions_answered']['questions_answered_number']))

    if 'keywords' in metadata_extractors and metadata_extractors['keywords']['keyword_number'] and metadata_extractors['keywords']['keyword_number'] > 0:
        extractors.append(KeywordExtractor(
            keywords=metadata_extractors['keywords']['keyword_number']))

    return MetadataExtractor(extractors=extractors) if len(extractors) > 0 else None


def generate_eval_for_file(file, chunk_size, num_questions, llm_model):
    text = ''
    with open(file.path) as f:
        contents = f.read()

        if mimetypes.MimeTypes().guess_type(file.name)[0] == 'text/plain':
            text = contents
        else:
            st.warning(
                "Unsupported file type for file: {}".format(file.filename))

    # Generate random starting index in the doc to draw question from
    num_of_chars = len(text)
    starting_index = random.randint(0, num_of_chars-chunk_size)
    sub_sequence = text[starting_index:starting_index+chunk_size]
    # Set up QAGenerationChain chain using GPT 3.5 as default
    chain = QAGenerationChain.from_llm(make_llm(llm_model))
    eval_set = []
    # Catch any QA generation errors and re-try until QA pair is generated
    remaning_answers = num_questions
    while remaning_answers > 0:
        try:
            qa_pair = chain.run(sub_sequence)
            eval_set.append(qa_pair)
            remaning_answers = remaning_answers - 1
        finally:
            starting_index = random.randint(0, num_of_chars-chunk_size)
            sub_sequence = text[starting_index:starting_index+chunk_size]
    eval_pair = list(itertools.chain.from_iterable(eval_set))
    return eval_pair


def generate_eval_for_files_in_folder(chunk_size, num_questions, llm_model):
    """
    Generate question answer pair from input text 
    @param text: text to generate eval set from
    @param chunk: chunk size to draw question from text
    @param logger: logger
    @return: dict with keys "question" and "answer"
    """

    eval_set = []

    files = os.scandir(documents_folder)
    for file in files:
        eval_set.append(generate_eval_for_file(
            file, chunk_size, num_questions, llm_model))

    return eval_set


def generate_experiments_params_set(embedding_models, llm_models, chunk_size, chunk_overlaps,
                                    text_splitters, retrievers, chunk_to_retrieves, nodes_for_title_extraction,
                                    summary_nodes, questions_answered_number, keyword_number):
    range_of_chunk_size = range(
        chunk_size[0], chunk_size[1], 100) if chunk_size[0] < chunk_size[1] else [chunk_size[0]]
    range_of_chunk_overlap = range(
        chunk_overlaps[0], chunk_overlaps[1], 10) if chunk_overlaps[0] < chunk_overlaps[1] else [chunk_overlaps[0]]
    range_of_chunk_to_retrieve = range(
        chunk_to_retrieves[0], chunk_to_retrieves[1]) if chunk_to_retrieves[0] < chunk_to_retrieves[1] else [chunk_to_retrieves[0]]

    experiments_to_run = product_list(embedding_models, llm_models, range_of_chunk_size, range_of_chunk_overlap, text_splitters,
                                      retrievers, range_of_chunk_to_retrieve)

    if len(nodes_for_title_extraction) > 0:
        range_of_nodes_for_title_extraction = range(
            nodes_for_title_extraction[0], nodes_for_title_extraction[1]) if nodes_for_title_extraction[0] < nodes_for_title_extraction[1] else nodes_for_title_extraction
        experiments_to_run = flat_product_list(
            experiments_to_run, range_of_nodes_for_title_extraction)
    else:
        experiments_to_run = flat_product_list(experiments_to_run, [None])

    if len(summary_nodes) > 0:
        experiments_to_run = flat_product_list(
            experiments_to_run, [summary_nodes])
    else:
        experiments_to_run = flat_product_list(experiments_to_run, [None])

    if len(questions_answered_number) > 0:
        range_of_questions_answered_number = range(
            questions_answered_number[0], questions_answered_number[1]) if questions_answered_number[0] < questions_answered_number[1] else questions_answered_number
        experiments_to_run = flat_product_list(
            experiments_to_run, [range_of_questions_answered_number])
    else:
        experiments_to_run = flat_product_list(experiments_to_run, [None])

    if len(keyword_number) > 0:
        range_of_keyword_number = range(
            keyword_number[0], keyword_number[1]) if keyword_number[0] < keyword_number[1] else keyword_number
        experiments_to_run = flat_product_list(
            experiments_to_run, [range_of_keyword_number])
    else:
        experiments_to_run = flat_product_list(experiments_to_run, [None])

    return list(map(lambda params: {
        'index_params': {
            'embedding_model': params[0],
            'llm_model': params[1],
            'chunk_size': params[2],
            'chunk_overlap': params[3],
            'text_splitter': params[4],
            'metadata_extractors': {
                'title': {
                    'nodes_for_title_extraction': params[7]
                },
                'summary': {
                    'summary_nodes': [params[8]] if params[8] is not None else []
                },
                'questions_answered': {
                    'questions_answered_number': params[9]
                },
                'keywords': {
                    'keyword_number': params[10]
                }
            }
        },
        'eval_params': {
            'retriever_mode': params[5],
            'chunks_to_retrieve': params[6]
        }
    }, experiments_to_run))


def grade_model_answer(predicted_dataset, predictions, grade_answer_prompt):
    """
    Grades the answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @param logger: logger
    @return: A list of scores for the distilled answers.
    """

    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=ChatAnthropic(temperature=0, model='claude-instant-1'),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def grade_model_retrieval(gt_dataset, predictions, grade_docs_prompt):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading.
    @return: list of scores for the retrieved documents.
    """

    if grade_docs_prompt == "Fast":
        prompt = GRADE_DOCS_PROMPT_FAST
    else:
        prompt = GRADE_DOCS_PROMPT

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=ChatAnthropic(temperature=0, model='claude-instant-1'),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(gt_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def run_evaluation(index, data_set, grade_prompt, **query_index_kwargs):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chain: Model chain used for answering questions
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param grade_prompt: String prompt used for grading model's performance
    @param retriever_type: String specifying the type of retriever used
    @param num_neighbors: Number of neighbors to retrieve using the retriever
    @param text: full document text
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_grade: A dictionary containing scores for the model's document retrieval.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """

    predictions_list = []
    retrieved_docs = []
    retrieved_nodes = []
    gt_dataset = []
    latencies_list = []

    for data in data_set:
        start_time = time.time()

        (answer, nodes_retrieved) = query_index(
            index, data["question"], **query_index_kwargs)

        predictions_list.append(
            {"question": data["question"], "answer": data["answer"], "result": answer})

        gt_dataset.append(data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        latencies_list.append(elapsed_time)

        # Extract text from retrieved docs
        retrieved_doc_text = ""
        for node in nodes_retrieved:
            retrieved_doc_text += "Doc %s: " % str(
                node.node.node_id) + node.node.text

        retrieved_nodes.append(
            ', '.join(map(lambda node: node.node.node_id, nodes_retrieved)))
        # Log
        retrieved = {"question": data["question"],
                     "answer": data["answer"], "result": retrieved_doc_text}
        retrieved_docs.append(retrieved)

    # Grade
    graded_answers = grade_model_answer(
        gt_dataset, predictions_list, grade_prompt)
    graded_retrieval = grade_model_retrieval(
        gt_dataset, retrieved_docs, grade_prompt)

    return graded_answers, graded_retrieval, latencies_list, predictions_list, retrieved_nodes


@st.cache_resource
def init_service_context(
        selected_embedding_model,
        selected_llm_model,
        chunk_size,
        chunk_overlap,
        selected_text_splitter,
        selected_metadata_extractors={},
        callback_manager=None,
):
    metadata_extractors = get_metadata_extractors(selected_metadata_extractors)
    text_splitter = get_text_splitter(
        selected_text_splitter, chunk_size, chunk_overlap)
    node_parser = SimpleNodeParser(
        text_splitter=text_splitter, metadata_extractor=metadata_extractors)

    llm_model = make_llm(selected_llm_model)

    if selected_embedding_model == 'text-ada-002':
        embed_model = OpenAIEmbedding()
    else:
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=selected_embedding_model))

    print("Using LLM model: ", selected_llm_model,
          "Using Embedding model: ", selected_embedding_model)

    llama_logger = LlamaLogger()

    if callback_manager is None:
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])

    return ServiceContext.from_defaults(
        llm=llm_model, embed_model=embed_model, node_parser=node_parser, llama_logger=llama_logger, callback_manager=callback_manager)


def get_index_folder_name(embedding_model,
                          llm_model,
                          chunk_size,
                          chunk_overlap,
                          text_splitter,
                          metadata_extractors={},
                          base_index_folder=base_index_name,
                          documents_folder=documents_folder,
                          ):
    index_full_name = f"{embedding_model}_{llm_model}_{chunk_size}_{chunk_overlap}_{text_splitter}_{repr(metadata_extractors)}"
    folder_name = hashlib.sha256(index_full_name.encode('UTF-8')).hexdigest()

    full_folder_path = os.path.join(
        base_index_folder, llm_model, embedding_model, folder_name)

    return full_folder_path


@st.cache_resource
def generate_index(embedding_model,
                   llm_model,
                   chunk_size,
                   chunk_overlap,
                   text_splitter,
                   metadata_extractors={},
                   base_index_folder=base_index_name,
                   documents_folder=documents_folder,
                   ):
    service_context = init_service_context(
        embedding_model,
        llm_model,
        chunk_size,
        chunk_overlap,
        text_splitter,
        metadata_extractors,
    )

    full_folder_path = get_index_folder_name(
        embedding_model,
        llm_model,
        chunk_size,
        chunk_overlap,
        text_splitter,
        metadata_extractors,
        base_index_folder,
        documents_folder,
    )

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
                'llm_model': llm_model,
                'embedding_model': embedding_model,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'text_splitter': text_splitter,
                'metadata_extractors': metadata_extractors
            }, f, indent=2)

    return index

# @st.cache_data(max_entries=200, persist=True)


def query_index(_index, query_text, retriever_mode, chunks_to_retrieve):
    if _index is None:
        return "Please initialize the index!"
    response = _index.as_query_engine(vector_store_query_mode=VectorStoreQueryMode(
        retriever_mode), similarity_top_k=chunks_to_retrieve).query(query_text)

    nodes_retrieved = response.source_nodes

    return (str(response), nodes_retrieved)
