import json
from typing import List

import pandas as pd
import altair as alt
import streamlit as st

from utils import strtobool
from common import generate_experiments_params_set, generate_index, run_evaluation, generate_eval_for_files_in_folder

query_params = st.experimental_get_query_params()

# Stored current state in session state
if 'ev_embedding_model' not in st.session_state:
    st.session_state.ev_embedding_model = query_params.get(
        'ev_embedding_model', ['text-ada-002'])

if 'ev_llm_model' not in st.session_state:
    st.session_state.ev_llm_model = query_params.get(
        'ev_llm_model', ['gpt-3.5-turbo'])

if 'ev_chunk_size' not in st.session_state:
    st.session_state.ev_chunk_size = list(map(int, query_params.get(
        'ev_chunk_size', [800, 1000])))

if 'ev_chunk_overlap' not in st.session_state:
    st.session_state.ev_chunk_overlap = list(map(int, query_params.get(
        'ev_chunk_overlap', [100, 100])))

if 'ev_text_splitter' not in st.session_state:
    st.session_state.ev_text_splitter = query_params.get(
        'ev_text_splitter', ['TokenTextSplitter'])

if 'ev_nodes_for_title_extraction' not in st.session_state:
    st.session_state.ev_nodes_for_title_extraction = list(map(int, query_params.get(
        'ev_nodes_for_title_extraction', [0, 0])))

if 'ev_summary_nodes' not in st.session_state:
    st.session_state.ev_summary_nodes = query_params.get(
        'ev_summary_nodes', ['self'])

if 'ev_questions_answered_number' not in st.session_state:
    st.session_state.ev_questions_answered_number = list(map(int, query_params.get(
        'ev_questions_answered_number', [0, 0])))

if 'ev_keyword_number' not in st.session_state:
    st.session_state.ev_keyword_number = list(map(int, query_params.get(
        'ev_keyword_number', [0, 0])))

if 'ev_retriever' not in st.session_state:
    st.session_state.ev_retriever = query_params.get(
        'ev_retriever', ['default'])

if 'ev_chunk_to_retrieve' not in st.session_state:
    st.session_state.ev_chunk_to_retrieve = list(map(int, query_params.get(
        'ev_chunk_to_retrieve', [2, 3])))

if 'ev_num_eval_questions' not in st.session_state:
    st.session_state.ev_num_eval_questions = int(query_params.get(
        'ev_num_eval_questions', [5])[0])

if 'ev_grade_answer_prompt' not in st.session_state:
    st.session_state.ev_grade_answer_prompt = query_params.get(
        'ev_grade_answer_prompt', ['Fast'])[0]

# Keep dataframe in memory to accumulate experimental results
if "existing_df" not in st.session_state:
    summary = pd.DataFrame(columns=['model',
                                    'retriever',
                                    'embedding',
                                    'num_neighbors',
                                    'Latency',
                                    'Retrieval score',
                                    'Answer score'])
    st.session_state.existing_df = summary
else:
    summary = st.session_state.existing_df


def grade_model_answer(predicted_dataset: List, predictions: List, grade_answer_prompt: str) -> List:
    """
    Grades the distilled answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @return: A list of scores for the distilled answers.
    """
    # Grade the distilled answer
    st.info("`Grading model answer ...`")
    # Set the grading prompt based on the grade_answer_prompt parameter
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        prompt=prompt
    )

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        predicted_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )

    return graded_outputs


def grade_model_retrieval(gt_dataset: List, predictions: List, grade_docs_prompt: str):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading. Either "Fast" or "Full"
    @return: list of scores for the retrieved documents.
    """
    # Grade the docs retrieval
    st.info("`Grading relevance of retrieved docs ...`")

    # Set the grading prompt based on the grade_docs_prompt parameter
    prompt = GRADE_DOCS_PROMPT

    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        prompt=prompt
    )

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        gt_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )
    return graded_outputs


with st.sidebar.form("user_input"):
    with st.expander("Build Index Params"):
        selected_embedding_model = st.multiselect(
            'Which embeddings should we use?',
            ['text-ada-002', 'intfloat/e5-large-v2',
             'sentence-transformers/all-MiniLM-L6-v2'], key='ev_embedding_model'
        )

        chunk_size = st.slider(
            'Nodes Chunk size',
            500, 2000, step=100, key='ev_chunk_size')

        chunk_overlap = st.slider(
            'Nodes Chunk Overlap',
            0, 150, step=10, key='ev_chunk_overlap')

        text_splitter = st.multiselect(
            'Split method',
            ['TokenTextSplitter', 'SentenceTextSplitter'], key='ev_text_splitter'
        )

        st.subheader("Document metadata extraction")

        nodes_for_title_extraction = st.slider(
            'Number of nodes to use for title extraction', 2, 10, key='ev_nodes_for_title_extraction')

        summary_nodes = st.multiselect(
            'Which nodes should we use for each node summary extraction?',
            ['prev', 'self', 'next'], key='ev_summary_nodes')

        questions_answered_number = st.slider(
            'Number of questions to answer', 1, 10, key='ev_questions_answered_number')

        keyword_number = st.slider(
            'Number of keywords to extract', 1, 20, key='ev_keyword_number')

    with st.expander("Retriever Params"):
        selected_llm = st.multiselect(
            'Which LLM should we use?',
            ['gpt-3.5-turbo', 'anthropic'],
            key='ev_llm_model'
        )

        retriever = st.multiselect(
            'Retriever',
            ['default', 'sparse', 'hybrid', 'text_search', 'svm',
             'logistic_regression', 'linear_regression', 'mmr'], key='ev_retriever'
        )

        chunk_to_retrieve = st.slider(
            'Number of chunks to retrieve',
            2, 5, key='ev_chunk_to_retrieve'
        )

    with st.expander("Evaluator Params"):
        num_eval_questions = st.slider(
            'Number of questions to evaluate (autogenerated if no test set is provided)',
            1, 10, key='ev_num_eval_questions'
        )

        grade_prompt = st.radio("`Grading style prompt`",
                                ("Fast",
                                 "Descriptive",
                                 "Descriptive w/ bias check",
                                 "OpenAI grading prompt"),
                                key='ev_grade_answer_prompt')

    experiment_params_submitted = st.form_submit_button("Submit evaluation")

st.experimental_set_query_params(
    ev_embedding_model=selected_embedding_model,
    ev_chunk_size=chunk_size,
    ev_chunk_overlap=chunk_overlap,
    ev_text_splitter=text_splitter,
    ev_nodes_for_title_extraction=nodes_for_title_extraction,
    ev_summary_nodes=summary_nodes,
    ev_questions_answered_number=questions_answered_number,
    ev_keyword_number=keyword_number,
    ev_llm_model=selected_llm,
    ev_retriever=retriever,
    ev_chunk_to_retrieve=chunk_to_retrieve,
    ev_num_eval_questions=num_eval_questions
)


st.header("`VectorDB auto-evaluator`")
st.info(
    "`I am an evaluation tool for question-answering using an existing vectorDB (currently Pinecone is supported) and an eval set. "
    "I will generate and grade an answer to each eval set question with the user-specific retrival setting, such as metadata filtering or self-querying retrieval."
    " Experiments with different configurations are logged. For an example eval set, see eval_sets/lex-pod-eval.json.`")

generate_eval_set_clicked = st.button(
    "Generate eval set")

eval_set = [
    {
        "question": "What was the benefit of being part of a batch in the Summer Founders Program?",
        "answer": "Being part of a batch in the Summer Founders Program solved the problem of isolation faced by founders."
    },
    {
        "question": "What did the author realize about AI during their first year of grad school?",
        "answer": "The author realized that the AI being practiced at the time was a hoax and that there was an unbridgeable gap between what the programs could do and actually understanding natural language."
    },
    {
        "question": "What did the author and Robert start trying to write in the summer of 1995?",
        "answer": "Software to build online stores"
    },
    {
        "question": "What job did the author get after deciding not to go back to RISD?",
        "answer": "The author got a job at a company called Interleaf."
    },
    {
        "question": "What is the name of the new Lisp language that the author wrote?",
        "answer": "The new Lisp language is called Bel."
    }
]

if generate_eval_set_clicked:
    st.info("`Generating eval set ...`")
    eval_set = generate_eval_for_files_in_folder(
        1024, num_eval_questions, 'gpt-3.5-turbo')

st.json(eval_set)

if experiment_params_submitted and len(eval_set) > 0:
    experiments_params_set = generate_experiments_params_set(selected_embedding_model, selected_llm, chunk_size,
                                                             chunk_overlap, text_splitter, retriever, chunk_to_retrieve,
                                                             nodes_for_title_extraction if nodes_for_title_extraction[1] != 0 else [
                                                             ],
                                                             summary_nodes if len(
                                                                 summary_nodes) > 0 else [],
                                                             questions_answered_number if questions_answered_number[1] != 0 else [
                                                             ],
                                                             keyword_number if keyword_number[1] != 0 else [])

    st.subheader("`Number of Experiments to run: %s`" %
                 len(experiments_params_set))

    progress_bar = st.progress(0, '%s Experiments completed' % 0)

    for i in range(len(experiments_params_set)):
        experiment_params = experiments_params_set[i]
        st.info("`Running experiment with params: %s`" % experiment_params)
        st.info("`Generating index ...`")
        index = generate_index(**experiment_params['index_params'])

        st.info("`Running Evaluation ...`")
        graded_answers, graded_retrieval, latency, predictions, retrieved_nodes = run_evaluation(
            index, eval_set, grade_prompt, **experiment_params['eval_params'])

        d = pd.DataFrame(predictions)
        d['answer score'] = [g['text'] for g in graded_answers]
        d['retrieved nodes'] = [g for g in retrieved_nodes]
        d['docs score'] = [g['text'] for g in graded_retrieval]
        d['latency'] = latency

        mean_latency = d['latency'].mean()
        correct_answer_count = len(
            [text for text in d['answer score'] if "Incorrect" not in text])
        correct_docs_count = len(
            [text for text in d['docs score'] if "Incorrect" not in text])
        percentage_answer = (correct_answer_count / len(graded_answers)) * 100
        percentage_docs = (correct_docs_count / len(graded_retrieval)) * 100

        st.subheader("`Results for experiment with params: %s`" %
                     experiment_params)
        st.info(
            "`I will grade the chain based on: 1/ the relevance of the retrived documents relative to the question and 2/ "
            "the summarized answer relative to the ground truth answer. You can see (and change) to prompts used for "
            "grading in text_utils`")
        st.dataframe(data=d, use_container_width=True)

        new_row = pd.DataFrame({'model': [selected_llm],
                                'retriever': [retriever],
                                'embedding': [selected_embedding_model],
                                'num_neighbors': [chunk_to_retrieve],
                                'Latency': [mean_latency],
                                'Retrieval score': [percentage_docs],
                                'Answer score': [percentage_answer]})
        summary = pd.concat([summary, new_row], ignore_index=True)

        progress_bar.progress(
            (i + 1) / len(experiments_params_set), '%s Experiments completed' % (i + 1))

    # Accumulate results
    st.subheader("`Aggregate Results`")
    st.info(
        "`Retrieval and answer scores are percentage of retrived documents deemed relevant by the LLM grader ("
        "relative to the question) and percentage of summarized answers deemed relevant (relative to ground truth "
        "answer), respectively. The size of point correponds to the latency (in seconds) of retrieval + answer "
        "summarization (larger circle = slower).`")

    st.dataframe(data=summary, use_container_width=True)
    st.session_state.existing_df = summary

    # Dataframe for visualization
    show = summary.reset_index().copy()
    show.columns = ['expt number', 'model', 'retriever', 'embedding',
                    'num_neighbors', 'Latency', 'Retrieval score', 'Answer score']
    show['expt number'] = show['expt number'].apply(
        lambda x: "Expt #: " + str(x + 1))
    c = alt.Chart(show).mark_circle().encode(x='Retrieval score',
                                             y='Answer score',
                                             size=alt.Size('Latency'),
                                             color='expt number',
                                             tooltip=['expt number', 'Retrieval score', 'Latency', 'Answer score'])
    st.altair_chart(c, use_container_width=True, theme="streamlit")

    # d = pd.DataFrame(predictions)
    # d['answer score'] = [g['text'] for g in graded_answers]
    # d['docs score'] = [g['text'] for g in graded_retrieval]
    # d['latency'] = latency

    # # Summary statistics
    # mean_latency = d['latency'].mean()
    # correct_answer_count = len(
    #     [text for text in d['answer score'] if "Incorrect" not in text])
    # correct_docs_count = len(
    #     [text for text in d['docs score'] if "Incorrect" not in text])
    # percentage_answer = (correct_answer_count / len(graded_answers)) * 100
    # percentage_docs = (correct_docs_count / len(graded_retrieval)) * 100

    # st.subheader("`Run Results`")
    # st.info(
    #     "`I will grade the chain based on: 1/ the relevance of the retrived documents relative to the question and 2/ "
    #     "the summarized answer relative to the ground truth answer. You can see (and change) to prompts used for "
    #     "grading in text_utils`")
    # st.dataframe(data=d, use_container_width=True)


# with st.form(key='file_inputs'):

#     uploaded_eval_set = st.file_uploader("`Please upload eval set (.json):` ",
#                                          type=['json'],
#                                          accept_multiple_files=False)

#     experiment_params_submitted = st.form_submit_button("Submit files")
