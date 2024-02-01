# ! pip install chromadb sentence-transformers langchain openai

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




"""## Making Vector DB"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel


def setup_vectordb(app):
  try:
    os.environ["OPENAI_API_KEY"] = "add api key here"

    '''
    #emporting clean_embedding change path
    embedding_df = pd.read_pickle('data/clean_embedding_df(2).pkl')


    loader = DataFrameLoader(embedding_df, page_content_column="content")
    docs = loader.load()
    #app.config['docs'] = docs

    test_docs = docs[:10000]
    app.config['test_docs'] = test_docs

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="TaylorAI/gte-tiny")

    # load it into Chroma
    #clean_vectordb = Chroma.from_documents(docs, embedding_function,persist_directory='clean_vectordb')

    # load it into Chroma
    app.config['vectordb'] = Chroma.from_documents(app.config['test_docs'], embedding_function, persist_directory='test_clean_vectordb')

    #query = "The scent of flower"
    #retrived_docs = test_vectordb.similarity_search_with_score(query,k=3)
    '''
    embedding_function = SentenceTransformerEmbeddings(model_name="TaylorAI/gte-tiny")
    app.config['vectordb'] = Chroma(persist_directory='test_clean_vectordb', embedding_function=embedding_function)
    print("Set up vector db successfully")

    # print results
    #for retrived_doc in retrived_docs:
    #  print(retrived_doc)
  except Exception as e:
    print("error:" + str(e))



"""## LLM"""
def setup_llm(app):
  try:

    summary_template = """If the new prompt is clearly formulated in such a way that it relies on the context of a previous conversation, take into
    account the chat history and formulate a standalone prompt. When taking into account the chat history, if the last two entries of the chat history provide
    sufficient context to formulate the standalone prompt, only make use of those two entries to form the standalone prompt. If more context is needed, firstly take into
    account the two earlier entries, and similarly, if that is enough context, only make use of the now total four entries to form the standalone prompt. Lastly, if there is
    still not enough context to formulate a standalone prompt, take into account the whole chat history. 
    If the new prompt does not clearly require context, simply reply with the new prompt as is.
    Just reply with the prompt without commentary.

    Chat History:
    {chat_history}
    New prompt: {question}
    Standalone prompt:
    """

    summary_prompt = PromptTemplate.from_template(summary_template)

    question_template = """You are an expert on smells, and provide replies according to that role. Your focus is on sharing your expertise of smells, and
    that is reflected in the way you reply. Your replies should focus only on smell-related details.
    Provide an explanatory reply taking into account your role, preferrably that answers the provided prompt, based on the following context:

    Context: {context}
    prompt: {question}
    Reply:
    """

    question_prompt = ChatPromptTemplate.from_template(question_template)

    retriever = app.config['vectordb'].as_retriever()
    model = ChatOpenAI()
    output_parser = StrOutputParser()

    summary_chain = summary_prompt | model | output_parser

    retrieve_context = RunnableParallel({'context': retriever, 'question': RunnablePassthrough()})

    standalone_chain = question_prompt | model | output_parser

    app.config['summary_chain'] = summary_chain
    app.config['retrieve_context'] = retrieve_context
    app.config['standalone_chain'] = standalone_chain
    print("Set up llm successfully")
  except Exception as e:
    print("error:" + str(e))

def chat(app, chat_history, question):
  if len(chat_history) == 0:
    standalone_question = question
  else:
    summary_chain_input = {'chat_history': chat_history, 'question': question}
    standalone_question = app.config['summary_chain'].invoke(summary_chain_input)

  print("Standalone question: " + standalone_question)
  retrieval = app.config['retrieve_context'].invoke(standalone_question)
  #print(retrieval)
  # Retrieval: {'context': [Doc(page_content="", metadata={"source": "", "text": ""}), ...], 'question': ''} (4 docs for first query I did, not sure if always 4)
  retrieved_context = retrieval["context"]
  retrieved_metadatas = []
  for doc in retrieved_context:
    retrieved_metadatas.append(doc.metadata)

  #print(retrieved_metadatas)

  # Can make decision how many docs to use for relevant answer and which fields. Testing rn with metadatas
  answer = app.config['standalone_chain'].invoke({"context": retrieved_metadatas, "question": standalone_question})
  chat_history.append({'question': standalone_question, 'answer': answer})
  if len(chat_history) > 6:
    chat_history = chat_history[2:]

  print("Answer: " + answer)
  print("New chat history: ")
  print(chat_history)

  return retrieved_metadatas, chat_history, answer

def parse_metadatas(retrieved_metadatas):
  final_metadatas = []

  for metadata in retrieved_metadatas:

    # Format: 'ID: http://data.odeuropa.eu/smell/0184567a-93dd-5ebf-8d8b-2b81efe96259 Source: http://data.odeuropa.eu/source/5a8dacda-49ad-5f71-987e-381631691df3'
    source_string_split = metadata['source'].split()
    id_link = source_string_split[1]
    source_link = source_string_split[3]

    # Changing 'data' to 'explorer' and 'smell' to 'smells' to get link to explorer for this excerpt
    id_link = id_link.replace('data', 'explorer')
    id_link = id_link.replace('smell', 'smells')

    final_metadatas.append({'idlink': id_link, 'sourcelink': source_link, 'text': metadata['text']})

  #Decide what to do with metadatas after
  return final_metadatas