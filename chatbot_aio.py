
import warnings
warnings.filterwarnings("ignore")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from groq import Groq
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import config


API_KEY = config.GEMINI_API

groq_api_key = config.GROQ_API


template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know
            {context}
            Question: {question}
            Helpful Answer:"""
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


class chatbot:
    
    def __init__(self,data, model="gemini-pro"):
        self.data = "".join(data)
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY,
                            temperature=0.2,convert_system_message_to_human=True)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=API_KEY)
        self.vector_index = self.prep_data()
        
        self.QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        self.qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=self.vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT}
        )

    def prep_data(self):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        texts = text_splitter.split_text(self.data)
        vector_index = Chroma.from_texts(texts, self.embeddings).as_retriever()
        
        return vector_index
    
class chatbotLLama:
    
    def __init__(self, data, model = 'llama3-70b-8192'):
        self.data = "".join(data)
        maxtoken = 7092
        
        self.model = ChatGroq(groq_api_key=groq_api_key, 
                              model_name=model, 
                              max_tokens=maxtoken)
        cohere_api_key = config.COHERE_API
        self.embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=cohere_api_key)
        self.vector_index = self.create_vector_db()

    def qa_bot(self):
        embeddings = self.embeddings
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = self.model
        qa_prompt = self.set_custom_prompt()
        qa = self.retrieval_qa_chain(llm, qa_prompt, db)
        return qa
    
    def set_custom_prompt(self):
        """
        Prompt template for QA retrieval for each vectorstore
        """
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt
    
    def retrieval_qa_chain(self, llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                               chain_type='stuff',
                                               retriever=db.as_retriever(search_kwargs={'k': 2}),
                                               return_source_documents=False,
                                               chain_type_kwargs={'prompt': prompt})
        return qa_chain
    
    def create_vector_db(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        texts = text_splitter.split_text(self.data)

        embeddings = self.embeddings
        db = FAISS.from_texts(texts, self.embeddings)
        db.save_local(DB_FAISS_PATH)
        return db
    
    def final_result(self, query):
        qa_result = self.qa_bot()
        response = qa_result({'query': query})
        return response
    
class chatbotMix:
    
    def __init__(self,data, model = 'mixtral-8x7b-32768'):
        self.data = "".join(data)
        maxtoken = 7092
        self.model = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
            )

        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        self.vector_index = self.prep_data()
        
        self.QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        self.qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=self.vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT}
        )

    def prep_data(self):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        texts = text_splitter.split_text(self.data)
        vector_index = Chroma.from_texts(texts, self.embeddings).as_retriever()
        
        return vector_index
    
