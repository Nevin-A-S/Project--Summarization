import warnings
warnings.filterwarnings("ignore")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
import config

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.
{context}
Question: {question}
Helpful Answer:"""

DB_FAISS_PATH = 'vectorstore/db_faiss'

class RAGSystem:
    def __init__(self, data):
        self.data = "".join(data)
        self.embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=config.COHERE_API)
        self.vector_store = self.create_vector_store()

    def create_vector_store(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        texts = text_splitter.split_text(self.data)
        db = FAISS.from_texts(texts, self.embeddings)
        db.save_local(DB_FAISS_PATH)
        return db

    def load_vector_store(self):
        return FAISS.load_local(DB_FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True)

    def get_qa_chain(self, model):
        db = self.load_vector_store()
        prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
        return RetrievalQA.from_chain_type(
            llm=model,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )

class chatbot:
    def __init__(self, data, model="gemini-pro"):
        self.rag_system = RAGSystem(data)
        self.model = ChatGoogleGenerativeAI(model=model, google_api_key=config.GEMINI_API,
                                            temperature=0.2, convert_system_message_to_human=True)
        self.qa_chain = self.rag_system.get_qa_chain(self.model)

    def final_result(self, query):
        response = self.qa_chain({'query': query})
        return response

class chatbotLLama:
    def __init__(self, data, model='llama3-70b-8192'):
        self.rag_system = RAGSystem(data)
        self.model = ChatGroq(groq_api_key=config.GROQ_API, model_name=model, max_tokens=7092)
        self.qa_chain = self.rag_system.get_qa_chain(self.model)

    def final_result(self, query):
        response = self.qa_chain({'query': query})
        return response

class chatbotMix:
    def __init__(self, data, model='mixtral-8x7b-32768'):
        self.rag_system = RAGSystem(data)
        self.model = ChatGroq(groq_api_key=config.GROQ_API, model_name=model)
        self.qa_chain = self.rag_system.get_qa_chain(self.model)

    def final_result(self, query):
        response = self.qa_chain({'query': query})
        return response
