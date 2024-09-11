import streamlit as st
from scraper import scraper
from summarizer import summarizer
from chatbot_aio import chatbot as GBot, chatbotLLama as LBot, chatbotMix as MBot
from youtubedata import create_transcript_youtube
from googleSearch import google_content_cleaned
from pdf_data import extract_pdf_text

class WebSummarizerChatbot:
    def __init__(self):
        st.set_page_config(page_title="Web Summarizer Chatbot")
        st.markdown("<h1 style='text-align: center;'>Web Summarizer Chatbot</h1>", unsafe_allow_html=True)
        self.options = ['Web Link', 'Youtube Video', 'Topic Search', 'PDF']
        self.models = ['mixtral-8x7b-32768', 'llama3-70b-8192', 'gemini-pro']

        # Initialize session state for text, chatbot, and model
        if "text" not in st.session_state:
            st.session_state.text = None
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = None
        if "model" not in st.session_state:
            st.session_state.model = None

    def run(self):
        self.utility = st.selectbox('Utility', options=self.options)
        self.model = st.sidebar.selectbox('Choose a model', self.models)
        
        # Check if the model has changed and reset the chatbot if needed
        if st.session_state.model != self.model:
            st.session_state.chatbot = None
            st.session_state.model = self.model  # Update model in session state
        
        self.handle_link()

    def handle_link(self):
        # Reset URL and content when the utility changes
        if 'current_utility' not in st.session_state or st.session_state.current_utility != self.utility:
            st.session_state.current_utility = self.utility
            st.session_state.url = None
            st.session_state.text = None
            st.session_state.chatbot = None

        # Input based on the selected utility
        if self.utility in ['Web Link', 'Youtube Video', 'Topic Search']:
            st.session_state.url = st.text_input(f"Enter {self.utility}:")
        else:
            st.session_state.url = st.file_uploader("Upload PDF file", type=["pdf"])

        # Fetch content and summarize when the button is pressed
        if st.button("Get Summary"):
            if self.fetch_content():
                self.summary = summarizer(st.session_state.text, self.model)
                st.success(self.summary)
            else:
                st.error("Error: Unable to fetch content. Please check the input and try again.")

        self.initialize_chat_messages()
        self.handle_chat_input()

    def fetch_content(self):
        try:
            if self.utility == 'Web Link':
                st.session_state.text = scraper(st.session_state.url)
            elif self.utility == 'Youtube Video':
                st.session_state.text = create_transcript_youtube(st.session_state.url)
            elif self.utility == 'Topic Search':
                st.session_state.text = google_content_cleaned(st.session_state.url)
            elif self.utility == 'PDF' and st.session_state.url is not None:
                st.session_state.text = extract_pdf_text(st.session_state.url)

            # Reinitialize chatbot only if the text has changed
            if st.session_state.text:
                self.initialize_chatbot()

            return st.session_state.text is not None
        except Exception as e:
            st.error(f"Error fetching content: {e}")
            return False

    def initialize_chatbot(self):
        # Initialize chatbot object based on the selected model
        if self.model == 'gemini-pro':
            st.session_state.chatbot = GBot(st.session_state.text)
        elif self.model == 'llama3-70b-8192':
            st.session_state.chatbot = LBot(st.session_state.text)
        else:
            st.session_state.chatbot = MBot(st.session_state.text)

    def initialize_chat_messages(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_chat_input(self):
        # Handle user input for the chatbot
        prompt = st.chat_input(f"Enter your question ({self.model}): ")
        if prompt:
            self.handle_chat_message(prompt, "user")
            response = self.get_model_response(prompt)
            self.handle_chat_message(response, "assistant")

    def get_model_response(self, prompt):
        # Ensure that the chatbot is initialized when a question is asked
        if not st.session_state.chatbot:
            self.initialize_chatbot()

        # Get response based on the selected model
        try:
            if self.model == 'gemini-pro' or self.model == 'mixtral-8x7b-32768':
                # For `gemini-pro` and `mixtral`, use `qa_chain`
                return st.session_state.chatbot.qa_chain({"query": prompt})['result']
            elif self.model == 'llama3-70b-8192':
                # For `llama3`, use `final_result`
                return st.session_state.chatbot.final_result(prompt)['result']
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "Error: Unable to process the request."

    def handle_chat_message(self, content, role):
        # Add chat message to session state
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.markdown(content)

if __name__ == "__main__":
    app = WebSummarizerChatbot()
    app.run()
