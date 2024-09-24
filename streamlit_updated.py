import streamlit as st
from scraper import scraper
from summarizer import summarizer
from chatbot_aio import chatbot as GBot, chatbotLLama as LBot, chatbotMix as MBot
from youtubedata import create_transcript_youtube
from googleSearch import google_content_cleaned
from pdf_data import extract_pdf_text

class WebSummarizerChatbot:
    def __init__(self):
        self.initialize_main_class()

    def initialize_main_class(self):
        st.set_page_config(page_title="Web Summarizer Chatbot")
        st.markdown("<h1 style='text-align: center;'>Web Summarizer Chatbot</h1>", unsafe_allow_html=True)
        self.options = ['Website', 'Google Search', 'PDF', 'Youtube']
        self.models = ['mixtral-8x7b-32768', 'llama3-70b-8192', 'gemini-pro']

        if "text" not in st.session_state:
            st.session_state.text = None
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = None
        if "model" not in st.session_state:
            st.session_state.model = None
        if "last_url" not in st.session_state:
            st.session_state.last_url = None

    def run(self):
        self.utility = st.selectbox('Select Input Source', options=self.options)
        self.model = st.sidebar.selectbox('Choose a model', self.models)
        
        self.handle_input()

    def handle_input(self):
        if 'current_utility' not in st.session_state or st.session_state.current_utility != self.utility:
            st.session_state.current_utility = self.utility
            st.session_state.url = None
            st.session_state.text = None
            st.session_state.chatbot = None

        if self.utility in ['Website', 'Youtube', 'Google Search']:
            st.session_state.url = st.text_input(f"Enter {self.utility}:")
        else:
            st.session_state.url = st.file_uploader("Upload PDF file", type=["pdf"])

        if st.button("Process and Summarize"):
            self.process_and_summarize()

        self.initialize_chat_messages()
        self.handle_chat_input()

    def process_and_summarize(self):
        if self.fetch_content():
            self.summarize_content()
        else:
            st.error("Error: Unable to fetch content. Please check the input and try again.")

    def fetch_content(self):
        try:
            if self.utility == 'Website':
                st.session_state.text = scraper(st.session_state.url)
            elif self.utility == 'Youtube':
                st.session_state.text = create_transcript_youtube(st.session_state.url)
            elif self.utility == 'Google Search':
                st.session_state.text = google_content_cleaned(st.session_state.url)
            elif self.utility == 'PDF' and st.session_state.url is not None:
                st.session_state.text = extract_pdf_text(st.session_state.url)

            st.session_state.last_url = st.session_state.url
            return st.session_state.text is not None
        except Exception as e:
            st.error(f"Error fetching content: {e}")
            return False

    def summarize_content(self):
        self.summary = summarizer(st.session_state.text, self.model)
        st.success("Summary:")
        st.write(self.summary)
        self.initialize_chatbot()

    def initialize_chatbot(self):
        if self.model == 'gemini-pro':
            st.session_state.chatbot = GBot(st.session_state.text)
        elif self.model == 'llama3-70b-8192':
            st.session_state.chatbot = LBot(st.session_state.text)
        else:
            st.session_state.chatbot = MBot(st.session_state.text)
        st.session_state.model = self.model

    def initialize_chat_messages(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_chat_input(self):
        prompt = st.chat_input(f"Ask a question about the content ({self.model}): ")
        if prompt:
            if self.check_for_updates():
                # st.warning("Content or model has changed.")
                self.process_and_summarize()
            
            self.handle_chat_message(prompt, "user")
            response = self.get_model_response(prompt)
            self.handle_chat_message(response, "assistant")

    def check_for_updates(self):
        return (st.session_state.url != st.session_state.last_url or 
                self.model != st.session_state.model)

    def get_model_response(self, prompt):
        try:
            if self.model in ['gemini-pro', 'mixtral-8x7b-32768']:
                return st.session_state.chatbot.qa_chain({"query": prompt})['result']
            elif self.model == 'llama3-70b-8192':
                return st.session_state.chatbot.final_result(prompt)['result']
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "Error: Unable to process the request."

    def handle_chat_message(self, content, role):
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.markdown(content)

if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    app = WebSummarizerChatbot()
    app.run()
