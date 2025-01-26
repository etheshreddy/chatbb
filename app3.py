import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import pytesseract       
from PIL import Image     
import pyttsx3 
import docx2txt
import pandas as pd
import tempfile
import uuid
from moviepy.editor import VideoFileClip
import ffmpeg
import speech_recognition as sr
import librosa
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
import zipfile
import io
from audio_recorder_streamlit import audio_recorder

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_image_text(file):
    pytesseract.pytesseract.tesseract_cmd ='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    img = Image.open(file) 
    result = pytesseract.image_to_string(img)    
    return result

def get_word_text(file):
    result = docx2txt.process(file)
    return result

def get_csv_text(file):
    df = pd.read_csv(file)
    text_representation = df.to_csv(index=False, sep="\t")
    return text_representation

def get_text(file):
    return file.read().decode("utf-8")

def copy_to_local(file):
    tempPath = tempfile.mkdtemp()
    local_path = Path(tempPath, file.name)
    with open(local_path, mode='wb') as w:
        w.write(file.getvalue())
    return local_path

def get_audio_file(file, chunk_length=60000):
    recognizer = sr.Recognizer()
    tempPath = tempfile.mkdtemp()
    audio = AudioSegment.from_wav(file)
    length_audio = len(audio)
    text = ""
    for _, chunk in enumerate(range(0, length_audio, chunk_length)):
        chunk_audio = audio[chunk:chunk + chunk_length]
        chunk_path = os.path.join(tempPath, uuid.uuid4().hex + ".wav")
        chunk_audio.export(chunk_path, format="wav")

        with sr.AudioFile(chunk_path) as source:
            audioFile = recognizer.record(source)
        try:
            text += recognizer.recognize_google(audioFile)
        except Exception as e:
            return ""
    return text

def get_video_file(file):
    tempPath = tempfile.mkdtemp()
    save_path = Path(tempPath, file.name)
    with open(save_path, mode='wb') as w:
        w.write(file.getvalue())

    audio_file = os.path.join(tempPath, uuid.uuid4().hex + ".wav")
    ffmpeg.input(save_path)\
        .output(audio_file, format='wav', acodec='pcm_s16le')\
        .run()
    return get_audio_file(audio_file)

def get_text_from_files(files):
    text= ""
    for file in files:
        print(file)
        content = ""
        if file.type == 'application/pdf':
            content = get_pdf_text(file)
        elif file.type == 'audio/wav':
            content = get_audio_file(copy_to_local(file))
        elif file.type == 'video/mp4':
            content = get_video_file(file)
        elif file.type == 'image/png':
            content = get_image_text(file)
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            content = get_word_text(file)
        elif file.type == 'text/csv':
            content = get_csv_text(file)
        elif file.type == "text/plain":
            content = get_text(file)
        elif file.type == 'application/zip':
            with zipfile.ZipFile(file, "r") as z:
                extracted_files = []
                for filename in z.namelist():
                    print(filename)
                    with z.open(filename) as extracted_file:
                        extracted_files.append(io.BytesIO(extracted_file.read()))
                        extracted_files[-1].name = filename
                        extension = filename.split('.')[-1]
                        if extension == 'pdf':
                            type = 'application/pdf'
                        elif extension == 'wav':
                            type = 'audio/wav'
                        elif extension == 'mp4':
                            type = 'video/mp4'
                        elif extension == 'png':
                            type = 'image/png'
                        elif extension == 'docx':
                            type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                        elif extension == 'csv':
                            type = 'text/csv'
                        elif extension == 'zip':
                            type = 'application/zip'
                        extracted_files[-1].type = type

                content = get_text_from_files(extracted_files)
        
        text += "Filename: " + file.name + "\nContent:" + content + "\n"
    print(text)
    return text

def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to retrieve the website")
        return ""
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator="\n")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """You are a friendly and informative assistant. Your role is to answer questions based on the context provided below. \
If the context does not include the answer, you must use your general knowledge to respond in a helpful and friendly manner. \
Do not say "I don't know." Always provide a meaningful response, even if it requires educated guessing or explaining the limits of what is possible. \
Use general knowledge to provide the best possible answer to the question. \
Context:\n{context}\n
Question:\n{question}\n
ANSWER:
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def handle_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    
    st.session_state.chat_history.insert(0, {"question": user_question, "answer": response["output_text"], "index": len(st.session_state.chat_history), "like": False})

st.session_state["audio_data"] = None
def update_question():
    print('update_question')
    if st.session_state["audio_data"]:
        print('audio_data')
        files = []
        st.audio(st.session_state["audio_data"], format="audio/wav")

        with open("recorded_audio.wav", "wb") as f:
            f.write(st.session_state["audio_data"])
        with open("recorded_audio.wav", "rb") as f:
            files.append(io.BytesIO(f.read()))
            files[-1].name = "recorded_audio.wav"
            files[-1].type = 'audio/wav'
            st.session_state.input_box = get_text_from_files(files).split('Content:')[-1]
            if st.session_state.input_box_prev_value != st.session_state.input_box:
                print('rerun')
                st.session_state.input_box_prev_value = st.session_state.input_box
                st.experimental_rerun()
            st.session_state["audio_data"] = None

def main():
    # import asyncio
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)

    st.set_page_config("Smart Query Agent")
    st.header("Smart Query Agent")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "input_box" not in st.session_state:
        st.session_state.input_box = ""
        st.session_state.input_box_prev_value = ""

    user_question = st.text_input("Ask a Question", value=st.session_state.input_box)
    if st.button("Submit Question"):
        if user_question:
            handle_user_input(user_question)
            st.experimental_rerun()

    for chat in st.session_state.chat_history:
        st.write(f"*You:* {chat['question']}")
        st.markdown(f"*Agent:* {chat['answer']}")
        if st.button(f"👍 Like", key=f"like_{chat['index']}"):
            chat['like'] = not chat['like']
        st.write("You liked this response!" if chat['like'] else "")
        
        st.write("---")  

    with st.sidebar:
        st.title("Menu:")
        files = st.file_uploader("Upload your files and Click on the Submit & Process Button", accept_multiple_files=True)
        website_url = st.text_input("Website URL")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_files(files)
                print(raw_text)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.title("Click to record your Question")
        
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.session_state["audio_data"] = audio_bytes
            update_question()

        if st.button("Process Website"):
            if website_url:
                with st.spinner("Processing Website..."):
                    website_text = scrape_website(website_url)
                    if website_text:
                        text_chunks = get_text_chunks(website_text)
                        get_vector_store(text_chunks)
                        st.success("Website processing done")
            else:
                st.error("Please enter a valid website URL")
                
    st.sidebar.markdown("""
    <a href="https://t.me/aibox123bot" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/8/82/Telegram_logo.svg" alt="Telegram Bot" style="width:30px;height:30px;">
        Chat with our Telegram Bot
    </a>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()