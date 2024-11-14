# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import BitsAndBytesConfig
import gradio as gr
import zipfile
import os
import shutil
import easyocr
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
from transformers import BitsAndBytesConfig
import langchain as lc
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from pypdf import PdfReader
import pickle




# Bits and bytes Configuration for LLM model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  #nf8 for 8-bit quantization
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LLM Configuration and Model Loading
model_id = "meta-llama/Llama-3.2-3B-Instruct"

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_id,
    token="---"
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    token="---",
)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, token="---")

generate_text = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    temperature=0.5,
    max_new_tokens=800,
    repetition_penalty=1,
    do_sample=True
)

# Initialize EasyOCR reader for English
reader = easyocr.Reader(['en'])

# Embedding and vector store setup for multiple documents
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cuda"})
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

# Directories for PDFs and TXT files
pdf_directory_path = "pdf_db"
txt_directory_path = "txt_file_db"

# Function to read PDF files
def read_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    raw_text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Function to read txt files
def read_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()
    

# Function to load and process all PDFs in a directory
def load_pdfs_from_directory(directory_path):
    combined_text = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            text = read_pdf(pdf_path)
            combined_text.append(text)
    return combined_text

# Function to load and process all txt files in a directory
def load_txts_from_directory(directory_path):
    combined_text = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            txt_path = os.path.join(directory_path, filename)
            text = read_txt(txt_path)
            combined_text.append(text)
    return combined_text

# Function to save FAISS vector store
def save_faiss_vector_store(vector_store, vector_db_path):
    with open(vector_db_path, "wb") as f:
        pickle.dump(vector_store, f)

# Function to load FAISS vector store
def load_faiss_vector_store(vector_db_path):
    with open(vector_db_path, "rb") as f:
        return pickle.load(f)

# Function to process the uploaded zip file
def process_zip_file(zip_file):
    # Define paths for extraction and storing images and PDFs
    extract_path = "unzipped_folder"
    image_folder = 'images_db'
    pdf_folder = 'pdf_db'
    txt_file_db_dir = 'txt_file_db'

    # Create destination folders if they don't exist
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(txt_file_db_dir, exist_ok=True)

    file_list = []

    if zip_file is not None:
        # Extract the zip file
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Walk through the extracted folder and separate files
        for root, dirs, files in os.walk(extract_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_list.append(file_path)

                # Get the file extension in lowercase
                _, file_extension = os.path.splitext(filename)
                file_extension = file_extension.lower()

                # List of common image file extensions
                image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

                # Copy images to the image folder
                if file_extension in image_extensions:
                    shutil.copy(file_path, os.path.join(image_folder, filename))
                # Copy PDFs to the PDF folder
                elif file_extension == '.pdf':
                    shutil.copy(file_path, os.path.join(pdf_folder, filename))

    # OCR and text enhancement
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
            image_path = os.path.join(image_folder, filename)

            filename_txt = f"{os.path.splitext(filename)[0]}_enhanced_text.txt"
            file_path = os.path.join(txt_file_db_dir, filename_txt)

            if os.path.exists(file_path):
                continue

            image = cv2.imread(image_path)
            results = reader.readtext(image_path)

            all_text = ""
            for (bbox, text, confidence) in results:
                all_text += text + " "

            prompt = f"""
            The following text was extracted from an image using OCR and may have some errors or lack proper structure.
            Please correct any grammatical issues, enhance readability, and ensure it sounds natural and professional:

            Extracted Text:
            "{all_text.strip()}"

            Enhanced Text:
            """

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(**inputs, max_new_tokens=400, do_sample=True, top_p=0.95, top_k=60)

            processed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            enhanced_text = processed_text.split("Enhanced Text:")[1].strip()
            clean_text = enhanced_text.split("I made the following changes")[0].strip()

            with open(file_path, "w", encoding="utf-8") as file:
                file.write(clean_text)

    # Process PDFs and TXT files for vector database
    all_texts = []
    all_texts.extend(load_pdfs_from_directory(pdf_folder))
    all_texts.extend(load_txts_from_directory(txt_file_db_dir))

    # Split the text into chunks for embedding
    all_splitted_texts = []
    for text in all_texts:
        all_splitted_texts.extend(text_splitter.split_text(text))

    # Create vector store
    vector_store = FAISS.from_texts(all_splitted_texts, embedding)
    vector_db_path = "faiss_vector_store.pkl"
    save_faiss_vector_store(vector_store, vector_db_path)

    return f"Files have been processed and vector store saved. \n\nImage Files: {image_folder} \nPDF Files: {pdf_folder} \nEnhanced Text Files: {txt_file_db_dir}"

# Function for Question Answering using the loaded vector store
def question_answering(query):
    vector_db_path = "faiss_vector_store.pkl"
    vector_store = load_faiss_vector_store(vector_db_path)

    chain = ConversationalRetrievalChain.from_llm(
        HuggingFacePipeline(pipeline=generate_text), vector_store.as_retriever(), return_source_documents=True
    )

    chat_history = []
    answer = chain({
        "question": f"""{query} Please provide a response exclusively in English, drawing only from the dataset's information.
        If the question is beyond the dataset's content or cannot be answered based on the provided data,
        respond with an answer such as "I don't know about this information." I want a short answer of not more than 50 words. """,
        "chat_history": chat_history
    })

    # Process the response to extract the helpful answer
    text_store = []
    for text in answer["answer"].split("Helpful Answer: "):
        text_store.append(text)

    final_answer = ""
    for x in text_store[-1].split("\n"):
        final_answer += x.strip() + " "  # Concatenate all lines into one string with spaces

    return final_answer.strip()


# Gradio interface for file upload and processing
def upload_data(zip_file):
    if zip_file is None:
        return "Please upload a zip file containing the data."
    return process_zip_file(zip_file)

# Gradio interface for asking questions
def ask_question(question):
    if not os.path.exists("faiss_vector_store.pkl"):
        return "No data has been processed yet. Please upload data first."
    return question_answering(question)

# Set up the Gradio interface with separate buttons for uploading data and asking questions
with gr.Blocks() as interface:
    with gr.Row():
        data_upload = gr.File(label="Upload Data (Zip File)", file_count="single", file_types=[".zip"])
        upload_btn = gr.Button("Upload Data")
        upload_status = gr.Textbox(label="Processing Status", interactive=False)

        question_input = gr.Textbox(label="Ask a Question")
        question_btn = gr.Button("Ask Question")
        answer_output = gr.Textbox(label="Answer", interactive=False)

    upload_btn.click(upload_data, inputs=data_upload, outputs=upload_status)
    question_btn.click(ask_question, inputs=question_input, outputs=answer_output)

# Launch the interface
interface.launch(share=True)



