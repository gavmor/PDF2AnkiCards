import PyPDF2
import os
import llama_index
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
import time
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold

client = Gemini(
  api_key=os.getenv("GEMINI_API_KEY"),
  model="models/gemini-1.5-flash",
  safety_settings={
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE
  }
)

DEFAULT_PDF_NAME = 'linux-commands-handbook.pdf'
DEFAULT_FLASHCARDS_NAME = 'flashcards.txt'

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# Example:
# export PDF_FILE_PATH=/path/to/your/pdf.pdf
# export FLASHCARDS_FILE_PATH=/path/to/your/flashcards.txt
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH", os.path.join(ROOT_DIRECTORY, 'SOURCE_DOCUMENTS', DEFAULT_PDF_NAME))
FLASHCARDS_FILE_PATH = os.getenv("FLASHCARDS_FILE_PATH", os.path.join(ROOT_DIRECTORY, DEFAULT_FLASHCARDS_NAME))

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages])
    return text

def divide_text(text, section_size):
    sections = []
    start = 0
    end = section_size
    while start < len(text):
        section = text[start:end]
        sections.append(section)
        start = end
        end += section_size
    return sections

def inference_text(text):
    return f"Derive, from the following passage, a set of Anki (`question;answer`) flashcards:\n\n{text}"

def query_llm(text):
    return client.chat([
        ChatMessage(role="system", content="You are an Anki flashcard generator. Questions are to precede answers, joined with a semicolon (;) in the following format:\n\nquestion;answer. You are only interested in facts inferred from the text, but assume the reader does NOT have access to the text. Extract isolated propositions that stand on their own. Each passage is but one of many."),
        ChatMessage(role="user", content=inference_text("The sky is blue.")),
        ChatMessage(role="assistant", content="What color is the sky?;Blue.\nWhat, in the text, is blue?;The sky."),
        ChatMessage(role="user", content=inference_text(text)),
    ]).message.content

def   create_anki_cards(pdf_text,batch_size=5):
    SECTION_SIZE = 1000
    for i, text in tqdm(enumerate(divide_text(pdf_text, SECTION_SIZE))):
        time.sleep(1)
        
        try:
          with open(FLASHCARDS_FILE_PATH, "a", encoding='utf-8') as f:
            f.write(query_llm(text))

        except Exception as e:
            print(f"An error occurred in section {i}: {e}\n\n----------{text}\n\n===========")


if __name__ == "__main__":
    print("canary query: " + query_llm("The boy was irony."))
    
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: PDF file not found at {PDF_FILE_PATH}")
    else:
        create_anki_cards(read_pdf(PDF_FILE_PATH))