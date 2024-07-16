import PyPDF2
import os
import llama_index
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
import time

client = Gemini(
  api_key=os.getenv("GEMINI_API_KEY"),
  model="models/gemini-1.5-pro")

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
        ChatMessage(role="system", content="You are a graduate-level Anki flashcard generator. Questions are to precede answers, joined with a semicolon (;) in the following format:\n\nquestion;answer."),
        ChatMessage(role="user", content=inference_text("The sky is blue.")),
        ChatMessage(role="assistant", content="What color is the sky?;Blue.\nWhat, in the text, is blue?;The sky."),
        ChatMessage(role="user", content=inference_text(text)),
    ]).message.content

def create_anki_cards(pdf_text,batch_size=5):
    SECTION_SIZE = 1000
    divided_sections = divide_text(pdf_text, SECTION_SIZE)
    generated_flashcards = ' '
    open("flashcards.txt", "w", encoding='utf-8').close()
    for i, text in enumerate(divided_sections):
        time.sleep(1)
        if i % batch_size == 0:
            print(f"Processing batch starting with section {i}")
            # Reset generated_flashcards for each batch
            generated_flashcards = ''
        
        try:
            response_from_api = query_llm(text)
            generated_flashcards += "\n" + response_from_api
                       
        except Exception as e:
            print(f"An error occurred in section {i}: {e}\n\n----------{text}\n\n===========")
            
        if i % batch_size == batch_size - 1 or i == len(divided_sections) - 1:
            print(f"Completed processing up to section {i}")

            with open("flashcards.txt", "a", encoding='utf-8') as f:
                f.write(generated_flashcards)

    print("Finished generating flashcards")



if __name__ == "__main__":
    print("s: " + query_llm("The boy was irony."))
    
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: PDF file not found at {PDF_FILE_PATH}")
    else:
        pdf_text = read_pdf(PDF_FILE_PATH)
        create_anki_cards(pdf_text)