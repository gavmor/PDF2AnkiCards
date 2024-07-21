import PyPDF2
import os
from llm import infer_flashcards
import time
from tqdm import tqdm

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

def create_anki_cards(pdf_text,output):
    SECTION_SIZE = 1000
    for i, text in tqdm(enumerate(divide_text(pdf_text, SECTION_SIZE))):
        time.sleep(1)
        
        try:
          with open(output, "a", encoding='utf-8') as f:
            f.write(infer_flashcards(text))

        except Exception as e:
            print(f"An error occurred in section {i}: {e}\n\n----------{text}\n\n===========")


def main(input, output):
    print("canary query: " + infer_flashcards("The boy was irony."))
    
    if not os.path.exists(input):
        print(f"Error: PDF file not found at {input}")
    else:
        create_anki_cards(read_pdf(input), output)

if __name__ == "__main__":
    main(PDF_FILE_PATH, FLASHCARDS_FILE_PATH)