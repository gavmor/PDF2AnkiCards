import PyPDF2


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


def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages])
    return text