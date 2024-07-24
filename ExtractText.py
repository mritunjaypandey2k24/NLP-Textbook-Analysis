from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Paths to my PDF files
pdf_paths = [
    r'C:\Users\MRITUNJAY\Downloads\NLP_Project\Textbooks\Ajay D. Kshemkalyani, Mukesh Singhal - Distributed computing_ principles, algorithms, and systems-Cambridge University Press (2008).pdf',
    r'C:\Users\MRITUNJAY\Downloads\NLP_Project\Textbooks\(Adaptive Computation and Machine Learning) Ralf Herbrich - Learning Kernel Classifiers_ Theory and Algorithms-The MIT Press (2001).pdf',
    r'C:\Users\MRITUNJAY\Downloads\NLP_Project\Textbooks\Joel S. Cohen - Computer algebra and symbolic computation_ elementary algorithms-A K Peters_CRC Press (2002).pdf'
]

# To Extract content from each PDF
for pdf_path in pdf_paths:
    content = extract_text_from_pdf(pdf_path)
    with open('output_text_file.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(content)
    print(f'Extracted content from {pdf_path}')
