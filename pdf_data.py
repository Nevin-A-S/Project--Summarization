import PyPDF2

def extract_pdf_text(file_object):
    
    reader = PyPDF2.PdfReader(file_object)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

if __name__=='__main__':
    pdf_path = "C:/Users/LENOVO/Documents/Cerebral_palsy_Prediction_using_CNN_Depending_on_M.pdf"
    with open(pdf_path, 'rb') as file:
        pdf_text = extract_pdf_text(file)
    print(pdf_text)