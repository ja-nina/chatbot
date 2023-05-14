import PyPDF2
import pandas as pd
import os
  
book_name = 'kafka-the-metamorphosis'
cwd = os.getcwd()
path = f"{cwd}/input/raw_books/kafka-the-metamorphosis.pdf"
end_path = f"{cwd}/input/processed_books/{book_name}.txt"
df_path = f"{cwd}/input/processed_books/{book_name}.csv"

encoding = 'utf-8'
DEFAULT_CONTEXT_LENGTH = 8 # context + response length

def preprocess_pdf():
    phrases_to_snip_out = ['Free eBooks at Planet eBoo k.com', " -"]
    text = """"""
    pdfFileObj = open(path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    # printing number of pages in pdf file
    print("number of pages in document:", pdfReader.numPages)
    
    for page_number in range(2, pdfReader.numPages):
        pageObj = pdfReader.getPage(page_number)
        pageText =pageObj.extractText()
        for subtexts in phrases_to_snip_out:
            pageText = pageText.replace(subtexts, '')
        text += pageText    
    pdfFileObj.close()

    with open(end_path, 'w', encoding="Windows-1252") as end_file:
        end_file.writelines(text.format(encoding=encoding))
        
def text_into_df():
    with open(end_path, 'r', encoding="Windows-1252") as end_file:
        text_corpora = end_file.read()
        text = text_corpora.replace("\n", "")
    
    text_sentences = text.split(".")
    # prepare dictionnary
    number_of_sentences = len(text_sentences)
    contexts = [text_sentences[DEFAULT_CONTEXT_LENGTH + c:number_of_sentences - DEFAULT_CONTEXT_LENGTH + c] for c in range(DEFAULT_CONTEXT_LENGTH - 1)]
    dictionnary_to_df = dict([(f"context{i}", context) for i, context in enumerate(contexts, start = 1)])
    dictionnary_to_df["response"] = text_sentences[(DEFAULT_CONTEXT_LENGTH + 1): -(DEFAULT_CONTEXT_LENGTH-1)]
    text_df = pd.DataFrame(dictionnary_to_df)
    
    text_df.to_csv(df_path, index=False, encoding="utf-8")

preprocess_pdf()
text_into_df()

