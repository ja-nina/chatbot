import os

import PyPDF2
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

book_name = "ferdydurke"
cwd = os.getcwd()
path = f"{cwd}/input/raw_books/{book_name}.pdf"
end_path = f"{cwd}/input/processed_books/{book_name}.txt"
df_path = f"{cwd}/input/processed_books/{book_name}.csv"
encoding = "utf-8"
DEF_CONTEXT_LEN = 8  # context + response length


def preprocess_pdf():
    phrases_to_snip_out = ["Free eBooks at Planet eBoo k.com", "-"]
    text = """"""
    pdf_file_obj = open(path, "rb")
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    # printing number of pages in pdf file
    print("number of pages in document:", pdf_reader.numPages)

    for page_number in range(2, pdf_reader.numPages):
        page_obj = pdf_reader.getPage(page_number)
        page_text = page_obj.extractText()
        for subtexts in phrases_to_snip_out:
            page_text = page_text.replace(subtexts, " ")
        text += page_text
    pdf_file_obj.close()
    with open(end_path, "w", encoding=encoding) as end_file:
        end_file.writelines(text)


def text_into_df():
    with open(end_path, "r", encoding=encoding) as end_file:
        text_corpora = end_file.read()
        text = text_corpora.replace("\n", " ").replace("\t", " ").replace(";", "")

    dos = nlp(text)
    text_sentences = [sent for sent in dos.sents]
    # prepare dictionary
    number_of_sentences = len(text_sentences)
    contexts = [
        text_sentences[DEF_CONTEXT_LEN + c: number_of_sentences - DEF_CONTEXT_LEN + c]
        for c in range(DEF_CONTEXT_LEN - 1)
    ]
    dictionary_to_df = {
        f"context{i}": context
        for i, context in enumerate(contexts, start=1)
    }
    dictionary_to_df["response"] = text_sentences[(DEF_CONTEXT_LEN + 1): -(DEF_CONTEXT_LEN - 1)]
    text_df = pd.DataFrame(dictionary_to_df)

    text_df.to_csv(df_path, index=False, encoding=encoding, sep=";")


def main():
    preprocess_pdf()
    text_into_df()
    return 0


if __name__ == "__main__":
    main()
