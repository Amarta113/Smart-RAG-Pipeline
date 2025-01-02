import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from rag_chain import create_chain 
from vectorstore import create_retriever


def process_input(input_type, input_data):
    if input_type == "PDF":
            loader = PyPDFLoader(input_data)
            documents = loader.load_and_split()
    else:
        raise ValueError("Invalid input data for PDF")


    #text splitter
    txt_splitter  = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20,
    length_function = len,
    is_separator_regex= False
    )

    doc_list = []

    for page in documents:
        if not page.page_content.strip():
            continue
        page_split = txt_splitter.split_text(page.page_content)

        for page_sub_split in page_split:
            metadata = {"source": "Book", "page_no": page.metadata.get('page', 0) + 1 }
            doc_string = Document(page_content=page_sub_split, metadata=metadata)
            doc_list.append(doc_string)

    return doc_list


def main():
    st.set_page_config(page_title="Talk with PDF application", page_icon="🦜", layout="wide")
    st.title('🦜🔗 Talk with PDF application')

    uploaded_file = st.file_uploader("Upload a PDF file", type='pdf')

    if uploaded_file is not None:
         st.write("Processing the uploaded file...")
         doc_list = process_input('PDF', uploaded_file)

    model_name = 'BAAI/bge-small-en'    
    embed_model = HuggingFaceBgeEmbeddings(model_name=model_name)

    # Create retriever
    retriever = create_retriever(doc_list, embed_model) 

    st.text_input("Ask the question from the PDF", key='user_question')

    user_question = st.session_state.get("user_question", "")

    if user_question:
        chain = create_chain(retriever)
        answer = chain.invoke({"question": user_question})
        st.write(f"Answer: {answer}")    


if __name__ == '__main__':
    main()

    
