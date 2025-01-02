from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

def doc2str(docs):
    if not docs:
        return "No relevant documents found."
    return '\n\n'.join(doc.page_content for doc in docs)


def create_chain(retrievers):
    prompt_str = """ Answer the question based only on the following context: {context}
    Question: {question}
    Answer: """

    prompt = ChatPromptTemplate.from_template(prompt_str)

    llm = ChatGroq(model_name = "mixtral-8x7b-32768", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))
    chain = (
        {'context': retrievers | doc2str, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

