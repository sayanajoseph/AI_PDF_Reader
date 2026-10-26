
import streamlit as st
from io import BytesIO
from uuid import uuid4
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


st.set_page_config(page_title="AI PDF Reader")
st.title("AI PDF Reader")
st.write("Upload a PDF and ask questions about it.")


@st.cache_resource
def load_models():
    llm = ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0
    )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    return llm, embeddings


if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    
    if pdf_bytes != st.session_state.pdf_bytes:

        
        st.session_state.pdf_bytes = None
        st.session_state.vectorstore = None
        st.session_state.llm = None

        try:
            
            with st.spinner("Reading PDF..."):

                reader = PdfReader(BytesIO(pdf_bytes))
                documents = []

                total_pages = len(reader.pages)
                extraction_progress = st.progress(0)

                for page_number, page in enumerate(
                    reader.pages, start=1
                ):
                    text = page.extract_text() or ""

                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"page": page_number}
                            )
                        )

                    if page_number % 10 == 0 or page_number == total_pages:
                        extraction_progress.progress(
                            page_number / total_pages
                        )

            if not documents:
                st.error(
                    "No readable text found. "
                    "This PDF may contain scanned images."
                )
                st.stop()

            st.success(
                f"Extracted text from {len(documents)} pages."
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(documents)

            total_chunks = len(chunks)
            st.write(f"Total chunks: {total_chunks}")

            with st.spinner("Loading local models..."):
                llm, embeddings = load_models()

            vectorstore = Chroma(
                collection_name=f"pdf_{uuid4().hex}",
                embedding_function=embeddings
            )

            batch_size = 16
            progress = st.progress(0)
            status = st.empty()

            for start in range(0, total_chunks, batch_size):

                batch = chunks[start:start + batch_size]

                vectorstore.add_documents(
                    documents=batch,
                    ids=[str(uuid4()) for _ in batch]
                )

                completed = min(
                    start + len(batch),
                    total_chunks
                )

                progress.progress(completed / total_chunks)

                status.write(
                    f"Indexing PDF: {completed}/{total_chunks} chunks"
                )

           
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.vectorstore = vectorstore
            st.session_state.llm = llm
            st.session_state.page_count = len(documents)
            st.session_state.chunk_count = total_chunks

            st.success("PDF indexed successfully!")

        except Exception as e:
            st.session_state.pdf_bytes = None
            st.session_state.vectorstore = None
            st.session_state.llm = None

            st.error(f"Error processing PDF: {e}")



if st.session_state.vectorstore is not None:

    st.write("Pages:", st.session_state.page_count)
    st.write("Chunks:", st.session_state.chunk_count)

    
    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if st.button("Get Answer"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Retrieving information and generating answer..."
                ):

                    
                    relevant_chunks = (
                        st.session_state.vectorstore
                        .similarity_search(question, k=4)
                    )

                    context = "\n\n".join(
                        f"Page {doc.metadata['page']}:\n"
                        f"{doc.page_content}"
                        for doc in relevant_chunks
                    )

                    prompt = PromptTemplate.from_template("""
Answer using only the PDF passages below.
If the answer is not present, say:
"I could not find the answer in the uploaded PDF."
Use simple English and do not invent facts.

PDF passages:
{context}

Question: {question}

Answer:
""")

                    chain = (
                        prompt
                        | st.session_state.llm
                        | StrOutputParser()
                    )

                    answer = chain.invoke({
                        "context": context,
                        "question": question
                    })

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source pages")

                pages = sorted({
                    doc.metadata["page"]
                    for doc in relevant_chunks
                })

                st.write(", ".join(map(str, pages)))

            except Exception as e:
                st.error(f"Error generating answer: {e}")
