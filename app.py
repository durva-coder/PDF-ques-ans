
import os
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# Hosted Gemini models: no local PyTorch/Transformers models
llm = None
embeddings = None


def get_llm():
    global llm

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it in Render Environment settings."
        )

    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=512,
        )

    return llm


def get_embeddings():
    global embeddings

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it in Render Environment settings."
        )

    if embeddings is None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )

    return embeddings


def process_pdf(pdf_file):
    if pdf_file is None:
        return None, "Please upload a PDF file."

    pdf_path = str(pdf_file)

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            return None, "No readable text found in the PDF."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            return None, "No text chunks could be created."

        embedding_model = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        return retriever, (
            f"PDF processed successfully. "
            f"{len(chunks)} text chunks indexed."
        )

    except Exception as e:
        print(f"PDF processing error: {e}")
        return None, f"PDF processing failed: {str(e)}"


def answer_question(question, retriever):
    if not question or not question.strip():
        return "Please enter a question."

    if retriever is None:
        return "Please upload and process a PDF first."

    try:
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            return "No relevant text was found in the PDF."

        context = "\n\n".join(
            f"Page {doc.metadata.get('page', 'Unknown') + 1 if isinstance(doc.metadata.get('page'), int) else doc.metadata.get('page', 'Unknown')}:\n"
            f"{doc.page_content}"
            for doc in relevant_docs
        )

        prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using only the context provided below.
If the answer is not present in the context, say that you could not
find the answer in the uploaded PDF. Do not invent facts.

PDF context:
{context}

User question:
{question}

Answer:
"""

        response = get_llm().invoke(prompt)
        answer = response.content

        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", str(item))
                if isinstance(item, dict)
                else str(item)
                for item in answer
            )

        sources = "\n\nSources:\n"
        for doc in relevant_docs:
            page = doc.metadata.get("page")
            page_label = page + 1 if isinstance(page, int) else "Unknown"
            sources += f"- Page {page_label}\n"

        return f"{answer}{sources}"

    except Exception as e:
        print(f"Question answering error: {e}")
        return (
            "Could not generate an answer. Check your API key and quota "
            f"in Render logs. Details: {str(e)}"
        )


with gr.Blocks(title="PDF Q&A System") as demo:
    gr.Markdown("# PDF Question & Answer System")
    gr.Markdown("Upload a PDF and ask questions about its content.")

    retriever_state = gr.State(value=None)

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready to upload PDF.",
            )

        with gr.Column():
            question_input = gr.Textbox(
                label="Ask a question about your PDF",
                placeholder="What is this document about?",
                lines=2,
            )
            submit_btn = gr.Button("Get Answer", variant="primary")

    answer_output = gr.Textbox(
        label="Answer",
        lines=10,
        interactive=False,
    )

    pdf_input.change(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[retriever_state, status_text],
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, retriever_state],
        outputs=[answer_output],
    )

    question_input.submit(
        fn=answer_question,
        inputs=[question_input, retriever_state],
        outputs=[answer_output],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7866))

    demo.queue(max_size=1).launch(
        share=False,
        server_name="0.0.0.0",
        server_port=port,
    )
