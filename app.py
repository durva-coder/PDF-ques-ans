
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import gradio as gr
from huggingface_hub import login
import os

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Optional: Auth for gated models
# login(token=os.getenv("HF_TOKEN"))

# Load documents function
def load_docs(pdf_path):
    print(f"\n🕵️‍♂️ Loading PDF: {pdf_path}")
    loaders_tried = []
    docs = []

    # PyMuPDF
    try:
        import fitz
        pdf = fitz.open(pdf_path)
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            if text:
                from langchain.schema import Document
                docs.append(Document(page_content=text, metadata={"page": i, "source": pdf_path}))
        print(f"✅ PyMuPDF extracted {len(docs)} pages")
        return docs
    except Exception as e:
        loaders_tried.append(f"PyMuPDF ✗ {e}")

    # pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    from langchain.schema import Document
                    docs.append(Document(page_content=text, metadata={"page": i, "source": pdf_path}))
        print(f"✅ pdfplumber extracted {len(docs)} pages")
        return docs
    except Exception as e:
        loaders_tried.append(f"pdfplumber ✗ {e}")

    # PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    from langchain.schema import Document
                    docs.append(Document(page_content=text, metadata={"page": i, "source": pdf_path}))
        print(f"✅ PyPDF2 extracted {len(docs)} pages")
        return docs
    except Exception as e:
        loaders_tried.append(f"PyPDF2 ✗ {e}")

    # Report all failures
    print("❌ All loaders failed:")
    for msg in loaders_tried:
        print("   •", msg)
    return []
# Create vector store
def create_vector_store(docs):
    if not docs:
        print("No documents provided to create vector store")
        return None
    
    # Verify documents have content
    valid_docs = []
    for doc in docs:
        if isinstance(doc, dict):
            content = doc.get('page_content', '')
        else:
            content = doc.page_content
            
        if content and str(content).strip():
            valid_docs.append(doc)
    
    if not valid_docs:
        print("All documents are empty after validation")
        return None
    
    print(f"Creating vector store with {len(valid_docs)} valid documents")
    
    try:
        # Initialize embeddings - forcing CPU to avoid GPU memory issues
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Test embeddings work
        test_text = "This is a test"
        test_embedding = embeddings.embed_query(test_text)
        if not isinstance(test_embedding, list) or len(test_embedding) != 384:
            print(f"Unexpected embedding format: {type(test_embedding)}, length: {len(test_embedding) if hasattr(test_embedding, '__len__') else 'N/A'}")
            return None
        
        # Create vector store
        try:
            vectorstore = FAISS.from_documents(valid_docs, embeddings)
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            print("Vector store created successfully")
            return retriever
        except Exception as e:
            print(f"Error creating FAISS index: {str(e)}")
            # Fallback to in-memory creation if FAISS fails
            try:
                from langchain.vectorstores import Chroma
                vectorstore = Chroma.from_documents(valid_docs, embeddings)
                retriever = vectorstore.as_retriever()
                print("Created Chroma vector store as fallback")
                return retriever
            except Exception as fallback_e:
                print(f"Chroma fallback also failed: {str(fallback_e)}")
                return None
                
    except Exception as e:
        print(f"Error in vector store creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Initialize LLM - with fallback options
def initialize_llm():
    try:
        if device == "cuda" and torch.cuda.is_available():
            # Try with quantization first (requires bitsandbytes)
            try:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model_name = "microsoft/DialoGPT-medium"  # Smaller, more reliable model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print("Using quantized model on GPU")
                
            except ImportError:
                print("Bitsandbytes not available, using standard GPU loading")
                model_name = "microsoft/DialoGPT-medium"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
        else:
            # CPU fallback
            print("Using CPU model")
            model_name = "microsoft/DialoGPT-small"  # Even smaller for CPU
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return HuggingFacePipeline(pipeline=pipe)
        
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        # Ultimate fallback - use a simple text generation
        return None

# Initialize the LLM
llm = initialize_llm()

# Answer question function
def answer_question(question, retriever_state):
    if not question.strip():
        return "Please enter a question."
    
    if retriever_state is None:
        return "Please upload a PDF first."
    
    if llm is None:
        return "LLM not available. Please check your installation."
    
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_state,
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        
        answer = result.get("result", "No answer generated")
        source_docs = result.get("source_documents", [])
        
        # Format sources
        sources = ""
        if source_docs:
            sources = "\n\n🔍 Sources:\n"
            for i, doc in enumerate(source_docs[:3]):  # Limit to 3 sources
                page_num = doc.metadata.get('page', 'Unknown')
                content_preview = doc.page_content[:150].replace('\n', ' ')
                sources += f"📄 Page {page_num}: {content_preview}...\n"
        
        return f"{answer}{sources}"
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Process PDF function
def process_pdf(pdf_file):
    if pdf_file is None:
        return None, "Please upload a PDF file."

    # 1. Resolve the real file path
    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    print(f"▶️ process_pdf got: {pdf_path}")

    # 2. Load & split text in one go
    try:
        loader = PyPDFLoader(pdf_path)
        # Optional: split into ~1,000-char chunks with 200-char overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = loader.load_and_split(text_splitter)
        print(f"✅ PyPDFLoader + splitter produced {len(docs)} chunks")
    except Exception as e:
        err = f"❌ Error loading/splitting PDF: {e}"
        print(err)
        return None, err

    # 3. Build vector store
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        vs = FAISS.from_documents(docs, embeddings)
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("✅ FAISS retriever ready")
        return retriever, f"✅ PDF processed: {len(docs)} text chunks indexed."
    except Exception as e:
        err = f"❌ Error creating vector store: {e}"
        print(err)
        return None, err
# Gradio Interface
with gr.Blocks(title="PDF Q&A System") as demo:
    gr.Markdown("# 📚 PDF Question & Answer System")
    gr.Markdown("Upload a PDF and ask questions about its content!")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF", 
                file_types=[".pdf"],
                type="filepath"
            )
            status_text = gr.Textbox(
                label="Status", 
                interactive=False,
                value="Ready to upload PDF..."
            )
            
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question about your PDF",
                placeholder="What is this document about?",
                lines=2
            )
            submit_btn = gr.Button("Get Answer", variant="primary")
            
    answer_output = gr.Textbox(
        label="Answer",
        lines=10,
        interactive=False
    )
    
    # State to store the retriever
    retriever_state = gr.State(value=None)
    
    # Event handlers
    pdf_input.change(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[retriever_state, status_text]
    )
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, retriever_state],
        outputs=[answer_output]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, retriever_state],
        outputs=[answer_output]
    )

# Launch the app

# if __name__ == "__main__":
#     demo.queue(max_size=1).launch(
#         share=False,
#         server_name="127.0.0.1",
#         server_port=7866
#     )

if __name__ == "__main__":
    import os
    demo.queue(max_size=1).launch(
        share=False,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )