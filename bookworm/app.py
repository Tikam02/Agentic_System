# # #!/usr/bin/env python3
# # """
# # Local PDF Chat using Ollama with Phi3:Mini
# # A complete RAG implementation for chatting with PDF documents locally.

# # Prerequisites:
# # 1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
# # 2. Pull models: 
# #    - ollama pull phi3:mini
# #    - ollama pull nomic-embed-text
# # 3. Install requirements: pip install -r requirements.txt
# # """

# # import os
# # import logging
# # from pathlib import Path
# # from typing import List, Optional
# # import sys

# # # Core LangChain imports
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain_community.embeddings import OllamaEmbeddings
# # from langchain_community.chat_models import ChatOllama
# # from langchain_community.vectorstores import Chroma
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain.memory import ConversationBufferMemory
# # from langchain.prompts import PromptTemplate

# # # Configure logging
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(levelname)s - %(message)s'
# # )
# # logger = logging.getLogger(__name__)

# # class PDFChatBot:
# #     """Local PDF Chat using Ollama with Phi3:Mini"""
    
# #     def __init__(
# #         self,
# #         llm_model: str = "phi3:mini",
# #         embedding_model: str = "nomic-embed-text",
# #         chunk_size: int = 1000,
# #         chunk_overlap: int = 200,
# #         persist_directory: str = "./chroma_db"
# #     ):
# #         self.llm_model = llm_model
# #         self.embedding_model = embedding_model
# #         self.chunk_size = chunk_size
# #         self.chunk_overlap = chunk_overlap
# #         self.persist_directory = persist_directory
        
# #         # Initialize components
# #         self.embeddings = None
# #         self.llm = None
# #         self.vectorstore = None
# #         self.conversation_chain = None
# #         self.memory = None
        
# #         self._initialize_components()
    
# #     def _initialize_components(self):
# #         """Initialize LLM, embeddings, and memory"""
# #         try:
# #             logger.info(f"Initializing Ollama LLM: {self.llm_model}")
# #             self.llm = ChatOllama(
# #                 model=self.llm_model,
# #                 temperature=0.1,
# #                 top_p=0.9
# #             )
            
# #             logger.info(f"Initializing Ollama embeddings: {self.embedding_model}")
# #             self.embeddings = OllamaEmbeddings(
# #                 model=self.embedding_model,
# #                 show_progress=True
# #             )
            
# #             # Initialize conversation memory
# #             self.memory = ConversationBufferMemory(
# #                 memory_key="chat_history",
# #                 return_messages=True,
# #                 output_key="answer"
# #             )
            
# #             logger.info("Components initialized successfully")
            
# #         except Exception as e:
# #             logger.error(f"Failed to initialize components: {e}")
# #             raise
    
# #     def load_pdf(self, pdf_path: str) -> List:
# #         """Load and process PDF document"""
# #         if not os.path.exists(pdf_path):
# #             raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
# #         logger.info(f"Loading PDF: {pdf_path}")
        
# #         try:
# #             # Load PDF
# #             loader = PyPDFLoader(pdf_path)
# #             pages = loader.load()
            
# #             logger.info(f"Loaded {len(pages)} pages from PDF")
            
# #             # Split into chunks
# #             text_splitter = RecursiveCharacterTextSplitter(
# #                 chunk_size=self.chunk_size,
# #                 chunk_overlap=self.chunk_overlap,
# #                 length_function=len,
# #                 separators=["\n\n", "\n", " ", ""]
# #             )
            
# #             chunks = text_splitter.split_documents(pages)
# #             logger.info(f"Created {len(chunks)} text chunks")
            
# #             return chunks
            
# #         except Exception as e:
# #             logger.error(f"Error loading PDF: {e}")
# #             raise
    
# #     def create_vectorstore(self, documents: List, collection_name: str = "pdf_docs"):
# #         """Create or load vector store from documents"""
# #         try:
# #             logger.info("Creating vector store with ChromaDB")
            
# #             # Create vector store
# #             self.vectorstore = Chroma.from_documents(
# #                 documents=documents,
# #                 embedding=self.embeddings,
# #                 collection_name=collection_name,
# #                 persist_directory=self.persist_directory
# #             )
            
# #             # Persist the database
# #             self.vectorstore.persist()
# #             logger.info(f"Vector store created with {len(documents)} documents")
            
# #         except Exception as e:
# #             logger.error(f"Error creating vector store: {e}")
# #             raise
    
# #     def load_existing_vectorstore(self, collection_name: str = "pdf_docs"):
# #         """Load existing vector store"""
# #         try:
# #             if os.path.exists(self.persist_directory):
# #                 logger.info("Loading existing vector store")
# #                 self.vectorstore = Chroma(
# #                     collection_name=collection_name,
# #                     embedding_function=self.embeddings,
# #                     persist_directory=self.persist_directory
# #                 )
# #                 logger.info("Existing vector store loaded successfully")
# #                 return True
# #             return False
# #         except Exception as e:
# #             logger.error(f"Error loading existing vector store: {e}")
# #             return False
    
# #     def setup_conversation_chain(self):
# #         """Setup the conversational retrieval chain"""
# #         if not self.vectorstore:
# #             raise ValueError("Vector store not initialized. Load a PDF first.")
        
# #         try:
# #             # Create custom prompt template
# #             custom_prompt = PromptTemplate(
# #                 template="""You are an AI assistant helping users understand PDF documents. 
# #                 Use the following context to answer the question. If you cannot find the answer 
# #                 in the context, say "I cannot find that information in the provided document."

# #                 Context: {context}
# #                 Chat History: {chat_history}
# #                 Question: {question}

# #                 Answer: """,
# #                 input_variables=["context", "chat_history", "question"]
# #             )
            
# #             # Create retriever
# #             retriever = self.vectorstore.as_retriever(
# #                 search_type="similarity",
# #                 search_kwargs={"k": 4}
# #             )
            
# #             # Create conversation chain
# #             self.conversation_chain = ConversationalRetrievalChain.from_llm(
# #                 llm=self.llm,
# #                 retriever=retriever,
# #                 memory=self.memory,
# #                 return_source_documents=True,
# #                 verbose=True,
# #                 combine_docs_chain_kwargs={"prompt": custom_prompt}
# #             )
            
# #             logger.info("Conversation chain setup successfully")
            
# #         except Exception as e:
# #             logger.error(f"Error setting up conversation chain: {e}")
# #             raise
    
# #     def chat(self, question: str) -> dict:
# #         """Chat with the PDF"""
# #         if not self.conversation_chain:
# #             raise ValueError("Conversation chain not initialized. Setup the chain first.")
        
# #         try:
# #             logger.info(f"Processing question: {question}")
            
# #             # Get response
# #             response = self.conversation_chain({"question": question})
            
# #             return {
# #                 "answer": response["answer"],
# #                 "source_documents": response.get("source_documents", []),
# #                 "chat_history": response.get("chat_history", [])
# #             }
            
# #         except Exception as e:
# #             logger.error(f"Error during chat: {e}")
# #             raise
    
# #     def process_pdf_and_setup(self, pdf_path: str, collection_name: str = "pdf_docs"):
# #         """Complete pipeline: load PDF, create vector store, setup chain"""
# #         try:
# #             # Try to load existing vector store first
# #             if not self.load_existing_vectorstore(collection_name):
# #                 # Load and process new PDF
# #                 documents = self.load_pdf(pdf_path)
# #                 self.create_vectorstore(documents, collection_name)
            
# #             # Setup conversation chain
# #             self.setup_conversation_chain()
            
# #             logger.info("PDF processing and setup completed successfully")
            
# #         except Exception as e:
# #             logger.error(f"Error in PDF processing pipeline: {e}")
# #             raise


# # def main():
# #     """Main function demonstrating usage"""
    
# #     # Configuration
# #     PDF_PATH = "./Metamorphosis.pdf"  # Replace with your PDF path
    
# #     try:
# #         # Initialize chatbot
# #         print("🤖 Initializing PDF ChatBot with Ollama & Phi3:Mini...")
# #         chatbot = PDFChatBot()
        
# #         # Check if PDF exists
# #         if not os.path.exists(PDF_PATH):
# #             print(f"❌ Error: PDF file '{PDF_PATH}' not found!")
# #             print("Please update PDF_PATH variable with your PDF file path.")
# #             return
        
# #         # Process PDF and setup
# #         print(f"📄 Processing PDF: {PDF_PATH}")
# #         chatbot.process_pdf_and_setup(PDF_PATH)
        
# #         print("\n✅ Setup completed! You can now chat with your PDF.")
# #         print("Type 'quit' or 'exit' to stop.\n")
        
# #         # Interactive chat loop
# #         while True:
# #             try:
# #                 question = input("🔍 Ask a question: ").strip()
                
# #                 if question.lower() in ['quit', 'exit', 'q']:
# #                     print("👋 Goodbye!")
# #                     break
                
# #                 if not question:
# #                     continue
                
# #                 print("🤔 Thinking...")
                
# #                 # Get response
# #                 result = chatbot.chat(question)
                
# #                 print(f"\n💡 Answer: {result['answer']}")
                
# #                 # Show sources if available
# #                 sources = result.get('source_documents', [])
# #                 if sources:
# #                     print(f"\n📚 Sources ({len(sources)} documents):")
# #                     for i, doc in enumerate(sources[:2], 1):  # Show top 2 sources
# #                         page = doc.metadata.get('page', 'Unknown')
# #                         content_preview = doc.page_content[:100].replace('\n', ' ')
# #                         print(f"  {i}. Page {page}: {content_preview}...")
                
# #                 print("-" * 60)
                
# #             except KeyboardInterrupt:
# #                 print("\n👋 Goodbye!")
# #                 break
# #             except Exception as e:
# #                 print(f"❌ Error: {e}")
                
# #     except Exception as e:
# #         print(f"❌ Failed to initialize chatbot: {e}")
# #         print("\nTroubleshooting:")
# #         print("1. Ensure Ollama is running: ollama serve")
# #         print("2. Check if models are installed:")
# #         print("   - ollama pull phi3:mini")
# #         print("   - ollama pull nomic-embed-text")
# #         print("3. Install required packages: pip install -r requirements.txt")


# # if __name__ == "__main__":
# #     main() 




# import streamlit as st
# import os
# import tempfile
# import logging
# from pathlib import Path
# from typing import List, Optional

# # Core LangChain imports
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.chat_models import ChatOllama
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class PDFChatBot:
#     """Local PDF Chat using Ollama with Phi3:Mini"""
    
#     def __init__(
#         self,
#         llm_model: str = "phi3:mini",
#         embedding_model: str = "nomic-embed-text",
#         chunk_size: int = 1000,
#         chunk_overlap: int = 200,
#         persist_directory: str = "./chroma_db",
#         ollama_base_url: str = "http://localhost:11434"
#     ):
#         self.llm_model = llm_model
#         self.embedding_model = embedding_model
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.persist_directory = persist_directory
#         self.ollama_base_url = ollama_base_url
        
#         # Initialize components
#         self.embeddings = None
#         self.llm = None
#         self.vectorstore = None
#         self.conversation_chain = None
#         self.memory = None
        
#         self._initialize_components()
    
#     def _initialize_components(self):
#         """Initialize LLM, embeddings, and memory"""
#         try:
#             self.llm = ChatOllama(
#                 model=self.llm_model,
#                 temperature=0.1,
#                 top_p=0.9,
#                 base_url=self.ollama_base_url
#             )
            
#             self.embeddings = OllamaEmbeddings(
#                 model=self.embedding_model,
#                 base_url=self.ollama_base_url
#             )
            
#             self.memory = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 return_messages=True,
#                 output_key="answer"
#             )
            
#         except Exception as e:
#             logger.error(f"Failed to initialize components: {e}")
#             raise
    
#     def load_pdf(self, pdf_path: str) -> List:
#         """Load and process PDF document"""
#         if not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
#         try:
#             loader = PyPDFLoader(pdf_path)
#             pages = loader.load()
            
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=self.chunk_size,
#                 chunk_overlap=self.chunk_overlap,
#                 length_function=len,
#                 separators=["\n\n", "\n", " ", ""]
#             )
            
#             chunks = text_splitter.split_documents(pages)
#             return chunks
            
#         except Exception as e:
#             logger.error(f"Error loading PDF: {e}")
#             raise
    
#     def create_vectorstore(self, documents: List, collection_name: str = "pdf_docs"):
#         """Create vector store from documents"""
#         try:
#             self.vectorstore = Chroma.from_documents(
#                 documents=documents,
#                 embedding=self.embeddings,
#                 collection_name=collection_name,
#                 persist_directory=self.persist_directory
#             )
#             self.vectorstore.persist()
            
#         except Exception as e:
#             logger.error(f"Error creating vector store: {e}")
#             raise
    
#     def setup_conversation_chain(self):
#         """Setup the conversational retrieval chain"""
#         if not self.vectorstore:
#             raise ValueError("Vector store not initialized.")
        
#         try:
#             custom_prompt = PromptTemplate(
#                 template="""You are an AI assistant helping users understand PDF documents. 
#                 Use the following context to answer the question. If you cannot find the answer 
#                 in the context, say "I cannot find that information in the provided document."

#                 Context: {context}
#                 Chat History: {chat_history}
#                 Question: {question}

#                 Answer: """,
#                 input_variables=["context", "chat_history", "question"]
#             )
            
#             retriever = self.vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={"k": 4}
#             )
            
#             self.conversation_chain = ConversationalRetrievalChain.from_llm(
#                 llm=self.llm,
#                 retriever=retriever,
#                 memory=self.memory,
#                 return_source_documents=True,
#                 combine_docs_chain_kwargs={"prompt": custom_prompt}
#             )
            
#         except Exception as e:
#             logger.error(f"Error setting up conversation chain: {e}")
#             raise
    
#     def chat(self, question: str) -> dict:
#         """Chat with the PDF"""
#         if not self.conversation_chain:
#             raise ValueError("Conversation chain not initialized.")
        
#         try:
#             response = self.conversation_chain({"question": question})
#             return {
#                 "answer": response["answer"],
#                 "source_documents": response.get("source_documents", [])
#             }
            
#         except Exception as e:
#             logger.error(f"Error during chat: {e}")
#             raise


# def initialize_session_state():
#     """Initialize Streamlit session state"""
#     if 'chatbot' not in st.session_state:
#         st.session_state.chatbot = None
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'pdf_processed' not in st.session_state:
#         st.session_state.pdf_processed = False


# def main():
#     """Main Streamlit application"""
#     st.set_page_config(
#         page_title="PDF Chat with Ollama",
#         page_icon="📄",
#         layout="wide"
#     )
    
#     initialize_session_state()
    
#     st.title("📄 PDF Chat with Ollama & Phi3:Mini")
#     st.markdown("Upload a PDF and chat with its contents using local AI models.")
    
#     # Sidebar for configuration
#     with st.sidebar:
#         st.header("Configuration")
        
#         ollama_url = st.text_input(
#             "Ollama Base URL", 
#             value="http://localhost:11434",
#             help="URL where Ollama service is running"
#         )
        
#         st.header("Upload PDF")
#         uploaded_file = st.file_uploader(
#             "Choose a PDF file",
#             type="pdf",
#             help="Upload a PDF document to chat with"
#         )
        
#         if uploaded_file is not None:
#             if st.button("Process PDF", type="primary"):
#                 with st.spinner("Processing PDF..."):
#                     try:
#                         # Save uploaded file to temporary location
#                         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                             tmp_file.write(uploaded_file.getvalue())
#                             tmp_path = tmp_file.name
                        
#                         # Initialize chatbot
#                         st.session_state.chatbot = PDFChatBot(ollama_base_url=ollama_url)
                        
#                         # Process PDF
#                         documents = st.session_state.chatbot.load_pdf(tmp_path)
#                         st.session_state.chatbot.create_vectorstore(documents)
#                         st.session_state.chatbot.setup_conversation_chain()
                        
#                         st.session_state.pdf_processed = True
#                         st.session_state.messages = []  # Clear previous messages
                        
#                         # Cleanup temp file
#                         os.unlink(tmp_path)
                        
#                         st.success(f"PDF processed successfully! Found {len(documents)} text chunks.")
                        
#                     except Exception as e:
#                         st.error(f"Error processing PDF: {str(e)}")
#                         logger.error(f"PDF processing error: {e}")
        
#         # Model status
#         st.header("Model Status")
#         if st.session_state.pdf_processed:
#             st.success("✅ Ready to chat")
#         else:
#             st.warning("⏳ Upload and process a PDF first")
    
#     # Main chat interface
#     if st.session_state.pdf_processed and st.session_state.chatbot:
#         st.header("💬 Chat with your PDF")
        
#         # Display chat messages
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
                
#                 # Show sources for assistant messages
#                 if message["role"] == "assistant" and "sources" in message:
#                     with st.expander("📚 Sources"):
#                         for i, source in enumerate(message["sources"], 1):
#                             page = source.metadata.get('page', 'Unknown')
#                             content_preview = source.page_content[:200].replace('\n', ' ')
#                             st.text(f"{i}. Page {page}: {content_preview}...")
        
#         # Chat input
#         if prompt := st.chat_input("Ask a question about your PDF..."):
#             # Add user message
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)
            
#             # Get assistant response
#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking..."):
#                     try:
#                         result = st.session_state.chatbot.chat(prompt)
#                         response = result["answer"]
#                         sources = result.get("source_documents", [])
                        
#                         st.markdown(response)
                        
#                         # Add assistant message with sources
#                         st.session_state.messages.append({
#                             "role": "assistant", 
#                             "content": response,
#                             "sources": sources
#                         })
                        
#                         # Show sources
#                         if sources:
#                             with st.expander("📚 Sources"):
#                                 for i, source in enumerate(sources, 1):
#                                     page = source.metadata.get('page', 'Unknown')
#                                     content_preview = source.page_content[:200].replace('\n', ' ')
#                                     st.text(f"{i}. Page {page}: {content_preview}...")
                        
#                     except Exception as e:
#                         error_msg = f"Error: {str(e)}"
#                         st.error(error_msg)
#                         st.session_state.messages.append({
#                             "role": "assistant", 
#                             "content": error_msg
#                         })
        
#         # Clear chat button
#         if st.button("🗑️ Clear Chat"):
#             st.session_state.messages = []
#             st.rerun()
    
#     else:
#         st.info("👆 Please upload and process a PDF in the sidebar to start chatting.")
        
#         # Instructions
#         with st.expander("📋 Setup Instructions"):
#             st.markdown("""
#             **Prerequisites:**
#             1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
#             2. Pull required models:
#                ```bash
#                ollama pull phi3:mini
#                ollama pull nomic-embed-text
#                ```
#             3. Start Ollama service: `ollama serve`
            
#             **Usage:**
#             1. Upload a PDF file using the sidebar
#             2. Click "Process PDF" to analyze the document
#             3. Start chatting with your PDF content!
#             """)


# if __name__ == "__main__":
#     main()



import streamlit as st
import os
import tempfile
import logging
import requests
from pathlib import Path
from typing import List, Optional

# Updated LangChain imports for newer versions
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFChatBot:
    """Local PDF Chat using Ollama"""
    
    def __init__(
        self,
        llm_model: str = "phi3:mini",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_base_url = ollama_base_url
        
        # Initialize components
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.conversation_chain = None
        self.memory = None
        
        self._initialize_components()
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_model_availability(self, model_name: str) -> bool:
        """Check if a specific model is available in Ollama"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                return any(model_name in model for model in available_models)
            return False
        except:
            return False
    
    def _initialize_components(self):
        """Initialize LLM, embeddings, and memory"""
        try:
            # Check Ollama connection
            if not self._check_ollama_connection():
                raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_base_url}")
            
            # Check if models are available
            if not self._check_model_availability(self.llm_model):
                raise ValueError(f"Model {self.llm_model} not found. Run: ollama pull {self.llm_model}")
            
            if not self._check_model_availability(self.embedding_model):
                raise ValueError(f"Model {self.embedding_model} not found. Run: ollama pull {self.embedding_model}")
            
            # Initialize LLM (using Ollama instead of ChatOllama for better compatibility)
            self.llm = Ollama(
                model=self.llm_model,
                base_url=self.ollama_base_url,
                temperature=0.1
            )
            
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.ollama_base_url
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def load_pdf(self, pdf_path: str) -> List:
        """Load and process PDF document"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                raise ValueError("No content found in PDF")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(pages)
            logger.info(f"PDF processed into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def create_vectorstore(self, documents: List):
        """Create vector store from documents"""
        try:
            # Create a unique collection name based on document content
            collection_name = f"pdf_docs_{hash(str(documents[0].page_content[:100])) % 10000}"
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name
            )
            
            logger.info(f"Vector store created with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def setup_conversation_chain(self):
        """Setup the conversational retrieval chain"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        try:
            # Custom prompt template
            custom_prompt = PromptTemplate(
                template="""Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find that information in the provided document."

Context: {context}

Question: {question}

Answer:""",
                input_variables=["context", "question"]
            )
            
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Create conversation chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": custom_prompt}
            )
            
            logger.info("Conversation chain setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up conversation chain: {e}")
            raise
    
    def chat(self, question: str) -> dict:
        """Chat with the PDF"""
        if not self.conversation_chain:
            raise ValueError("Conversation chain not initialized.")
        
        try:
            response = self.conversation_chain({"question": question})
            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", [])
            }
            
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            raise


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False


def check_ollama_status(ollama_url: str) -> tuple[bool, list]:
    """Check Ollama status and available models"""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            return True, model_names
        return False, []
    except:
        return False, []


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="PDF Chat with Ollama",
        page_icon="📄",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("📄 PDF Chat with Ollama")
    st.markdown("Upload a PDF and chat with its contents using local Ollama models.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        ollama_url = st.text_input(
            "Ollama Base URL", 
            value="http://localhost:11434",
            help="URL where Ollama service is running"
        )
        
        # Check Ollama status
        ollama_connected, available_models = check_ollama_status(ollama_url)
        
        if ollama_connected:
            st.success("✅ Ollama is running")
            st.write("**Available Models:**")
            for model in available_models:
                st.text(f"• {model}")
        else:
            st.error("❌ Cannot connect to Ollama")
            st.write("Make sure Ollama is running: `ollama serve`")
        
        # Model selection
        st.subheader("Model Configuration")
        
        llm_model = st.selectbox(
            "LLM Model",
            options=["phi3:mini", "llama2", "mistral", "codellama"] + available_models,
            index=0,
            help="Choose the language model for chat"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model", 
            options=["nomic-embed-text", "all-minilm"] + available_models,
            index=0,
            help="Choose the embedding model for document processing"
        )
        
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None and ollama_connected:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Initialize chatbot with selected models
                        st.session_state.chatbot = PDFChatBot(
                            llm_model=llm_model,
                            embedding_model=embedding_model,
                            ollama_base_url=ollama_url
                        )
                        
                        # Process PDF
                        documents = st.session_state.chatbot.load_pdf(tmp_path)
                        st.session_state.chatbot.create_vectorstore(documents)
                        st.session_state.chatbot.setup_conversation_chain()
                        
                        st.session_state.pdf_processed = True
                        st.session_state.messages = []  # Clear previous messages
                        
                        # Cleanup temp file
                        os.unlink(tmp_path)
                        
                        st.success(f"✅ PDF processed! Found {len(documents)} chunks.")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"PDF processing error: {e}")
        
        # Status indicator
        st.header("📊 Status")
        if st.session_state.pdf_processed:
            st.success("🟢 Ready to chat")
        else:
            st.warning("🟡 Upload and process a PDF first")
    
    # Main chat interface
    if st.session_state.pdf_processed and st.session_state.chatbot:
        st.header("💬 Chat with your PDF")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            page = source.metadata.get('page', 'Unknown')
                            content_preview = source.page_content[:200].replace('\n', ' ')
                            st.text(f"{i}. Page {page}: {content_preview}...")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = st.session_state.chatbot.chat(prompt)
                        response = result["answer"]
                        sources = result.get("source_documents", [])
                        
                        st.markdown(response)
                        
                        # Add assistant message with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                        # Show sources
                        if sources:
                            with st.expander("📚 Sources", expanded=False):
                                for i, source in enumerate(sources, 1):
                                    page = source.metadata.get('page', 'Unknown')
                                    content_preview = source.page_content[:200].replace('\n', ' ')
                                    st.text(f"{i}. Page {page}: {content_preview}...")
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    else:
        st.info("👆 Please upload and process a PDF in the sidebar to start chatting.")
        
        # Setup instructions
        with st.expander("📋 Setup Instructions", expanded=not ollama_connected):
            st.markdown("""
            **1. Install Ollama:**
            ```bash
            # Linux/Mac
            curl -fsSL https://ollama.com/install.sh | sh
            
            # Windows: Download from https://ollama.com
            ```
            
            **2. Start Ollama:**
            ```bash
            ollama serve
            ```
            
            **3. Pull required models:**
            ```bash
            ollama pull phi3:mini
            ollama pull nomic-embed-text
            ```
            
            **4. Install Python dependencies:**
            ```bash
            pip install streamlit langchain langchain-community langchain-chroma chromadb pypdf requests
            ```
            
            **5. Run this app:**
            ```bash
            streamlit run app.py
            ```
            """)


if __name__ == "__main__":
    main()