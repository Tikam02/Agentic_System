import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# LangChain imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
from pydantic import BaseModel, Field
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration for model switching"""
    
    def __init__(self):
        self.llm_model = "phi3:mini"
        self.embedding_model = "mxbai-embed-large"
        self.ollama_base_url = "http://localhost:11434"
        self.chroma_persist_dir = "./chroma_db"

# Pydantic models for structured output
class EmployeeData(BaseModel):
    full_name: Optional[str] = Field(description="Employee's full name")
    email_address: Optional[str] = Field(description="Email address")
    phone_number: Optional[str] = Field(description="Phone number")
    company_name: Optional[str] = Field(description="Company name")
    role_designation: Optional[str] = Field(description="Job role or designation")
    country: Optional[str] = Field(description="Country")
    industry: Optional[str] = Field(description="Industry sector")

class WaterData(BaseModel):
    annual_water_usage: Optional[float] = Field(description="Annual water usage in litres")
    wastewater_generated: Optional[float] = Field(description="Wastewater generated in litres")
    water_recycled_percent: Optional[float] = Field(description="Percentage of water recycled")
    water_scarcity_index: Optional[float] = Field(description="Regional water scarcity index")
    water_law_compliance: Optional[str] = Field(description="Compliance with water laws (Yes/No)")
    rainwater_harvesting: Optional[str] = Field(description="Rainwater harvesting (Yes/No)")

class ChemicalData(BaseModel):
    chemicals_used: Optional[str] = Field(description="Types of chemicals used")
    chemical_storage_volume: Optional[float] = Field(description="Chemical storage volume in litres")
    hazardous_classification: Optional[str] = Field(description="Hazardous chemical classification")
    chemical_waste_generated: Optional[float] = Field(description="Chemical waste generated in kg/year")
    chemical_safety_compliance: Optional[str] = Field(description="Chemical safety compliance (Yes/No)")
    fume_outlets_count: Optional[int] = Field(description="Number of fume outlets")
    air_emissions: Optional[float] = Field(description="Air emissions in tons/year")
    particulate_emissions: Optional[float] = Field(description="Particulate matter emissions in tons/year")
    emission_control_systems: Optional[str] = Field(description="Emission control systems")
    air_quality_compliance: Optional[str] = Field(description="Air quality compliance (Yes/No)")

class LLMManager:
    """Manages LLM and embedding models with easy switching"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = OllamaLLM(
            model=config.llm_model,
            base_url=config.ollama_base_url
        )
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url
        )
    
    def switch_llm_model(self, model_name: str):
        """Switch to different LLM model"""
        self.config.llm_model = model_name
        self.llm = OllamaLLM(
            model=model_name,
            base_url=self.config.ollama_base_url
        )
        logger.info(f"Switched LLM model to: {model_name}")

class DocumentProcessor:
    """Handles PDF loading and text processing"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_and_split_pdf(self, pdf_path: Path) -> List[Document]:
        """Load PDF and split into chunks"""
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_file": pdf_path.name,
                    "file_type": "pdf"
                })
            
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(chunks)} chunks from {pdf_path.name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return []

class VectorStoreManager:
    """Manages ChromaDB vector store"""
    
    def __init__(self, config: Config, llm_manager: LLMManager):
        self.config = config
        self.embeddings = llm_manager.embeddings
        self.vectorstore = None
        
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create or update vector store with documents"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.config.chroma_persist_dir
        )
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=k)
        return []

class EmployeeDataAgent:
    """Extracts employee data using LangChain"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager.llm
        self.parser = PydanticOutputParser(pydantic_object=EmployeeData)
        
        self.prompt = PromptTemplate(
            template="""Extract employee information from the following document content.
            Focus on personal and professional details.
            
            Document content:
            {document_content}
            
            {format_instructions}
            
            Extract only the information that is clearly present in the document.
            """,
            input_variables=["document_content"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # Use new pipe syntax instead of deprecated LLMChain
        self.chain = self.prompt | self.llm | self.parser
    
    def extract(self, document_content: str) -> Dict[str, Any]:
        """Extract employee data from document content"""
        try:
            result = self.chain.invoke({"document_content": document_content[:2000]})
            return result.dict() if hasattr(result, 'dict') else {}
        except Exception as e:
            logger.error(f"Error extracting employee data: {e}")
            return {}

class ResourceDataAgent:
    """Extracts resource/environmental data using LangChain"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager.llm
        
    def extract_water_data(self, document_content: str) -> Dict[str, Any]:
        """Extract water-related data"""
        parser = PydanticOutputParser(pydantic_object=WaterData)
        
        prompt = PromptTemplate(
            template="""Extract water usage and management data from the following document.
            Focus on water consumption, recycling, and compliance metrics.
            
            Document content:
            {document_content}
            
            {format_instructions}
            """,
            input_variables=["document_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({"document_content": document_content[:2000]})
            return result.dict() if hasattr(result, 'dict') else {}
        except Exception as e:
            logger.error(f"Error extracting water data: {e}")
            return {}
    
    def extract_chemical_data(self, document_content: str) -> Dict[str, Any]:
        """Extract chemical-related data"""
        parser = PydanticOutputParser(pydantic_object=ChemicalData)
        
        prompt = PromptTemplate(
            template="""Extract chemical usage, storage, and safety data from the following document.
            Focus on chemical types, volumes, waste, and compliance metrics.
            
            Document content:
            {document_content}
            
            {format_instructions}
            """,
            input_variables=["document_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({"document_content": document_content[:2000]})
            return result.dict() if hasattr(result, 'dict') else {}
        except Exception as e:
            logger.error(f"Error extracting chemical data: {e}")
            return {}

class ExcelWriter:
    """Handles Excel file operations"""
    
    def __init__(self):
        self.column_mapping = {
            "full_name": "Full Name",
            "email_address": "Email Address",
            "phone_number": "Phone Number",
            "company_name": "Company Name",
            "role_designation": "Role / Designation",
            "country": "Country",
            "industry": "Industry",
            "annual_water_usage": "Annual Water Usage (litres)",
            "wastewater_generated": "Wastewater Generated (litres)",
            "water_recycled_percent": "Water Recycled (%)",
            "chemicals_used": "Chemicals Used",
            "chemical_storage_volume": "Chemical Storage Volume (L)",
            "chemical_waste_generated": "Chemical Waste Generated (kg/year)",
            "air_emissions": "Air Emissions (tons/year)"
        }
    
    def write_data(self, records: List[Dict], excel_path: Path):
        """Write extracted data to Excel file"""
        try:
            # Read existing Excel structure
            df = pd.read_excel(excel_path)
            
            # Process each record
            for record in records:
                row_data = {}
                for key, value in record.items():
                    if key in self.column_mapping and value is not None:
                        row_data[self.column_mapping[key]] = value
                
                # Add row to dataframe
                if row_data:
                    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
            
            # Write back to Excel
            df.to_excel(excel_path, index=False)
            logger.info(f"Successfully wrote {len(records)} records to Excel")
            
        except Exception as e:
            logger.error(f"Error writing to Excel: {e}")

class ESGOrchestrator:
    """Main orchestrator coordinating all agents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.doc_processor = DocumentProcessor(self.llm_manager)
        self.vector_manager = VectorStoreManager(config, self.llm_manager)
        self.employee_agent = EmployeeDataAgent(self.llm_manager)
        self.resource_agent = ResourceDataAgent(self.llm_manager)
        self.excel_writer = ExcelWriter()
        
    def process_esg_pipeline(self, project_dir: Path) -> List[Dict]:
        """Main processing pipeline"""
        
        logger.info("Starting ESG processing pipeline...")
        
        # Define paths
        employee_dir = project_dir / "employee"
        resources_dir = project_dir / "resources"
        excel_path = project_dir / "forms" / "esg_form.xls"
        
        # Debug: Check directory structure
        logger.info(f"Project directory: {project_dir}")
        logger.info(f"Employee dir exists: {employee_dir.exists()}")
        logger.info(f"Resources dir exists: {resources_dir.exists()}")
        logger.info(f"Excel file exists: {excel_path.exists()}")
        
        all_documents = []
        extracted_records = []
        
        # Process employee PDFs
        employee_data_list = []
        if employee_dir.exists():
            pdf_files = list(employee_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} employee PDF files: {[f.name for f in pdf_files]}")
            
            for pdf_file in pdf_files:
                logger.info(f"Processing employee PDF: {pdf_file.name}")
                
                # Load and chunk PDF
                chunks = self.doc_processor.load_and_split_pdf(pdf_file)
                if not chunks:
                    logger.warning(f"No chunks extracted from {pdf_file.name}")
                    continue
                    
                all_documents.extend(chunks)
                
                # Extract employee data from full document content
                full_content = " ".join([chunk.page_content for chunk in chunks])
                logger.info(f"Content length for {pdf_file.name}: {len(full_content)} characters")
                
                employee_data = self.employee_agent.extract(full_content)
                logger.info(f"Extracted employee data: {employee_data}")
                
                if employee_data:
                    employee_data_list.append(employee_data)
        else:
            logger.warning(f"Employee directory not found: {employee_dir}")
        
        # Process resource PDFs
        resource_data = {}
        if resources_dir.exists():
            pdf_files = list(resources_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} resource PDF files: {[f.name for f in pdf_files]}")
            
            for pdf_file in pdf_files:
                logger.info(f"Processing resource PDF: {pdf_file.name}")
                
                # Load and chunk PDF
                chunks = self.doc_processor.load_and_split_pdf(pdf_file)
                if not chunks:
                    logger.warning(f"No chunks extracted from {pdf_file.name}")
                    continue
                    
                all_documents.extend(chunks)
                
                # Extract resource data
                full_content = " ".join([chunk.page_content for chunk in chunks])
                logger.info(f"Content length for {pdf_file.name}: {len(full_content)} characters")
                
                if "water" in pdf_file.name.lower():
                    water_data = self.resource_agent.extract_water_data(full_content)
                    logger.info(f"Extracted water data: {water_data}")
                    resource_data.update(water_data)
                elif "chemical" in pdf_file.name.lower():
                    chemical_data = self.resource_agent.extract_chemical_data(full_content)
                    logger.info(f"Extracted chemical data: {chemical_data}")
                    resource_data.update(chemical_data)
        else:
            logger.warning(f"Resources directory not found: {resources_dir}")
        
        # Create vector store for future queries
        if all_documents:
            self.vector_manager.create_vectorstore(all_documents)
            logger.info(f"Created vector store with {len(all_documents)} documents")
        
        # Combine employee and resource data
        logger.info(f"Employee data records: {len(employee_data_list)}")
        logger.info(f"Resource data fields: {list(resource_data.keys())}")
        
        for employee_data in employee_data_list:
            combined_record = {**employee_data, **resource_data}
            extracted_records.append(combined_record)
        
        # Write to Excel
        if extracted_records and excel_path.exists():
            self.excel_writer.write_data(extracted_records, excel_path)
        elif not excel_path.exists():
            logger.error(f"Excel file not found: {excel_path}")
        
        logger.info(f"Pipeline completed. Processed {len(extracted_records)} records.")
        return extracted_records

def main():
    """Main execution function"""
    
    # Initialize configuration
    config = Config()
    
    # Initialize orchestrator
    orchestrator = ESGOrchestrator(config)
    
    # Process ESG reports (current directory since we're running from ESG/)
    project_dir = Path(".")
    results = orchestrator.process_esg_pipeline(project_dir)
    
    print(f"Successfully processed {len(results)} records")
    print(f"Using models: LLM={config.llm_model}, Embeddings={config.embedding_model}")
    
    # Example of switching models during runtime
    # orchestrator.llm_manager.switch_llm_model("mistral:latest")

if __name__ == "__main__":
    main()