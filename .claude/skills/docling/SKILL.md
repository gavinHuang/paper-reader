---
name: docling
description: Open-source document processing toolkit for AI applications. Parse PDFs, DOCX, PPTX, XLSX, HTML, images, and audio into structured formats. Advanced PDF understanding with layout analysis, table extraction, OCR support, and Visual Language Models. Export to Markdown, HTML, JSON. Integrates with LangChain, LlamaIndex, Haystack. When Claude needs to parse complex documents locally, extract structured data, or process documents in air-gapped/sensitive environments.
license: MIT License (individual models retain their original licenses)
---

# Docling Document Processing Guide

## Overview

Docling is an open-source Python library that prepares documents for generative AI applications. It parses diverse document formats and converts them into unified, machine-readable representations optimized for AI workflows. Unlike cloud-based parsing services, Docling runs entirely locally, making it ideal for sensitive data and air-gapped environments.

## Quick Start

```python
from docling.document_converter import DocumentConverter

# Initialize converter
converter = DocumentConverter()

# Convert a document (supports URLs and local files)
source = "https://arxiv.org/pdf/2408.09869"
result = converter.convert(source)

# Export to markdown
markdown_text = result.document.export_to_markdown()
print(markdown_text)
```

## Setup

### Installation

```bash
pip install docling
```

**Platform Support:** macOS, Linux, and Windows on x86_64 and arm64 architectures

**Recommendation:** Create and activate a virtual Python environment before installing to avoid dependency conflicts.

### No API Key Required

Docling runs entirely locally on your machine. No cloud API keys or accounts are needed, making it ideal for:
- Sensitive or confidential documents
- Air-gapped environments
- Cost-sensitive applications
- Privacy-focused workflows

## Supported File Types

Docling supports a wide range of document and media formats:

### Documents
- **PDF** (.pdf) - Advanced layout analysis and table extraction
- **Word** (.docx)
- **PowerPoint** (.pptx)
- **Excel** (.xlsx)
- **HTML** (.html)

### Images
- **PNG** (.png)
- **JPEG** (.jpg, .jpeg)
- **TIFF** (.tiff)

### Audio
- **WAV** (.wav)
- **MP3** (.mp3)
- **WebVTT** (.vtt) - Subtitle/caption files

## Basic Document Conversion

### Convert Local PDF

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Convert local file
result = converter.convert("./document.pdf")

# Access the parsed document
doc = result.document

# Export to markdown
markdown = doc.export_to_markdown()
print(markdown)
```

### Convert from URL

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Convert remote document
source = "https://arxiv.org/pdf/2408.09869"
result = converter.convert(source)

# Save to markdown file
with open("output.md", "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())
```

### Convert Multiple Formats

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Docling automatically handles different formats
documents = [
    "./report.pdf",
    "./presentation.pptx",
    "./data.xlsx",
    "./notes.docx"
]

for doc_path in documents:
    result = converter.convert(doc_path)

    # Save with same name but .md extension
    output_name = doc_path.rsplit(".", 1)[0] + ".md"
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())

    print(f"Converted: {doc_path} -> {output_name}")
```

## Export Formats

Docling supports multiple export formats for different use cases:

### Markdown Export

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Export to markdown
markdown = result.document.export_to_markdown()

# Save to file
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

### HTML Export

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Export to HTML
html = result.document.export_to_html()

# Save to file
with open("output.html", "w", encoding="utf-8") as f:
    f.write(html)
```

### JSON Export (Lossless)

```python
from docling.document_converter import DocumentConverter
import json

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Export to JSON (preserves all document structure)
doc_dict = result.document.export_to_dict()

# Save to file
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(doc_dict, f, indent=2, ensure_ascii=False)
```

### DocTags Export

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Export to DocTags format
doctags = result.document.export_to_doctags()
print(doctags)
```

## Advanced PDF Understanding

Docling provides advanced PDF analysis capabilities:

### Layout Analysis

Docling automatically performs:
- **Page layout detection** - Identifies columns, sections, and layout structure
- **Reading order identification** - Determines the correct reading sequence
- **Table structure recognition** - Extracts tables with proper structure
- **Code detection** - Identifies and preserves code blocks
- **Formula extraction** - Extracts mathematical formulas
- **Image categorization** - Classifies images (figures, diagrams, photos)

All of this happens automatically with the default converter:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("complex_document.pdf")

# Layout information is preserved in the export
markdown = result.document.export_to_markdown()

# Tables are converted to markdown tables
# Code blocks are preserved
# Images are referenced
# Reading order is maintained
```

### Extract Tables

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document_with_tables.pdf")

# Tables are automatically extracted and included in markdown
markdown = result.document.export_to_markdown()

# Tables appear as markdown tables:
# | Column 1 | Column 2 | Column 3 |
# |----------|----------|----------|
# | Data 1   | Data 2   | Data 3   |

with open("tables_output.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

### Extract Figures and Images

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document_with_images.pdf")

# Access the document object
doc = result.document

# Export figures (implementation depends on your configuration)
# Figures are referenced in the markdown output
markdown = doc.export_to_markdown()

# Save markdown with figure references
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

## OCR for Scanned Documents

Docling provides extensive OCR support for processing scanned PDFs and images without native text layers.

### Using Tesseract OCR

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions

# Configure Tesseract OCR
pipeline_options = PdfPipelineOptions()
pipeline_options.ocr_options = TesseractOcrOptions()

converter = DocumentConverter(
    pipeline_options=pipeline_options
)

# Convert scanned PDF
result = converter.convert("scanned_document.pdf")
markdown = result.document.export_to_markdown()

with open("ocr_output.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

### Using RapidOCR

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

# Configure RapidOCR
pipeline_options = PdfPipelineOptions()
pipeline_options.ocr_options = RapidOcrOptions()

converter = DocumentConverter(
    pipeline_options=pipeline_options
)

result = converter.convert("scanned_document.pdf")
markdown = result.document.export_to_markdown()
```

### Using EasyOCR

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

# Configure EasyOCR with language support
pipeline_options = PdfPipelineOptions()
pipeline_options.ocr_options = EasyOcrOptions(
    lang=["en", "es"]  # English and Spanish
)

converter = DocumentConverter(
    pipeline_options=pipeline_options
)

result = converter.convert("multilingual_scanned.pdf")
markdown = result.document.export_to_markdown()
```

## Visual Language Model (VLM) Pipeline

Docling supports Visual Language Models for enhanced document understanding:

### Using GraniteDocling VLM

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import VlmPipelineOptions

# Configure VLM pipeline
pipeline_options = VlmPipelineOptions(
    vlm_model="granite_docling"
)

converter = DocumentConverter(
    pipeline_options=pipeline_options
)

# Convert with VLM for enhanced understanding
result = converter.convert("complex_layout.pdf")
markdown = result.document.export_to_markdown()
```

### CLI with VLM

```bash
# Use VLM pipeline from command line
docling --pipeline vlm --vlm-model granite_docling document.pdf
```

**Note:** GraniteDocling (258M parameters) provides enhanced document understanding and can be accelerated with MLX on Apple Silicon.

## Audio Processing

Docling can process audio files and extract text using Automatic Speech Recognition:

### Convert Audio to Text

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Process audio file (WAV, MP3, etc.)
result = converter.convert("audio_recording.wav")

# Export transcription
transcript = result.document.export_to_markdown()

with open("transcript.md", "w", encoding="utf-8") as f:
    f.write(transcript)
```

### Process WebVTT Captions

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Process subtitle/caption files
result = converter.convert("captions.vtt")

# Export as text
text = result.document.export_to_markdown()
```

## Batch Processing

### Process Multiple Documents

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Process all PDFs in a directory
input_dir = Path("./documents")
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

for pdf_file in input_dir.glob("*.pdf"):
    print(f"Processing: {pdf_file.name}")

    try:
        result = converter.convert(str(pdf_file))

        # Save markdown output
        output_file = output_dir / f"{pdf_file.stem}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.document.export_to_markdown())

        print(f"Saved: {output_file}")

    except Exception as e:
        print(f"Error processing {pdf_file.name}: {e}")
```

### Batch Processing with Error Handling

```python
from pathlib import Path
from docling.document_converter import DocumentConverter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

converter = DocumentConverter()

input_dir = Path("./documents")
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

successful = []
failed = []

# Process all supported document types
patterns = ["*.pdf", "*.docx", "*.pptx", "*.xlsx"]

for pattern in patterns:
    for doc_file in input_dir.glob(pattern):
        try:
            logger.info(f"Converting: {doc_file.name}")
            result = converter.convert(str(doc_file))

            # Save output
            output_file = output_dir / f"{doc_file.stem}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.document.export_to_markdown())

            successful.append(doc_file.name)
            logger.info(f"Success: {doc_file.name}")

        except Exception as e:
            failed.append((doc_file.name, str(e)))
            logger.error(f"Failed: {doc_file.name} - {e}")

# Summary
logger.info(f"Processed: {len(successful)} successful, {len(failed)} failed")
if failed:
    logger.info("Failed files:")
    for filename, error in failed:
        logger.info(f"  - {filename}: {error}")
```

## Structured Data Extraction (Beta)

Docling supports structured information extraction for targeted data recovery:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Convert document
result = converter.convert("invoice.pdf")

# Access structured document data
doc = result.document

# The document object contains structured information
# including metadata, sections, tables, etc.

# Export to JSON for programmatic access
doc_dict = doc.export_to_dict()

# Access specific elements
# (exact API depends on document structure)
```

## Chunking for RAG Applications

### Basic Chunking

```python
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HybridChunker

# Convert document
converter = DocumentConverter()
result = converter.convert("document.pdf")

# Create chunker
chunker = HybridChunker()

# Chunk the document
chunks = list(chunker.chunk(result.document))

# Each chunk contains a segment of the document
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.text[:100]}...")
```

### Chunking with Custom Configuration

```python
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HybridChunker

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Configure chunker
chunker = HybridChunker(
    max_tokens=512,  # Maximum tokens per chunk
    overlap_tokens=50  # Overlap between chunks
)

chunks = list(chunker.chunk(result.document))

# Use chunks for RAG pipeline
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Metadata: {chunk.meta}")
```

## Integration with RAG Frameworks

### LangChain Integration

```python
from docling.document_converter import DocumentConverter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Convert document with Docling
converter = DocumentConverter()
result = converter.convert("document.pdf")

# Get markdown text
text = result.document.export_to_markdown()

# Create LangChain text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split into chunks
chunks = text_splitter.split_text(text)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(api_key="your-api-key")
vectorstore = FAISS.from_texts(chunks, embeddings)

# Query the vector store
query = "What are the main findings?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)
```

### LlamaIndex Integration

```python
from docling.document_converter import DocumentConverter
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

# Convert with Docling
converter = DocumentConverter()
result = converter.convert("document.pdf")

# Create LlamaIndex Document
doc_text = result.document.export_to_markdown()
llama_doc = Document(text=doc_text)

# Create index
embed_model = OpenAIEmbedding(api_key="your-api-key")
index = VectorStoreIndex.from_documents(
    [llama_doc],
    embed_model=embed_model
)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
print(response)
```

### Haystack Integration

```python
from docling.document_converter import DocumentConverter
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever

# Convert with Docling
converter = DocumentConverter()
result = converter.convert("document.pdf")

# Create Haystack document
text = result.document.export_to_markdown()
haystack_doc = Document(content=text)

# Add to document store
document_store = InMemoryDocumentStore()
document_store.write_documents([haystack_doc])

# Use with retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
)

# Query
results = retriever.retrieve(query="What are the key points?")
```

## CLI Usage

### Basic Conversion

```bash
# Convert local file
docling document.pdf

# Convert from URL
docling https://arxiv.org/pdf/2206.01062

# Specify output directory
docling document.pdf --output ./output/

# Convert multiple files
docling document1.pdf document2.docx document3.pptx
```

### Using VLM Pipeline

```bash
# Use Visual Language Model for enhanced understanding
docling --pipeline vlm --vlm-model granite_docling document.pdf

# With output directory
docling --pipeline vlm --vlm-model granite_docling document.pdf --output ./output/
```

### Export Format Options

```bash
# Export as markdown (default)
docling document.pdf

# Export as HTML
docling document.pdf --export-format html

# Export as JSON
docling document.pdf --export-format json
```

## Advanced Features

### PII Detection and Obfuscation

Docling can detect and obfuscate Personally Identifiable Information:

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure PII detection
pipeline_options = PdfPipelineOptions()
pipeline_options.enable_pii_detection = True

converter = DocumentConverter(
    pipeline_options=pipeline_options
)

result = converter.convert("document_with_pii.pdf")

# PII will be detected and can be obfuscated
markdown = result.document.export_to_markdown()
```

### Visual Grounding

Visual grounding capabilities enable linking text to specific regions in the document:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Access visual grounding information
doc = result.document

# Visual grounding data is available in the document structure
# (exact API depends on configuration)
```

### Custom Pipeline Configuration

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Create custom pipeline options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = True
pipeline_options.do_ocr = True

converter = DocumentConverter(
    pipeline_options=pipeline_options
)

result = converter.convert("document.pdf")
```

## Vector Store Integrations

### Milvus Integration

```python
from docling.document_converter import DocumentConverter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

# Convert document
converter = DocumentConverter()
result = converter.convert("document.pdf")
text = result.document.export_to_markdown()

# Split into chunks
chunks = text.split("\n\n")

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Insert into Milvus
# (collection setup code omitted for brevity)
```

### Weaviate Integration

```python
from docling.document_converter import DocumentConverter
import weaviate

# Convert document
converter = DocumentConverter()
result = converter.convert("document.pdf")
text = result.document.export_to_markdown()

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Create chunks and insert
chunks = text.split("\n\n")

for i, chunk in enumerate(chunks):
    data_object = {
        "content": chunk,
        "chunk_id": i
    }
    client.data_object.create(data_object, "Document")
```

### Qdrant Integration

```python
from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Convert document
converter = DocumentConverter()
result = converter.convert("document.pdf")
text = result.document.export_to_markdown()

# Chunk and embed
chunks = text.split("\n\n")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Insert into Qdrant
client = QdrantClient(host="localhost", port=6333)

points = [
    PointStruct(
        id=i,
        vector=embedding.tolist(),
        payload={"text": chunk}
    )
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

client.upsert(collection_name="documents", points=points)
```

## Common Workflows

### Research Paper Processing

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Process academic papers
papers_dir = Path("./papers")
output_dir = Path("./processed_papers")
output_dir.mkdir(exist_ok=True)

for paper in papers_dir.glob("*.pdf"):
    print(f"Processing: {paper.name}")

    result = converter.convert(str(paper))

    # Export to markdown with preserved structure
    markdown = result.document.export_to_markdown()

    output_file = output_dir / f"{paper.stem}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Saved: {output_file}")
```

### Invoice Processing

```python
from docling.document_converter import DocumentConverter
import json

converter = DocumentConverter()

# Process invoices
result = converter.convert("invoice.pdf")

# Export structured data
invoice_data = result.document.export_to_dict()

# Save as JSON
with open("invoice_data.json", "w", encoding="utf-8") as f:
    json.dump(invoice_data, f, indent=2)

# Also save readable markdown
markdown = result.document.export_to_markdown()
with open("invoice.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

### Multi-Format Document Library

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Process entire document library
library_dir = Path("./document_library")
output_dir = Path("./processed_library")
output_dir.mkdir(exist_ok=True)

# All supported formats
extensions = [".pdf", ".docx", ".pptx", ".xlsx", ".html"]

for ext in extensions:
    for doc in library_dir.glob(f"**/*{ext}"):
        print(f"Processing: {doc.relative_to(library_dir)}")

        try:
            result = converter.convert(str(doc))

            # Preserve directory structure
            relative_path = doc.relative_to(library_dir)
            output_path = output_dir / relative_path.with_suffix(".md")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.document.export_to_markdown())

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error: {e}")
```

## Comparison: Docling vs LlamaParse

| Feature | Docling | LlamaParse |
|---------|---------|------------|
| **Licensing** | MIT (Open Source) | Proprietary |
| **Cost** | Free | 1,000 pages/day free, paid tiers |
| **Execution** | Local | Cloud API |
| **API Key** | Not required | Required |
| **Privacy** | Full (local processing) | Data sent to cloud |
| **Air-gapped** | Yes | No |
| **PDF Layout** | Advanced | Advanced |
| **Table Extraction** | Yes | Yes |
| **Chart Extraction** | Yes (with VLM) | Yes |
| **OCR Support** | Yes (multiple engines) | Yes |
| **VLM Support** | Yes (GraniteDocling) | No |
| **Audio Processing** | Yes | No |
| **Custom Prompts** | No | Yes |
| **Auto Mode** | N/A | Yes |
| **Export Formats** | Markdown, HTML, JSON, DocTags | Markdown, Text |

**Use Docling when:**
- Processing sensitive or confidential documents
- Working in air-gapped environments
- Avoiding cloud dependencies
- No budget for parsing costs
- Need audio processing
- Want complete control over the processing pipeline

**Use LlamaParse when:**
- Want GenAI-native parsing with custom prompts
- Need auto-optimization for cost/quality balance
- Processing large volumes with cloud scalability
- Want translation during parsing
- Prefer managed service over local setup

## Best Practices

1. **Virtual Environments:** Always use a virtual environment to avoid dependency conflicts
2. **Local Processing:** Take advantage of local execution for sensitive documents
3. **Batch Processing:** Process multiple documents efficiently with error handling
4. **Format Selection:** Choose the right export format for your use case:
   - Markdown: Best for RAG and LLM applications
   - JSON: Best for programmatic access and structured data
   - HTML: Best for web display
5. **OCR Configuration:** Select the appropriate OCR engine for your needs:
   - Tesseract: General purpose, widely supported
   - RapidOCR: Fast processing
   - EasyOCR: Strong multilingual support
6. **VLM for Complex Layouts:** Use VLM pipeline for documents with complex visual layouts
7. **Chunking Strategy:** Use appropriate chunking for your RAG application
8. **Error Handling:** Always wrap conversions in try-except blocks for production use

## Troubleshooting

### Installation Issues

```bash
# Use virtual environment
python -m venv docling_env
source docling_env/bin/activate  # On Windows: docling_env\Scripts\activate
pip install docling
```

### Memory Issues (Large Documents)

```python
# Process in batches or page ranges if supported
# Monitor memory usage for very large documents
import gc

converter = DocumentConverter()
result = converter.convert("large_document.pdf")
markdown = result.document.export_to_markdown()

# Clear memory
del result
gc.collect()
```

### OCR Not Working

```bash
# Install OCR dependencies
pip install docling[ocr]

# For Tesseract, ensure it's installed on your system:
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from GitHub
```

### VLM Model Issues

```python
# Ensure VLM dependencies are installed
# Some models may require additional setup
# Check documentation for specific model requirements
```

## Resources

- **Official Documentation:** https://docling-project.github.io/docling/
- **GitHub Repository:** https://github.com/docling-project/docling
- **Technical Report:** https://arxiv.org/abs/2408.09869
- **Code Examples:** https://docling-project.github.io/docling/examples/
- **Discord Community:** Available via documentation
- **MCP Server:** For agent integration capabilities

## Next Steps

1. Install Docling with `pip install docling`
2. Try the quick start example with a sample PDF
3. Explore different export formats
4. Test OCR with scanned documents
5. Integrate with your preferred RAG framework (LangChain, LlamaIndex, Haystack)
6. Set up batch processing for your document workflow
7. Explore VLM capabilities for complex documents
