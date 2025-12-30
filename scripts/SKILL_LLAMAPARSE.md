---
name: llamaparse
description: GenAI-native PDF and document parsing using LlamaParse. Extract structured markdown from PDFs, parse tables and charts, translate documents, and customize output with natural language instructions. Integrates with LlamaIndex for building LLM applications. When Claude needs to parse complex PDFs, extract data from charts/graphs, or convert unstructured documents into clean structured data.
license: Proprietary. Requires LlamaCloud API key
---

# LlamaParse Document Processing Guide

## Overview

LlamaParse is a GenAI-native parsing platform for converting complex documents into clean, structured data optimized for LLM applications. It uses advanced AI models to parse PDFs, PowerPoint, Word docs, Excel, HTML, images and more into markdown with high accuracy, including accurate table and chart extraction.

## Quick Start

```python
from llama_parse import LlamaParse

# Initialize parser (requires LLAMA_CLOUD_API_KEY environment variable)
parser = LlamaParse(
    result_type="markdown",  # "markdown" or "text"
)

# Parse a document
file_name = "./document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Access parsed content
for doc in documents:
    print(doc.text)
```

## Setup

### Installation

```bash
pip install llama-index
pip install llama-parse
```

**Recommendation:** Create and activate a virtual Python environment before installing to avoid dependency conflicts.

### API Key Configuration

1. Create a free account at: https://cloud.llamaindex.ai/login
2. Navigate to **API Keys** from the left sidebar
3. Click **Generate New Key**
4. Copy and save your key securely (you won't be able to view it again)

```bash
# Set API key as environment variable
export LLAMA_CLOUD_API_KEY='llx-...'
```

For Windows PowerShell:
```powershell
$env:LLAMA_CLOUD_API_KEY='llx-...'
```

## Parse: Convert Documents to Structured Markdown

### Basic Parsing

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    # api_key="llx-...",  # Optional if environment variable is set
    result_type="markdown",  # "markdown" or "text"
)

file_name = "./document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Save markdown output
with open("output.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
```

### Parse with Advanced Features

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",
    extract_charts=True,           # Extract data from charts/graphs
    auto_mode=True,                # Optimize parsing cost/quality balance
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
)

file_name = "./document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Save output
with open("output.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
```

### Supported File Types

LlamaParse supports parsing:
- **PDF** (.pdf)
- **PowerPoint** (.pptx)
- **Word** (.docx)
- **Excel** (.xlsx)
- **HTML** (.html)
- **Images** (.jpg, .jpeg, .png)
- **Audio files** (various formats)
- And more...

## Advanced Features

### Extract Charts and Graphs

By default, LlamaParse skips charts to save on costs and processing time. Enable chart extraction for data-rich documents:

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",
    extract_charts=True,           # Enable chart data extraction
    auto_mode=True,                # Premium mode for charts/tables
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
)

file_name = "./report_with_charts.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Charts will be converted to markdown tables
with open("output.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
```

**Auto Mode:** Intelligently determines when to use premium parsing modes (for charts, tables, complex layouts) to optimize cost while maintaining quality. Only invokes premium modes when genuinely needed.

### Translate Documents

Parse and translate documents in a single operation:

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",
    user_prompt="If the input is not in English, translate the output into English."
)

file_name = "./spanish_document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Output will be in English
with open("output_translated.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
```

### Custom Prompts

Use natural language instructions to customize parsing behavior:

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",
    user_prompt="Extract all financial data and organize by quarter. Include only numerical tables."
)

file_name = "./financial_report.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)
```

**Prompt Examples:**
- "Extract only the executive summary section"
- "Convert all tables to JSON format"
- "Focus on extracting dates, names, and amounts"
- "Ignore headers and footers"
- "Summarize each section before extracting details"

### Target Specific Pages

Parse only selected pages (pages are 0-indexed):

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",
    target_pages="0,10,12,22-33"  # Pages 1, 11, 13, and 23-34
)

file_name = "./large_document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)
```

### Ignore Headers and Footers

Exclude content from the top and bottom of pages:

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",
    bbox_top=0.1,      # Ignore top 10% of page
    bbox_bottom=0.05   # Ignore bottom 5% of page
)

file_name = "./document_with_headers.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)
```

## Integration with Vector Databases

### Store Parsed Data in Elasticsearch

```python
from llama_parse import LlamaParse
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext

# Parse document
parser = LlamaParse(result_type="markdown")

file_name = "./document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Create Elasticsearch vector store
es_store = ElasticsearchStore(
    index_name="llama-parse-docs",
    es_cloud_id="your-cloud-id",
    es_api_key="your-api-key",
)

# Parse into nodes
node_parser = SimpleNodeParser()
nodes = node_parser.get_nodes_from_documents(documents)

# Create index with embeddings
storage_context = StorageContext.from_defaults(vector_store=es_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=OpenAIEmbedding(api_key="your-openai-key"),
)
```

### Store in Astra DB

```python
from llama_parse import LlamaParse
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext

# Parse document
parser = LlamaParse(result_type="markdown")

file_name = "./document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Create Astra DB vector store
astra_db_store = AstraDBVectorStore(
    token="your-astra-token",
    api_endpoint="your-api-endpoint",
    collection_name="llama_parse_collection",
    embedding_dimension=1536,
)

# Parse into nodes and create index
node_parser = SimpleNodeParser()
nodes = node_parser.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults(vector_store=astra_db_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=OpenAIEmbedding(api_key="your-openai-key"),
)
```

## Structured Extraction with Schemas

### Extract Invoices

```python
from llama_parse import LlamaParse
from pydantic import BaseModel, Field
from typing import List

class LineItem(BaseModel):
    description: str = Field(description="Item description")
    quantity: int = Field(description="Quantity")
    unit_price: float = Field(description="Price per unit")
    total: float = Field(description="Total amount")

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    date: str = Field(description="Invoice date")
    vendor_name: str = Field(description="Vendor name")
    line_items: List[LineItem] = Field(description="List of invoice items")
    total_amount: float = Field(description="Total invoice amount")

# Define schema in prompt
parser = LlamaParse(
    result_type="markdown",
    user_prompt=f"Extract invoice data following this schema: {Invoice.schema_json()}"
)

file_name = "./invoice.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)
```

### Extract Resumes

```python
from llama_parse import LlamaParse
from pydantic import BaseModel, Field
from typing import List

class Education(BaseModel):
    degree: str = Field(description="Degree name")
    institution: str = Field(description="School/university name")
    graduation_year: str = Field(description="Year of graduation")

class Experience(BaseModel):
    job_title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    duration: str = Field(description="Employment duration")
    responsibilities: List[str] = Field(description="Key responsibilities")

class Resume(BaseModel):
    name: str = Field(description="Candidate name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    education: List[Education] = Field(description="Educational background")
    experience: List[Experience] = Field(description="Work experience")
    skills: List[str] = Field(description="List of skills")

parser = LlamaParse(
    result_type="markdown",
    user_prompt=f"Extract resume data following this schema: {Resume.schema_json()}"
)

file_name = "./resume.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)
```

## Common Workflows

### Process Directory of PDFs

```python
from pathlib import Path
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",
    extract_charts=True,
    auto_mode=True,
)

data_folder = Path("data/")

for filepath in data_folder.glob("*.pdf"):
    print(f"Processing: {filepath.name}")
    
    extra_info = {"file_name": str(filepath)}
    
    with open(filepath, "rb") as f:
        documents = parser.load_data(f, extra_info=extra_info)
    
    # Save markdown output
    output_md = filepath.stem + ".md"
    with open(output_md, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.text)
    
    print(f"Saved: {output_md}")
```

### Batch Processing with Error Handling

```python
from pathlib import Path
from llama_parse import LlamaParse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = LlamaParse(
    result_type="markdown",
    extract_charts=True,
    auto_mode=True,
)

data_folder = Path("data/")
output_folder = Path("output/")
output_folder.mkdir(exist_ok=True)

failed_files = []

for filepath in data_folder.glob("**/*.pdf"):
    try:
        logger.info(f"Processing: {filepath}")
        
        extra_info = {"file_name": str(filepath)}
        
        with open(filepath, "rb") as f:
            documents = parser.load_data(f, extra_info=extra_info)
        
        # Save output
        output_path = output_folder / f"{filepath.stem}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.text)
        
        logger.info(f"Successfully processed: {filepath.name}")
        
    except Exception as e:
        logger.error(f"Failed to process {filepath.name}: {str(e)}")
        failed_files.append(filepath.name)

# Summary
logger.info(f"Processing complete. Failed files: {len(failed_files)}")
if failed_files:
    logger.info(f"Failed files: {', '.join(failed_files)}")
```

### Async Processing (Multiple Documents)

```python
import asyncio
from pathlib import Path
from llama_parse import LlamaParse

async def parse_document(parser, filepath):
    """Parse a single document asynchronously"""
    try:
        extra_info = {"file_name": str(filepath)}
        
        with open(filepath, "rb") as f:
            # Note: LlamaParse doesn't have async methods in the basic API
            # This example shows the pattern for when async is available
            documents = parser.load_data(f, extra_info=extra_info)
        
        # Save output
        output_path = f"{filepath.stem}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.text)
        
        return f"✓ {filepath.name}"
    
    except Exception as e:
        return f"✗ {filepath.name}: {str(e)}"

async def main():
    parser = LlamaParse(
        result_type="markdown",
        extract_charts=True,
        auto_mode=True,
    )
    
    files = [Path(f) for f in Path("data/").glob("*.pdf")]
    
    # Process in parallel batches
    batch_size = 5
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        results = await asyncio.gather(
            *[parse_document(parser, f) for f in batch]
        )
        for result in results:
            print(result)

# Run async processing
asyncio.run(main())
```

### Extract Tables from PDFs

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",
    auto_mode=True,
    auto_mode_trigger_on_table_in_page=True,
    user_prompt="Focus on extracting all tables. Preserve table structure and formatting."
)

file_name = "./document_with_tables.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Tables will be in markdown format
with open("tables_output.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
```

### Extract and Save as JSON

```python
import json
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="text",  # Use text for cleaner JSON conversion
)

file_name = "./document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Convert to JSON structure
output_data = {
    "source": file_name,
    "pages": len(documents),
    "content": [
        {
            "page": idx + 1,
            "text": doc.text,
            "metadata": doc.metadata
        }
        for idx, doc in enumerate(documents)
    ]
}

# Save as JSON
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)
```

## Parsing Modes

LlamaParse offers different parsing modes to balance cost, speed, and quality:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Default** | Fast, cost-effective | Simple documents without complex charts |
| **Auto Mode** | Intelligent switching | Mixed documents, optimize cost/quality |
| **Premium** | Maximum accuracy | Complex layouts, critical data extraction |

### Configure Parsing Modes

```python
from llama_parse import LlamaParse

# Cost-effective: Default mode
parser_fast = LlamaParse(
    result_type="markdown",
)

# Balanced: Auto mode (recommended)
parser_auto = LlamaParse(
    result_type="markdown",
    extract_charts=True,
    auto_mode=True,
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
)

# High quality: Always use premium features
parser_premium = LlamaParse(
    result_type="markdown",
    extract_charts=True,
    # Auto mode disabled - always use premium parsing
)
```

## Integration with LlamaIndex

### Build a Simple RAG Application

```python
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

# Parse documents
parser = LlamaParse(
    result_type="markdown",
    extract_charts=True,
    auto_mode=True,
)

file_name = "./knowledge_base.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Create vector index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(api_key="your-openai-key"),
)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What are the key findings?")
print(response)
```

### Create Document Agent

```python
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# Parse documents
parser = LlamaParse(result_type="markdown")

file_name = "./document.pdf"
extra_info = {"file_name": file_name}

with open(file_name, "rb") as f:
    documents = parser.load_data(f, extra_info=extra_info)

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine tool
query_engine = index.as_query_engine()

# Create agent
llm = OpenAI(model="gpt-4", api_key="your-openai-key")
agent = ReActAgent.from_tools(
    [query_engine],
    llm=llm,
    verbose=True
)

# Use agent
response = agent.chat("Summarize the main points from the document")
print(response)
```

## Quick Reference

| Task | Key Parameters | Example |
|------|----------------|---------|
| Basic parsing | `result_type="markdown"` | Simple document conversion |
| Extract charts | `extract_charts=True` | Documents with graphs/visualizations |
| Auto mode | `auto_mode=True` | Balance cost and quality |
| Translation | `user_prompt="Translate..."` | Multi-language documents |
| Specific pages | `target_pages="0,5-10"` | Large documents, partial extraction |
| Ignore headers | `bbox_top=0.1, bbox_bottom=0.05` | Documents with headers/footers |
| Custom prompts | `user_prompt="..."` | Specialized extraction needs |

## Result Types

### Markdown Output
- Preserves document structure
- Tables in markdown format
- Headers and hierarchy
- Best for downstream processing

### Text Output
- Plain text extraction
- No structural formatting
- Simpler, faster processing
- Best for simple text analysis

## Free Tier

LlamaParse offers a generous free tier:
- **1,000 pages per day** free
- No credit card required
- Full feature access

## Best Practices

1. **Use Auto Mode:** Balance cost and quality automatically
2. **Target Specific Pages:** Process only what you need to reduce costs
3. **Custom Prompts:** Guide extraction with natural language instructions
4. **Batch Processing:** Process multiple documents efficiently
5. **Error Handling:** Always wrap parsing in try-except blocks
6. **Save Intermediate Results:** Store parsed markdown for reuse
7. **Test Small First:** Validate on sample pages before full document
8. **Virtual Environments:** Isolate dependencies to avoid conflicts

## Troubleshooting

### API Key Issues
```python
# Set environment variable
import os
os.environ["LLAMA_CLOUD_API_KEY"] = "your-key"

# Or pass directly
parser = LlamaParse(api_key="your-key", result_type="markdown")
```

### File Path Issues
```python
# Use absolute paths
from pathlib import Path
file_path = Path("/absolute/path/to/document.pdf").resolve()

# Or ensure correct relative path
import os
file_path = os.path.abspath("./document.pdf")
```

### Memory Issues (Large Files)
```python
# Process page by page
parser = LlamaParse(
    result_type="markdown",
    target_pages="0-9"  # Process in chunks
)
```

## Resources

- **Documentation:** https://docs.cloud.llamaindex.ai/llamaparse
- **Examples Repository:** https://github.com/run-llama/llama_cloud_services/tree/main/examples
- **Blog Post:** https://www.llamaindex.ai/blog/pdf-parsing-llamaparse
- **LlamaCloud:** https://cloud.llamaindex.ai
- **Parsing Modes:** https://docs.cloud.llamaindex.ai/llamaparse/parsing/parsing_modes
- **Supported File Types:** https://docs.cloud.llamaindex.ai/llamaparse/features/supported_document_types
- **Invoice Schema:** https://docs.cloud.llamaindex.ai/llamaparse/schemas/invoice
- **Resume Schema:** https://docs.cloud.llamaindex.ai/llamaparse/schemas/resume

## Next Steps

1. Create a free LlamaCloud account
2. Generate your API key
3. Install llama-parse and llama-index
4. Try parsing your first document
5. Explore examples in the GitHub repository
6. Integrate with your LlamaIndex applications
