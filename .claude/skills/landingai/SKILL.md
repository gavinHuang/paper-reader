---
name: landingai
description: AI-powered document extraction toolkit using LandingAI's Advanced Document Extraction (ADE). Parse PDFs and documents into structured markdown, split documents into sub-documents, and extract structured data using AI. When Claude needs to intelligently process, analyze, or extract information from complex documents at scale.
license: Proprietary. Requires LandingAI API key
---

# LandingAI ADE Document Processing Guide

## Overview

This guide covers AI-powered document processing using LandingAI's Advanced Document Extraction (ADE) library. ADE uses advanced AI models to parse documents into structured markdown, split documents, and extract structured data with high accuracy.

## Quick Start

```python
from pathlib import Path
from landingai_ade import LandingAIADE

# Initialize client (requires VISION_AGENT_API_KEY environment variable)
client = LandingAIADE()

# Parse a document
response = client.parse(
    document=Path("document.pdf"),
    model="dpt-2-latest"
)

# Access structured content
print(response.markdown)
print(f"Pages: {len(response.splits)}")
```

## Setup

### Installation

```bash
pip install landingai-ade
```

### API Key Configuration

```bash
# Set API key as environment variable
export VISION_AGENT_API_KEY=<your-api-key>
```

Generate your API key at: https://va.landing.ai/my/settings/api-key

### EU Endpoints

```python
from landingai_ade import LandingAIADE

# Use EU endpoint if your API key is from EU
client = LandingAIADE(environment="eu")
```

## Parse: Convert Documents to Structured Markdown

### Parse Local Files

```python
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Parse a local document
response = client.parse(
    document=Path("/path/to/document.pdf"),
    model="dpt-2-latest"
)

# Access parsed content
print(response.chunks)  # List of parsed chunks
print(response.markdown)  # Complete markdown representation

# Save markdown output
with open("output.md", "w", encoding="utf-8") as f:
    f.write(response.markdown)
```

### Parse Remote URLs

```python
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Parse from URL (http, https, ftp, ftps)
response = client.parse(
    document_url="https://example.com/document.pdf",
    model="dpt-2-latest"
)

# Save markdown
with open("output.md", "w", encoding="utf-8") as f:
    f.write(response.markdown)
```

### Parse with Parameters

```python
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Customize parsing behavior
response = client.parse(
    document=Path("/path/to/document.pdf"),
    model="dpt-2-latest",
    split="page"  # Split by page
)
```

### Parse Large Documents (Async Jobs)

For documents up to 1,000 pages or 1 GB:

```python
import time
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Step 1: Create parse job
job = client.parse_jobs.create(
    document=Path("/path/to/large_document.pdf"),
    model="dpt-2-latest"
)

job_id = job.job_id
print(f"Job {job_id} created.")

# Step 2: Poll for completion
while True:
    response = client.parse_jobs.get(job_id)
    if response.status == "completed":
        print(f"Job {job_id} completed.")
        break
    print(f"Job {job_id}: {response.status} ({response.progress * 100:.0f}% complete)")
    time.sleep(5)

# Step 3: Access parsed data
print(f"Number of chunks: {len(response.data.chunks)}")

# Save markdown
with open("output.md", "w", encoding="utf-8") as f:
    f.write(response.data.markdown)
```

### List Parse Jobs

```python
from landingai_ade import LandingAIADE

client = LandingAIADE()

# List all parse jobs
response = client.parse_jobs.list()
for job in response.jobs:
    print(f"Job {job.job_id}: {job.status}")
```

### Working with Parse Output

#### Access Text Chunks

```python
# Filter text chunks
for chunk in response.chunks:
    if chunk.type == 'text':
        print(f"Chunk {chunk.id}: {chunk.markdown}")
```

#### Filter by Page

```python
# Get chunks from specific page
page_0_chunks = [chunk for chunk in response.chunks if chunk.grounding.page == 0]
```

#### Get Chunk Locations

```python
# Access bounding box coordinates
for chunk in response.chunks:
    box = chunk.grounding.box
    print(f"Chunk at page {chunk.grounding.page}: ({box.left}, {box.top}, {box.right}, {box.bottom})")
```

#### Access Chunk Types

```python
# Get detailed chunk type from grounding
for chunk_id, grounding in response.grounding.items():
    print(f"Chunk {chunk_id} has type: {grounding.type}")
```

## Split: Classify and Separate Documents

### Split from Parse Response

```python
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Parse document
parse_response = client.parse(
    document=Path("/path/to/document.pdf"),
    model="dpt-2-latest"
)

# Define split rules
split_class = [
    {
        "name": "Bank Statement",
        "description": "Document from a bank that summarizes all account activity over a period of time."
    },
    {
        "name": "Pay Stub",
        "description": "Document that details an employee's earnings, deductions, and net pay for a specific pay period.",
        "identifier": "Pay Stub Date"
    }
]

# Split using markdown from parse response
split_response = client.split(
    split_class=split_class,
    markdown=parse_response.markdown,
    model="split-latest"
)

# Access splits
for split in split_response.splits:
    print(f"Classification: {split.classification}")
    print(f"Identifier: {split.identifier}")
    print(f"Pages: {split.pages}")
```

### Split from Markdown Files

```python
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

split_class = [
    {
        "name": "Invoice",
        "description": "A document requesting payment for goods or services.",
        "identifier": "Invoice Number"
    },
    {
        "name": "Receipt",
        "description": "A document acknowledging that payment has been received."
    }
]

# Split from local markdown file
split_response = client.split(
    split_class=split_class,
    markdown=Path("/path/to/output.md"),
    model="split-latest"
)

# Or split from remote markdown
split_response = client.split(
    split_class=split_class,
    markdown_url="https://example.com/document.md",
    model="split-latest"
)

# Access splits
for split in split_response.splits:
    print(f"Classification: {split.classification}")
    print(f"Pages: {split.pages}")
    print(f"Markdown: {split.markdowns[0][:100]}...")
```

### Working with Split Output

#### Filter by Classification

```python
# Get specific document types
invoices = [split for split in split_response.splits if split.classification == "Invoice"]
print(f"Found {len(invoices)} invoices")
```

#### Group by Identifier

```python
from collections import defaultdict

splits_by_id = defaultdict(list)
for split in split_response.splits:
    if split.identifier:
        splits_by_id[split.identifier].append(split)

for identifier, splits in splits_by_id.items():
    print(f"Identifier '{identifier}': {len(splits)} split(s)")
```

## Extract: Get Structured Data

### Extract with Pydantic Models

```python
from pathlib import Path
from landingai_ade import LandingAIADE
from landingai_ade.lib import pydantic_to_json_schema
from pydantic import BaseModel, Field

# Define extraction schema
class PayStubData(BaseModel):
    employee_name: str = Field(description="The employee's full name")
    employee_ssn: str = Field(description="The employee's Social Security Number")
    gross_pay: float = Field(description="The gross pay amount")

client = LandingAIADE()

# Parse document
parse_response = client.parse(
    document=Path("/path/to/pay-stub.pdf"),
    model="dpt-2-latest"
)

# Convert Pydantic model to JSON schema
schema = pydantic_to_json_schema(PayStubData)

# Extract structured data
extract_response = client.extract(
    schema=schema,
    markdown=parse_response.markdown,
    model="extract-latest"
)

# Access extracted data
print(extract_response.extraction)
print(extract_response.extraction_metadata)
```

### Extract with JSON Schema (Inline)

```python
import json
from pathlib import Path
from landingai_ade import LandingAIADE

# Define schema as dictionary
schema_dict = {
    "type": "object",
    "properties": {
        "employee_name": {
            "type": "string",
            "description": "The employee's full name"
        },
        "employee_ssn": {
            "type": "string",
            "description": "The employee's Social Security Number"
        },
        "gross_pay": {
            "type": "number",
            "description": "The gross pay amount"
        }
    },
    "required": ["employee_name", "employee_ssn", "gross_pay"]
}

client = LandingAIADE()

# Parse document
parse_response = client.parse(
    document=Path("/path/to/pay-stub.pdf"),
    model="dpt-2-latest"
)

# Convert to JSON string
schema_json = json.dumps(schema_dict)

# Extract data
extract_response = client.extract(
    schema=schema_json,
    markdown=parse_response.markdown,
    model="extract-latest"
)

print(extract_response.extraction)
```

### Extract from Markdown Files

```python
import json
from pathlib import Path
from landingai_ade import LandingAIADE

schema_dict = {
    "type": "object",
    "properties": {
        "employee_name": {"type": "string", "description": "Full name"},
        "gross_pay": {"type": "number", "description": "Gross pay amount"}
    }
}

client = LandingAIADE()
schema_json = json.dumps(schema_dict)

# Extract from local markdown
extract_response = client.extract(
    schema=schema_json,
    markdown=Path("/path/to/output.md"),
    model="extract-latest"
)

# Or extract from remote markdown
extract_response = client.extract(
    schema=schema_json,
    markdown_url="https://example.com/document.md",
    model="extract-latest"
)

print(extract_response.extraction)
```

### Extract with JSON Schema File

```python
import json
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Parse document
parse_response = client.parse(
    document=Path("/path/to/document.pdf"),
    model="dpt-2-latest"
)

# Load schema from file
with open("schema.json", "r") as f:
    schema_json = f.read()

# Extract data
extract_response = client.extract(
    schema=schema_json,
    markdown=parse_response.markdown,
    model="extract-latest"
)

print(extract_response.extraction)
```

### Extract Nested Subfields

```python
from pathlib import Path
from pydantic import BaseModel, Field
from landingai_ade import LandingAIADE
from landingai_ade.lib import pydantic_to_json_schema

# Define nested models
class PatientDetails(BaseModel):
    patient_name: str = Field(description="Full name of the patient")
    date: str = Field(description="Date the form was filled out")

class EmergencyContact(BaseModel):
    contact_name: str = Field(description="Full name of emergency contact")
    relationship: str = Field(description="Relationship to patient")
    phone: str = Field(description="Primary phone number")

# Main schema with nested fields
class MedicalForm(BaseModel):
    patient_details: PatientDetails = Field(description="Patient information")
    emergency_contact: EmergencyContact = Field(description="Emergency contact info")

client = LandingAIADE()

# Parse document
parse_response = client.parse(
    document=Path("/path/to/medical-form.pdf"),
    model="dpt-2-latest"
)

# Extract with nested schema
schema = pydantic_to_json_schema(MedicalForm)
extract_response = client.extract(
    schema=schema,
    markdown=parse_response.markdown,
    model="extract-latest"
)

print(extract_response.extraction)
```

### Extract Variable-Length Lists

```python
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from landingai_ade import LandingAIADE
from landingai_ade.lib import pydantic_to_json_schema

# Define list item models
class LineItem(BaseModel):
    description: str = Field(description="Item description")
    amount: float = Field(description="Item amount")

class WireInstruction(BaseModel):
    bank_name: str = Field(description="Bank name")
    account_no: str = Field(description="Account number")
    swift_code: str = Field(description="SWIFT code")

# Schema with list fields
class Invoice(BaseModel):
    line_items: List[LineItem] = Field(description="Invoice line items")
    wire_instructions: List[WireInstruction] = Field(description="Wire transfer instructions")

client = LandingAIADE()

# Parse document
parse_response = client.parse(
    document=Path("/path/to/invoice.pdf"),
    model="dpt-2-latest"
)

# Extract lists
schema = pydantic_to_json_schema(Invoice)
extract_response = client.extract(
    schema=schema,
    markdown=parse_response.markdown,
    model="extract-latest"
)

print(extract_response.extraction)
```

### Link Extracted Data to Document Locations

```python
from pathlib import Path
from landingai_ade import LandingAIADE
from landingai_ade.lib import pydantic_to_json_schema
from pydantic import BaseModel, Field

class PayStubData(BaseModel):
    employee_name: str = Field(description="Employee's full name")
    gross_pay: float = Field(description="Gross pay amount")

client = LandingAIADE()

# Parse document
parse_response = client.parse(
    document=Path("/path/to/pay-stub.pdf"),
    model="dpt-2-latest"
)

# Extract data
schema = pydantic_to_json_schema(PayStubData)
extract_response = client.extract(
    schema=schema,
    markdown=parse_response.markdown,
    model="extract-latest"
)

# Link to source location
chunk_id = extract_response.extraction_metadata["employee_name"]["references"][0]
grounding = parse_response.grounding[chunk_id]

print(f"Employee name: {extract_response.extraction['employee_name']}")
print(f"Found on page {grounding.page}")
print(f"Location: ({grounding.box.left:.3f}, {grounding.box.top:.3f})")
print(f"Chunk type: {grounding.type}")
```

## Common Workflows

### Process Directory of Documents

```python
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

data_folder = Path("data/")

for filepath in data_folder.glob("*"):
    if filepath.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']:
        print(f"Processing: {filepath.name}")
        
        response = client.parse(
            document=filepath,
            model="dpt-2-latest"
        )
        
        # Save markdown
        output_md = filepath.stem + ".md"
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(response.markdown)
```

### Async Processing (Multiple Documents)

```python
import asyncio
from pathlib import Path
from landingai_ade import AsyncLandingAIADE

client = AsyncLandingAIADE()

async def process_document(filepath):
    response = await client.parse(
        document=filepath,
        model="dpt-2-latest"
    )
    
    # Save output
    with open(f"{filepath.stem}.md", "w", encoding="utf-8") as f:
        f.write(response.markdown)
    
    return response

async def main():
    files = [Path(f) for f in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]]
    results = await asyncio.gather(*[process_document(f) for f in files])
    print(f"Processed {len(results)} documents")

asyncio.run(main())
```

### Save Parsed Output

```python
import json
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Parse document
response = client.parse(
    document=Path("/path/to/document.pdf"),
    model="dpt-2-latest"
)

# Create output directory
output_dir = Path("ade_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Save JSON
with open(output_dir / "parse_results.json", "w", encoding="utf-8") as f:
    json.dump(response.model_dump(), f, indent=2, default=str)

# Save markdown
with open(output_dir / "output.md", "w", encoding="utf-8") as f:
    f.write(response.markdown)
```

### Visualize Chunks with Bounding Boxes

```python
from pathlib import Path
from landingai_ade import LandingAIADE
from PIL import Image, ImageDraw
import pymupdf

# Define colors for chunk types
COLORS = {
    "chunkText": (40, 167, 69),
    "chunkTable": (0, 123, 255),
    "chunkFigure": (255, 0, 255),
    "chunkForm": (220, 20, 60),
}

def draw_boxes(parse_response, pdf_path):
    pdf = pymupdf.open(pdf_path)
    
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)
        
        for gid, grounding in parse_response.grounding.items():
            if grounding.page != page_num:
                continue
            
            box = grounding.box
            x1 = int(box.left * img.width)
            y1 = int(box.top * img.height)
            x2 = int(box.right * img.width)
            y2 = int(box.bottom * img.height)
            
            color = COLORS.get(grounding.type, (128, 128, 128))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        img.save(f"page_{page_num + 1}_annotated.png")
    
    pdf.close()

client = LandingAIADE()
response = client.parse(
    document=Path("document.pdf"),
    model="dpt-2-latest"
)
draw_boxes(response, Path("document.pdf"))
```

## Quick Reference

| Task | Function | Key Parameters |
|------|----------|---------------|
| Parse local file | `client.parse()` | `document=Path()` |
| Parse remote file | `client.parse()` | `document_url="https://..."` |
| Parse large file | `client.parse_jobs.create()` | `document=Path()` |
| Split document | `client.split()` | `split_class=[...]`, `markdown=` |
| Extract data | `client.extract()` | `schema=`, `markdown=` |
| Convert Pydantic | `pydantic_to_json_schema()` | `Model` |
| Async parse | `AsyncLandingAIADE()` | Use with `asyncio` |

## Response Objects

### ParseResponse
- `chunks`: List of parsed regions
- `markdown`: Complete markdown representation
- `grounding`: Chunk location metadata
- `splits`: Organization by page/section
- `metadata`: Processing information

### SplitResponse
- `splits`: List of sub-documents
  - `classification`: Split type name
  - `identifier`: Unique identifier
  - `pages`: Page numbers
  - `markdowns`: Content per page
- `metadata`: Processing information

### ExtractResponse
- `extraction`: Extracted key-value pairs
- `extraction_metadata`: Chunk references per field
- `metadata`: Processing information

## Models

- **Parse**: `dpt-2-latest` (recommended)
- **Split**: `split-latest`
- **Extract**: `extract-latest`

## Next Steps

- API Reference: https://docs.landing.ai/api-reference/tools/ade-parse
- Chunk Types: https://docs.landing.ai/ade/ade-chunk-types
- JSON Response: https://docs.landing.ai/ade/ade-json-response
- GitHub: https://github.com/landing-ai/ade-python
