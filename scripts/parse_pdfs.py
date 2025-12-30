import os
from pathlib import Path
from llama_parse import LlamaParse
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv('llamaparse-key')
if not api_key:
    raise ValueError("llamaparse-key not found in .env file")

# PDF files to parse (those without corresponding .md files)
pdf_files = [
    "self-evolving-agents/s10462-025-11422-4.pdf",
    "self-evolving-agents/76083b1b-07d6-4099-9b17-e4da08599706.pdf",
    "self-evolving-agents/32960e66-927a-416e-85c8-12f089812d0c.pdf",
    "self-evolving-agents/77246551-482e-4312-99b4-786192a59e5e.pdf",
    "self-evolving-agents/96e34750-4a80-462f-953e-628a10a816c6.pdf",
    "self-evolving-agents/f47afba4-2787-453b-b3bb-d804f68dbd32.pdf",
    "self-evolving-agents/e0c663d0-cee2-44c3-a902-754aa3186279.pdf",
    "self-evolving-agents/de640784-9f94-420e-ab02-c7b8809d09e2.pdf",
    "self-evolving-agents/73f5769f-333c-4ba3-b751-533aeb157422.pdf",
    "self-evolving-agents/2512.16922v1.pdf",
]

# Initialize parser
parser = LlamaParse(
    api_key=api_key,
    result_type="markdown",
    extract_charts=True,
    auto_mode=True,
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
)

failed_files = []
successful_files = []

for pdf_file in pdf_files:
    pdf_path = Path(pdf_file)

    # Check if PDF exists
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_file}")
        failed_files.append(pdf_file)
        continue

    # Check if markdown already exists
    md_path = pdf_path.with_suffix('.md')
    if md_path.exists():
        logger.info(f"Skipping {pdf_path.name} - markdown already exists")
        continue

    try:
        logger.info(f"Processing: {pdf_path.name}")

        extra_info = {"file_name": str(pdf_path)}

        # Parse the PDF
        with open(pdf_path, "rb") as f:
            documents = parser.load_data(f, extra_info=extra_info)

        # Save markdown output
        with open(md_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.text)
                f.write("\n\n")

        logger.info(f"✓ Successfully processed: {pdf_path.name} -> {md_path.name}")
        successful_files.append(pdf_path.name)

    except Exception as e:
        logger.error(f"✗ Failed to process {pdf_path.name}: {str(e)}")
        failed_files.append(pdf_path.name)

# Summary
logger.info(f"\n{'='*60}")
logger.info(f"Processing complete!")
logger.info(f"Successfully processed: {len(successful_files)} files")
logger.info(f"Failed: {len(failed_files)} files")

if successful_files:
    logger.info(f"\nSuccessful files:")
    for file in successful_files:
        logger.info(f"  ✓ {file}")

if failed_files:
    logger.info(f"\nFailed files:")
    for file in failed_files:
        logger.info(f"  ✗ {file}")
