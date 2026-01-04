#!/usr/bin/env python3
"""
Simple script to parse PDF using LlamaParse
"""
import os
import sys
from pathlib import Path
from llama_parse import LlamaParse


def load_api_key():
    """Load API key from environment or .env file"""
    # Check environment variable first
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if api_key:
        return api_key

    # Try to load from .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("llamaparse-key="):
                    return line.split("=", 1)[1].strip()

    raise ValueError("LLAMA_CLOUD_API_KEY not found. Please set it in .env file or environment variable.")


def parse_pdf(pdf_path: str, output_path: str = None):
    """
    Parse a PDF file using LlamaParse and save as markdown

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path for the output markdown file
    """
    # Load API key
    api_key = load_api_key()

    # Initialize parser with auto mode for best quality
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        extract_charts=True,
        auto_mode=True,
        auto_mode_trigger_on_image_in_page=True,
        auto_mode_trigger_on_table_in_page=True,
    )

    # Convert to Path object
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"Error: File not found: {pdf_path}")
        return

    print(f"Processing: {pdf_file.name}")

    # Set up extra info
    extra_info = {"file_name": str(pdf_file)}

    try:
        # Parse the document
        with open(pdf_file, "rb") as f:
            documents = parser.load_data(f, extra_info=extra_info)

        # Determine output path
        if output_path is None:
            output_path = pdf_file.with_suffix('.md')
        else:
            output_path = Path(output_path)

        # Save markdown output
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.text)
                f.write("\n\n")

        print(f"Successfully converted to: {output_path}")
        print(f"Total pages processed: {len(documents)}")

    except Exception as e:
        print(f"Error processing {pdf_file.name}: {str(e)}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_with_llamaparse.py <pdf_file> [output_file]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    parse_pdf(pdf_path, output_path)
