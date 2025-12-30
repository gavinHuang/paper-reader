import os
from pathlib import Path
from landingai_ade import LandingAIADE
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv('landingai-key')

if not api_key:
    raise ValueError("API key 'landingai-key' not found in .env file")

# Set the API key as environment variable for LandingAI
os.environ['VISION_AGENT_API_KEY'] = api_key

# Initialize client
client = LandingAIADE()

# Define the folder containing PDFs
pdf_folder = Path("self-evolving-agents")

# Get all PDF files
pdf_files = list(pdf_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files to process")

# Process each PDF
for i, pdf_path in enumerate(pdf_files, 1):
    print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

    try:
        # Parse the PDF
        response = client.parse(
            document=pdf_path,
            model="dpt-2-latest"
        )

        # Create output markdown file with same name
        output_path = pdf_path.with_suffix('.md')

        # Save markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.markdown)

        print(f"  [OK] Saved to: {output_path.name}")
        print(f"  Pages: {len(response.splits)}")
        print(f"  Chunks: {len(response.chunks)}")

    except Exception as e:
        print(f"  [ERROR] Error processing {pdf_path.name}: {str(e)}")
        continue

print(f"\n{'='*60}")
print("Processing complete!")
