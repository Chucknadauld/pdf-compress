#!/usr/bin/env python3
"""
PDF Summarizer — Reads PDFs from input_pdfs/ and saves summaries to output_summaries/

Usage:
    1. Set your API key:  export ANTHROPIC_API_KEY="your-key-here"
    2. Drop PDF files into the input_pdfs/ folder
    3. Run:  python3 summarize_pdfs.py
"""

import os
import sys
from pathlib import Path

import anthropic
import pymupdf


INPUT_DIR = Path(__file__).parent / "input_pdfs"
OUTPUT_DIR = Path(__file__).parent / "output_summaries"

MAX_CHARS = 150_000  # trim very large documents to stay within token limits


def extract_text(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def summarize(text: str, filename: str, client: anthropic.Anthropic) -> str:
    """Send extracted text to Claude and return a summary."""
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[...truncated due to length...]"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Below is the full text extracted from a PDF document named \"{filename}\". "
                    "Please provide a thorough, well-organized summary that captures all the key points, "
                    "main arguments, important details, and conclusions. Use clear headings and bullet points "
                    "where appropriate.\n\n"
                    "--- DOCUMENT TEXT ---\n\n"
                    f"{text}"
                ),
            }
        ],
    )
    return message.content[0].text


def main():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("  export ANTHROPIC_API_KEY=\"your-key-here\"")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {INPUT_DIR}/")
        print("Drop your PDFs there and run this script again.")
        sys.exit(0)

    print(f"Found {len(pdfs)} PDF(s) to summarize.\n")

    for pdf_path in pdfs:
        name = pdf_path.stem
        output_path = OUTPUT_DIR / f"{name}_summary.txt"

        print(f"Processing: {pdf_path.name}")

        # Extract text
        text = extract_text(pdf_path)
        if not text.strip():
            print(f"  Skipped — no extractable text (may be a scanned/image PDF).\n")
            continue

        print(f"  Extracted {len(text):,} characters from {pdf_path.name}")

        # Summarize
        print(f"  Summarizing with Claude...")
        summary = summarize(text, pdf_path.name, client)

        # Save
        output_path.write_text(summary, encoding="utf-8")
        print(f"  Saved → {output_path}\n")

    print("Done!")


if __name__ == "__main__":
    main()
