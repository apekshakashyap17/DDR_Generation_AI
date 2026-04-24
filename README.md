# DDR Generation AI

An AI-powered pipeline that automatically generates Detailed Diagnostic Reports (DDR) from multi-page inspection and thermal imaging PDFs — turning raw unstructured documents into structured, client-ready reports.

## What it does

Upload inspection and thermal imaging PDFs and the system extracts, analyzes, and synthesizes the information into a formatted diagnostic report — no manual reading or summarizing required.

## How it works

1. PDF pages are converted into images (`page-to-image-converter.py`)
2. Images are sent to a multimodal vision model (GPT-4.1-mini via OpenRouter) for information extraction
3. Extracted data is processed with conflict detection and severity classification logic
4. A final structured report is generated and saved to the output folder (`final_extract.py`)

## Tech Stack

- **LLM:** GPT-4.1-mini via OpenRouter
- **PDF Processing:** PyMuPDF
- **Language:** Python

## Setup
pip install -r requirements.txt


Create a `.env` file and add your OpenRouter API key:

Then run:
python final_extract.py


## Sample Data

Sample inspection PDFs are included in the `sample data` folder for testing.

## Note

OpenRouter permits only 20 free API calls — don't run the project unnecessarily unless you have a paid subscription. The final report is generated as a text file which you can convert to a document or PDF.
