# Smart Report Extractor

Smart Report Extractor is a robust, multi-layered data extraction pipeline built on FastAPI. It parses unstructured PDF documents specifically **Invoices**, **Bank Statements**, and **Resumes**—and accurately extracts structured intelligence using a consensus-based confidence scoring system.

## 🚀 Features
- **Heuristic Type Detection**: Automatically identifies the layout type of a document without AI.
- **Offline Reliability**: Works fully offline out of the box using deterministic tools.
- **Consensus Merger**: Runs multiple offline extraction layers concurrently (Positional, Generative NER, Regex) and mathematically calculates the most likely true-value field.
- **Intelligent Fallback**: Conditionally fires an asynchronous Anthropic LLM abstraction attempt *only* if the offline system detects low overall document confidence.
- **Sleek Interface**: Ships with a single-file Vanilla JS interface offering drag-and-drop support and a built-in results parser.
- **Versatile Exports**: Streams parsed data back to the client as either formatted JSON or flattened CSV.

## 🧠 Architecture Setup

The pipeline executes through strict sequential stages:
1. `pdfplumber` attempts to rip the crude text layer and structural tables.
2. The core heuristic flags the `ReportType`.
3. Three offline layers process the text independently.
4. The `.merger` runs a consensus check on the field dictionaries. If `overall_confidence < 0.85`, it optionally activates the LLM API.
5. The `summarizer` builds a concluding context string.
6. The `exporter` transforms the dict into JSON/CSV streams.

## ⚙️ Installation

**1. Create a Virtual Environment**
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Install the foundational NLP Model**
The SpaCy Named Entity Recognition layer requires a specific English language model to operate. You must download it manually globally into your environment:
```bash
python -m spacy download en_core_web_md
```

**4. Add Environment Variables**
Create a `.env` file in the root if you want the LLM extraction fallback and API Summarizer to function.
```ini
ANTHROPIC_API_KEY=sk-ant-api03-...
```

## 💻 Running the App
Run the server locally with Live-Reloading enabled:
```bash
uvicorn main:app --reload
```
Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the frontend interface.

## 🧪 Testing
The system utilizes `pytest` to confidently assert all core components work. You can execute them by simply running:
```bash
pytest
```
