# RAG-Augmented Crash Narrative Interpreter

A RAG-augmented LLM system that reads Louisiana officer crash narratives and outputs structured JSON predictions for key crash report fields, including Manner of Crash (MOC), First Harmful Event (FHE), and Vehicle Maneuver codes.

This project was developed for **CSC 7644/7644E – Large Language Models** at Louisiana State University, Spring 2026.

## Background

Crash analysts at the Center for Analytics & Research in Transportation Safety (CARTS) at LSU manually read officer crash narratives and assign 20+ structured codes by cross-referencing the 393-page Louisiana eCrash User Guide. This process is slow (10–15 minutes per report), subjective, and inconsistent across different analysts. This project automates that workflow using Retrieval-Augmented Generation (RAG) combined with structured prompting.

## Features

- Reads State Case Numbers from an Excel file and locates matching narrative Word files
- Retrieves relevant coding rules from the eCrash User Guide using FAISS vector search
- Uses GPT-4o with chain-of-thought prompting to interpret narratives and assign codes
- Validates output with Pydantic schema enforcement
- Streamlit web interface with single-narrative and batch processing modes
- Auto-fills predictions back into the source Excel file

## Architecture

Excel (case numbers) -> Word file lookup -> Narrative extraction
|
v
eCrash User Guide -> Chunking -> Embedding -> FAISS index
|
v
RAG retrieval
|
v
GPT-4o with structured prompt + CoT
|
v
Pydantic validation -> JSON output
|
v
Excel auto-fill


## Tech Stack

- **LLM:** OpenAI GPT-4o (ChatGPT API)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** FAISS (faiss-cpu)
- **Document Processing:** PyMuPDF (PDF), python-docx (Word)
- **Output Validation:** Pydantic
- **Web Interface:** Streamlit
- **Language:** Python 3.12

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key
- Louisiana eCrash User Guide PDF
- Crash narratives saved as Word files (named by State Case Number)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/HamidrezaAmin/CrashInterpreter.git
cd CrashInterpreter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your OpenAI API key:
OPENAI_API_KEY=your_api_key_here


4. Place the eCrash User Guide PDF in the project root as `LA eCrash User Guide.pdf`.

5. Place crash narrative Word files in a `narratives/` folder, named by State Case Number (e.g., `2025020863.docx`).

## Usage

### 1. Build the RAG Index

Open `CrashInterpreter.ipynb` in Jupyter or VS Code and run cells in order. This will:
- Extract text from the eCrash User Guide
- Chunk and embed the content
- Build a FAISS vector index

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

A browser window will open with two tabs:

- **Single Narrative:** Enter a State Case Number, the system finds the matching Word file, runs interpretation, and shows the JSON output. A "Confirm" button saves the prediction to the Excel file.
- **Batch Processing:** Reads all case numbers from `State_number.xlsx`, processes each one, and saves results back to the Excel file.

## File Structure

LLM_Project/
├── app.py                      # Streamlit web interface
├── CrashInterpreter.ipynb      # Notebook with RAG pipeline build steps
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Files excluded from Git
├── .env                        # OpenAI API key (not tracked)
├── LA eCrash User Guide.pdf    # RAG corpus (not tracked)
├── ecrash_faiss.index          # FAISS index (generated)
├── ecrash_metadata.json        # Chunk metadata (generated)
├── narratives/                 # Crash narrative Word files (not tracked)
└── State_number.xlsx           # Input Excel with case numbers


## Code Design

The system follows an agentic pipeline pattern:

1. **Input layer:** Reads State Case Numbers from Excel
2. **Document layer:** Locates and reads the matching Word narrative
3. **Retrieval layer:** Queries the FAISS index for relevant coding rules
4. **Reasoning layer:** Sends the narrative + retrieved rules to GPT-4o with a structured prompt that enforces chain-of-thought reasoning
5. **Validation layer:** Pydantic checks that the JSON output conforms to the expected schema
6. **Output layer:** Writes predictions back to the Excel file

The system prompt enforces critical coding rules including the parked vs. stopped distinction, point-of-contact rules for Manner of Crash, and vehicle-in-transport determination.

## Evaluation Results

Tested on 14 manually labeled crash narratives:

| Field | Accuracy |
|-------|----------|
| Manner of Crash (MOC) | 78.6% |
| First Harmful Event (FHE) | 92.9% |
| V1 Maneuver | 78.6% |
| V2 Maneuver | 100.0% |
| V3 Maneuver | 100.0% |
| **Overall** | **87.9%** |

## Limitations

- Restricted dataset access: crash narratives require credentialed access to the Louisiana eCrash system
- The model occasionally misinterprets ambiguous narratives where multiple coding interpretations are valid
- The system is designed as a decision-support tool and should not replace human review

## Future Work

- Expand evaluation to a larger labeled dataset
- Add few-shot examples to improve MOC and V1 maneuver accuracy
- Implement self-verification prompting loop
- Consider fine-tuning on labeled crash narratives if RAG accuracy plateaus

## Author

Hamidreza Aminorroayaei  
Graduate Assistant, Center for Analytics & Research in Transportation Safety (CARTS)  
Louisiana State University

## License

This project was developed as coursework for CSC 7644 at LSU.