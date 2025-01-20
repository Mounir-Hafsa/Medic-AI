

# Medic-AI (AI-driven Pharmacy (Over The Counter) assistant for French Drugs)

This repository contains a Streamlit application for an AI-driven Pharmacy assistant. The application uses:
- **Streamlit** for the front-end UI
- **OpenAI** (via the `openai` Python package) for generating text completions
- **ChromaDB** (`chromadb` library) for vector-based document storage and retrieval
- **text-embedding-ada-003** or a similar model for text embeddings
- Patient information, drug search, and drug recommendation workflows

> **Important**: This project is a demonstration tool only and **must not** be used to provide real medical advice.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Setup](#setup)
5. [Running the App](#running-the-app)
6. [Usage](#usage)
7. [Notes on Data & Security](#notes-on-data--security)
8. [Troubleshooting](#troubleshooting)
9. [Disclaimer](#disclaimer)
10. [License](#license)

---

## Features

- **Patient Info Form**: Collect user details such as age, gender, weight, allergies, current medications, etc.
- **OTC Drug Recommendations**: Suggest appropriate over-the-counter medications based on symptoms and patient details.
- **Drug Information Retrieval**: Query a local ChromaDB for relevant drug data (including side effects, dosages, contraindications).
- **Interactive Chat**: Users can reset and start new queries within the same session.
- **Auto-Generated Search Queries**: The system automatically constructs advanced Boolean queries in French to retrieve matching drug records from the local database.
- **OpenAI-Driven Summaries & Recommendations**: Summarize medication info, provide disclaimers, and highlight relevant sections.

---

## Prerequisites

1. **Python 3.8 or above** (recommended Python 3.9+).
2. [**pip**](https://pip.pypa.io/en/stable/) or another Python package manager (e.g., **Conda** or **Poetry**).
3. [**Git**](https://git-scm.com/downloads) (if you want to clone this repo directly).
4. An **OpenAI API key**:
   - Sign up at [https://platform.openai.com/signup](https://platform.openai.com/signup)
   - Create or retrieve your API key from the OpenAI Dashboard.

---

## Installation

1. **Clone** the repository (or download and unzip):
    ```bash
    git clone https://github.com/Mounir-Hafsa/Medic-AI)
    cd Medic-AI
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
   Make sure your `requirements.txt` includes: openai streamlit chromadb (and any other libraries used by the app).

---

## Setup

### 1. OpenAI API Key

Because the app references `st.secrets["OpenAI_key"]`, you have two main ways to store your API key:

#### **Method A: `secrets.toml` file**

1. Create a folder named `.streamlit` in the root of your repository (if it doesn’t already exist).
2. Inside `.streamlit`, create a file named `secrets.toml`.
3. Add your API key:
 ```toml
 [general]
 OpenAI_key = "YOUR_API_KEY"
 ```
4. Streamlit will automatically load the key under `st.secrets["OpenAI_key"]`.

#### **Method B: Environment Variables**

If you prefer environment variables, you can set them in your shell or a `.env` file, then reference them in the code accordingly (but that would require modifying the current code to use `os.getenv("OPENAI_API_KEY")` or similar).

### 2. ChromaDB

The application references a **PersistentClient** for ChromaDB, which will create or use a local SQLite database file (e.g., `database` directory) to store embeddings. Make sure you have the necessary permissions to read/write to this location.

If you run into issues, confirm that the path you specified in:
```python
chroma_client = chromadb.PersistentClient(path="database")
```
is accessible and writable.

---

## Running the App
1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```
2. **Open your browser** and go to `http://localhost:8501` (or another port if specified).

3. Usage:
   - Fill in the patient information (age, gender, allergies, etc.) and click **Confirm**.
   - Select the type of query you want to perform:
     - Looking for the right OTC Drug Recommendations: Enter symptoms, severity/duration, previous treatments. The AI will suggest potential OTC options (for demonstration purposes only).
     - Looking for Certain Drug Info: Enter a drug name or ID, and the AI will attempt to retrieve relevant structured info from ChromaDB and transform it into a standardized JSON format. 
   - Follow the prompts and review the results:
     - The app displays results, disclaimers, and final summary.

---

## Disclaimer
This application does not provide medical advice. It is a demonstration and is not intended to replace professional healthcare advice. Always consult a healthcare professional for medical diagnoses or treatments.

## License
This project is open-source. You may use, modify, or distribute it in accordance with the LICENSE file in this repository, or under the terms agreed upon by your organization.
