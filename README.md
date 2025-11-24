# 🎙️ Nirmaan AI Communication Coach

## 📋 Project Overview
This AI-powered tool analyzes and scores students' spoken communication skills from their self-introduction transcripts. It was built as part of the **Nirmaan AI Intern Case Study**, blending rule-based heuristics with modern NLP to deliver rubric-based scoring out of 100.

- **Rule-based logic** (regex, threshold checks)
- **NLP with spaCy** for NER and tokenization
- **Semantic similarity** via Sentence Transformers
- **Sentiment analysis** using VADER

## 🛠️ Tech Stack & Architecture
- **Frontend:** Streamlit (Python)
- **NLP & Logic:**
  - `spacy` – NER for names/locations  
  - `sentence-transformers` – semantic matching for hobbies/goals  
  - `language-tool-python` – grammar checking (requires Java)  
  - `vaderSentiment` – sentiment/enthusiasm scoring  
  - `pandas` – data framing for tables

## 📊 Scoring Logic ("Product Brain")
Scores are weighted across five rubric criteria:

1. **Content & Structure (40%)**
   - Salutation classification
   - Keyword coverage via NER/regex/semantic hits
   - Flow check (greeting → intro → closing)

2. **Speech Rate (10%)**
   - Formula: `(word_count / duration_sec) * 60`
   - Ideal 111–140 WPM earns full marks

3. **Language & Grammar (20%)**
   - Grammar via LanguageTool (dialect-aware filtering)
   - Vocabulary via Type-Token Ratio (TTR)

4. **Clarity (15%)**
   - Filler word detection (`um`, `uh`, `like`, etc.)
   - Target < 3% filler rate

5. **Engagement (15%)**
   - VADER positivity and compound scores
   - Rewards enthusiastic tone (>0.3 positivity or >0.8 compound)

## 🚀 How to Run Locally
### Prerequisites
- Python 3.8+
- Java runtime (LanguageTool dependency)

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/nirmaan-ai-scorer.git
cd nirmaan-ai-scorer

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt

streamlit run app.py
```

## ☁️ Deployment (Streamlit Cloud)
Create a `packages.txt` in the repo root containing:
```
default-jdk
```
Then deploy via Streamlit Community Cloud (New App → choose repo → Deploy).

## 📂 Project Structure
```
├── app.py           # Streamlit UI + validation
├── scorer.py        # Scoring engine and rubric logic
├── requirements.txt # Python dependencies
├── packages.txt     # System dependencies (Java)
└── README.md        # Documentation
```

