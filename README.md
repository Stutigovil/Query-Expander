# 💬 Conversational AI Pipeline

A production-ready end-to-end conversational AI system built with Python, featuring:
- **Query Expansion**: Rewrite ambiguous user messages into full questions using Claude API
- **Topic Classification**: 2-level hierarchical topic classification using fine-tuned DistilBERT
- **Conversation Management**: SQLite-based persistent conversation history
- **Streamlit UI**: Beautiful, responsive web interface

## 🎯 System Architecture

### 1. Query Expansion
- Uses Google Gemini API (`gemini-1.5-flash`)
- Resolves pronouns and references from conversation history
- Handles general/chitchat queries with `[GENERAL]` tag
- Sliding window context: last 20 messages (10 dialogue exchanges)

### 2. Topic Classification
- **Level 1**: 6 main verticals (Politics, Sports, Technology, Health & Medicine, Finance, General)
- **Level 2**: ~60 subdomains (e.g., Indian Politics, Cricket, Artificial Intelligence, etc.)
- Fine-tuned DistilBERT models
- Confidence threshold: 0.75 for fallback to General category
- Taxonomy validation ensures L2 belongs to predicted L1

### 3. Database
- SQLite (`conversations.db`) — zero config, single file
- Stores: message content, role, timestamp, expansion, and classification metadata
- Per-session persistence using UUID session IDs

### 4. UI
- Streamlit web application
- Real-time chat interface
- Inline metadata display:
  - Original query
  - Expanded query
  - Topic tag (Level1 > Level2)
  - Classification confidence

## 📋 Quick Start

### 1. Clone/Setup Project
```bash
cd /path/to/Superkalam
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy template and add your API key
cp .env.example .env

# Edit .env and add your Google Gemini API key
nano .env  # or use your preferred editor
```

Example `.env`:
```
GOOGLE_API_KEY=AIzaSy...
```

Get your API key from: https://makersuite.google.com/app/apikey

### 5. Train Classifiers
```bash
python train_classifier.py
```

This will:
- Read `dataset.csv`
- Train Level 1 classifier (6 classes)
- Train Level 2 classifier (~60 classes)
- Save models to `model_level1/` and `model_level2/`
- Save label encoders to `label_encoders/`
- Print accuracy and F1 scores

**Note**: Training takes ~5-10 minutes on CPU, faster on GPU.

### 6. Run Application
```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

## 📁 Project Structure

```
project/
├── app.py                      # Streamlit UI (main entry point)
├── conversation_manager.py     # SQLite + session state management
├── query_expander.py           # Claude API expansion logic
├── topic_classifier.py         # DistilBERT inference
├── response_aggregator.py      # Combines pipeline outputs
├── train_classifier.py         # Model training script
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
├── .env                        # (Create this) Actual API key
├── README.md                   # This file
├── dataset.csv                 # Training data
├── conversations.db            # SQLite database (created automatically)
├── model_level1/               # Level 1 classifier (created by train script)
├── model_level2/               # Level 2 classifier (created by train script)
└── label_encoders/             # LabelEncoders & taxonomy (created by train script)
    ├── le_level1.pkl
    ├── le_level2.pkl
    └── taxonomy.json
```
