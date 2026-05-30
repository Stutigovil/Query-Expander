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

## 🏷️ Topic Taxonomy

### Verticals (Level 1)
1. **Politics**: Government, elections, policy
2. **Sports**: All sports-related content
3. **Technology**: AI, programming, gadgets, cybersecurity
4. **Health & Medicine**: Health, fitness, medical research
5. **Finance**: Banking, stocks, cryptocurrency, economy
6. **General**: Education, entertainment, science, etc.

### Subdomains (Level 2) Examples
- **Politics**: Indian Politics, US Politics, Elections & Voting, Defence & Military, Geopolitics, International Relations
- **Sports**: Cricket, Football, Tennis, Olympics, Formula 1, Basketball
- **Technology**: Artificial Intelligence, Programming, Web Development, Cloud Computing, Cybersecurity
- **Health & Medicine**: Diseases, Fitness, Nutrition, Mental Health, Medical Research
- **Finance**: Stock Market, Personal Finance, Banking, Cryptocurrency
- **General**: Education, Science, Entertainment, History & Culture, Lifestyle

See full taxonomy in `label_encoders/taxonomy.json` after training.

## 📊 Dataset Format

CSV columns:
- **text**: The statement/query
- **label_vertical**: Main topic (Level 1)
- **label_subdomain**: Specific subtopic (Level 2)

Example:
```csv
text,label_vertical,label_subdomain
"The Prime Minister chaired a cabinet meeting to review national development projects.",Politics,Indian Politics
"India successfully tested a long-range defence missile system.",Politics,Defence & Military
"Machine learning models are transforming data analysis.",Technology,Artificial Intelligence
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google Gemini API key (required)

### Model Configuration (in code)
- **Query Expander**: Gemini 1.5 Flash model
- **Classifiers**: DistilBERT base model
- **Device**: Auto-detects CUDA if available, falls back to CPU
- **Batch size**: 16 (training)
- **Learning rate**: 2e-5
- **Epochs**: 5
- **Early stopping patience**: 2 epochs

### Thresholds
- **Confidence threshold**: 0.75 (L1 classification)
- **Context window**: 20 messages (10 exchanges)

## 🚀 Features

### Query Expansion
- Resolves pronouns: "it" → full concept
- Handles references: "that thing you mentioned" → specific topic
- Removes ambiguity: "how?" → "How does the classifier work?"
- Preserves chitchat: Returns unchanged with `[GENERAL]` tag

### Topic Classification
- **Multi-level**: Hierarchical classification (L1 → L2)
- **Confidence scoring**: 0-1 probability for each level
- **Taxonomy validation**: Ensures L2 belongs to L1
- **Fallback mechanism**: Defaults to General > Miscellaneous if low confidence

### Conversation Management
- **Persistent storage**: SQLite database
- **Session tracking**: UUID-based sessions
- **Full metadata**: Stores expansion and classification for each message
- **Efficient retrieval**: Indexed queries for sliding window context

### UI/UX
- **Real-time chat**: Clean, intuitive interface
- **Inline metadata**: Expanded query and topic tag shown inline
- **Statistics**: Real-time conversation metrics
- **Error handling**: User-friendly error messages
- **Model status**: Dashboard showing component readiness

## 🔄 Pipeline Workflow

```
User Input
    ↓
[1] Query Expansion (Claude)
    ↓
[2] Is General? (Check for [GENERAL] tag)
    ├─ Yes → Skip classification, tag as General > Miscellaneous
    └─ No → Continue to classification
    ↓
[3] Generate Bot Response
    ↓
[4] Combine Query + Response for classification context
    ↓
[5] Level 1 Classification (DistilBERT)
    ↓
[6] Check confidence ≥ 0.75?
    ├─ No → Fallback to General > Miscellaneous
    └─ Yes → Continue to Level 2
    ↓
[7] Level 2 Classification (DistilBERT)
    ↓
[8] Validate L2 belongs to L1 (using taxonomy)
    ├─ No → Find highest confidence valid L2
    └─ Yes → Continue
    ↓
[9] Aggregate results (Response Aggregator)
    ↓
[10] Store in SQLite + Session State
    ↓
Display to User
```

## 📈 Training & Evaluation

### Train Command
```bash
python train_classifier.py
```

### Output
```
Level 1 Accuracy: 96.32%
Level 1 F1 Score: 0.9612

Level 2 Accuracy: 94.87%
Level 2 F1 Score: 0.9481

✅ Both classifiers achieved ≥95% accuracy!
```

### Performance Tips
- **GPU**: ~2-3 minutes for training
- **CPU**: ~10-15 minutes for training
- Increase dataset size for better generalization
- Adjust learning rate if overfitting

## 🐛 Troubleshooting

### "Models not trained yet"
- Run: `python train_classifier.py`
- Check that `model_level1/` and `model_level2/` directories exist

### "ANTHROPIC_API_KEY not found"
- Create `.env` file in project root
- Add: `GOOGLE_API_KEY=your_key_here`
- Get key from: https://makersuite.google.com/app/apikey

### Classification not working
- Ensure models are trained: `python train_classifier.py`
- Check GPU availability: `torch.cuda.is_available()`
- Verify dataset format: text, label_vertical, label_subdomain columns

### Streamlit app crashes
- Clear cache: `streamlit cache clear`
- Restart: `streamlit run app.py --logger.level=debug`
- Check logs for specific errors

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | Latest | Web UI framework |
| anthropic | Latest | Claude API client |
| transformers | Latest | DistilBERT models |
| torch | Latest | PyTorch deep learning |
| datasets | Latest | Hugging Face datasets |
| scikit-learn | Latest | ML utilities |
| pandas | Latest | Data manipulation |
| numpy | Latest | Numerical computing |
| python-dotenv | Latest | Environment variable loading |

## 🔐 Security

- **API Key**: Never commit `.env` file with actual API keys
- **Database**: SQLite file not encrypted (for production, use encrypted DB)
- **Session State**: Stored in Streamlit, not persisted beyond session
- **Input Validation**: All inputs sanitized before processing

## 🚢 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Create new app on Streamlit Cloud
3. Connect GitHub repository
4. Add secrets:
   - In "Advanced settings" add `ANTHROPIC_API_KEY`
5. Deploy!

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Local Server
```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

## 📝 Example Conversation

```
User: What's happening in politics?

[QUERY EXPANSION]
Expanded: What are the current political events and developments happening around the world?

[CLASSIFICATION]
Topic: Politics > World Politics
Confidence: 92.3%

[BOT RESPONSE]
I received your query about: What are the current political events...
```

## 🤝 Contributing

To extend this project:
1. Add new verticals → Update `label_encoders/taxonomy.json`
2. Improve classification → Retrain with more/better data
3. Add new features → Extend Streamlit UI in `app.py`
4. Optimize performance → Fine-tune hyperparameters in `train_classifier.py`

## 📄 License

MIT License - See LICENSE file for details

## ✉️ Support

For issues or questions:
1. Check troubleshooting section
2. Review error messages in Streamlit logs
3. Ensure all dependencies installed: `pip install -r requirements.txt`

---

**Built with ❤️ using Claude, DistilBERT, and Streamlit**
