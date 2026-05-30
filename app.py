"""
Streamlit UI: Main Application
Real-time conversational AI pipeline with query expansion and topic classification.
"""

import streamlit as st
import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from conversation_manager import ConversationManager
from query_expander import QueryExpander
from response_aggregator import ResponseAggregator

# Load environment variables from .env file
load_dotenv()

# Model paths (absolute)
_MODELS_BASE = os.path.join(os.path.dirname(__file__), "Models")
MODEL_L1_DIR = os.path.join(_MODELS_BASE, "model_level1")
MODEL_L2_DIR = os.path.join(_MODELS_BASE, "model_level2")
ENCODER_DIR = os.path.join(_MODELS_BASE, "label_encoders")
CONFIDENCE_THR = 0.75
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Page configuration
st.set_page_config(
    page_title="Conversational AI Pipeline",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state early
if "session_id" not in st.session_state:
    import uuid
    import json
    
    # Try to load session_id from persistent file
    session_file = ".streamlit_session_id"
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r') as f:
                st.session_state.session_id = json.load(f).get("session_id", str(uuid.uuid4()))
        except:
            st.session_state.session_id = str(uuid.uuid4())
    else:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Save session_id to file for persistence
    try:
        with open(session_file, 'w') as f:
            json.dump({"session_id": st.session_state.session_id}, f)
    except:
        pass

if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS for better UI
st.markdown("""
    <style>
    .info-box {
        background-color: #f0f2f6;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .topic-tag {
        background-color: #e8f4f8;
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        font-size: 12px;
        font-weight: 600;
        color: #000;
    }
    .confidence-high {
        color: #1f77b4;
    }
    .confidence-medium {
        color: #ff7f0e;
    }
    .confidence-low {
        color: #d62728;
    }
    .expanded-query {
        font-style: italic;
        color: #555;
        padding: 8px;
        background-color: #fff9e6;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_conversation_manager():
    """Initialize conversation manager (cached to avoid reinit on reruns)."""
    return ConversationManager(db_path="conversations.db")


@st.cache_resource
def initialize_query_expander():
    """Initialize query expander with Claude API (cached)."""
    try:
        return QueryExpander()
    except ValueError as e:
        return None


@st.cache_resource
def initialize_topic_classifier():
    """Load pre-trained topic classifier models directly (cached) - optional."""
    result = {
        'tokenizer': None,
        'model_l1': None,
        'model_l2': None,
        'le_l1': None,
        'le_l2': None,
        'taxonomy': None,
        'ready': False,
        'error': None
    }
    
    try:
        # Check if encoders and taxonomy exist
        le_l1_path = os.path.join(ENCODER_DIR, "le_level1.pkl")
        le_l2_path = os.path.join(ENCODER_DIR, "le_level2.pkl")
        tax_path = os.path.join(ENCODER_DIR, "taxonomy.json")
        
        # If any encoder files are missing, return not ready but no error  
        # (classifier is optional)
        if not all(os.path.exists(p) for p in [le_l1_path, le_l2_path, tax_path]):
            result['error'] = "Encoder files not found - run training notebook to generate them"
            return result
        
        # Try loading tokenizer - use MODEL_L1_DIR first, fallback to checkpoint
        try:
            result['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_L1_DIR)
        except Exception:
            try:
                checkpoint_dir = os.path.join(_MODELS_BASE, "checkpoint-438")
                result['tokenizer'] = AutoTokenizer.from_pretrained(checkpoint_dir)
            except Exception as e:
                result['error'] = f"Could not load tokenizer: {str(e)}"
                return result
        
        # Try loading models
        try:
            result['model_l1'] = DistilBertForSequenceClassification.from_pretrained(
                MODEL_L1_DIR,
                low_cpu_mem_usage=True
            ).to(DEVICE).eval()
        except Exception:
            checkpoint_dir = os.path.join(_MODELS_BASE, "checkpoint-438")
            if os.path.exists(checkpoint_dir):
                result['model_l1'] = DistilBertForSequenceClassification.from_pretrained(
                    checkpoint_dir,
                    low_cpu_mem_usage=True
                ).to(DEVICE).eval()
        
        try:
            result['model_l2'] = DistilBertForSequenceClassification.from_pretrained(
                MODEL_L2_DIR,
                low_cpu_mem_usage=True
            ).to(DEVICE).eval()
        except Exception:
            checkpoint_dir = os.path.join(_MODELS_BASE, "checkpoint-438")
            if os.path.exists(checkpoint_dir):
                result['model_l2'] = DistilBertForSequenceClassification.from_pretrained(
                    checkpoint_dir,
                    low_cpu_mem_usage=True
                ).to(DEVICE).eval()
        
        # Load encoders
        with open(le_l1_path, "rb") as f:
            result['le_l1'] = pickle.load(f)
        with open(le_l2_path, "rb") as f:
            result['le_l2'] = pickle.load(f)
        
        # Load taxonomy
        with open(tax_path) as f:
            result['taxonomy'] = json.load(f)

        # Verify model-vs-encoder compatibility before marking the classifier ready.
        l1_num_labels = int(getattr(result['model_l1'].config, 'num_labels', 0) or 0) if result['model_l1'] else 0
        l2_num_labels = int(getattr(result['model_l2'].config, 'num_labels', 0) or 0) if result['model_l2'] else 0
        l1_encoder_labels = len(result['le_l1'].classes_) if result['le_l1'] else 0
        l2_encoder_labels = len(result['le_l2'].classes_) if result['le_l2'] else 0

        if not result['model_l1'] or not result['model_l2'] or not result['tokenizer']:
            result['error'] = "Classifier components could not be loaded from Models/"
            return result

        if l1_num_labels and l1_num_labels != l1_encoder_labels:
            result['error'] = (
                f"L1 model mismatch: model has {l1_num_labels} labels but encoder has {l1_encoder_labels}."
            )
            return result

        if l2_num_labels and l2_num_labels != l2_encoder_labels:
            result['error'] = (
                f"L2 model mismatch: model has {l2_num_labels} labels but encoder has {l2_encoder_labels}. "
                f"Replace {MODEL_L2_DIR} with the trained 41-class L2 model before using classification."
            )
            return result
        
        result['ready'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


def _infer(model, tokenizer, text):
    """Run inference: returns (predicted_idx, softmax_probs)."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


def predict_topic(combined_text, models_dict):
    """Classify the combined expanded query and bot answer into L1 and L2 topics."""
    result = {
        'level1': 'General',
        'level2': 'Miscellaneous/General Knowledge',
        'level1_confidence': 0.0,
        'level2_confidence': 0.0,
        'level1_result': {'label': 'General', 'confidence': 0.0},
        'level2_result': {'label': 'Miscellaneous/General Knowledge', 'confidence': 0.0},
        'confidence': 0.0,
        'success': False,
        'error': None
    }
    
    try:
        if not models_dict.get('ready'):
            result['error'] = models_dict.get('error', 'Models not loaded')
            return result
        
        tokenizer = models_dict['tokenizer']
        model_l1 = models_dict['model_l1']
        model_l2 = models_dict['model_l2']
        le_l1 = models_dict['le_l1']
        le_l2 = models_dict['le_l2']
        taxonomy = models_dict['taxonomy']
        
        # L1 inference
        l1_idx, l1_probs = _infer(model_l1, tokenizer, combined_text)
        l1_conf = float(l1_probs[l1_idx])
        l1_label = le_l1.inverse_transform([l1_idx])[0]
        result['level1_result'] = {
            'label': l1_label,
            'confidence': l1_conf,
        }
        
        # Confidence threshold
        if l1_conf < CONFIDENCE_THR:
            result['level1'] = 'General'
            result['level2'] = 'Miscellaneous/General Knowledge'
            result['level1_confidence'] = l1_conf
            result['level2_confidence'] = 0.0
            result['level1_result'] = {
                'label': 'General',
                'confidence': l1_conf,
            }
            result['level2_result'] = {
                'label': 'Miscellaneous/General Knowledge',
                'confidence': 0.0,
            }
            result['confidence'] = l1_conf
            result['success'] = True
            return result
        
        # L2 inference is driven by the predicted L1 label.
        valid_subdomains = taxonomy.get(l1_label, [])

        l2_idx, l2_probs = _infer(model_l2, tokenizer, combined_text)
        l2_label = le_l2.inverse_transform([l2_idx])[0]
        l2_conf = float(l2_probs[l2_idx])
        
        # Taxonomy gating keeps L2 inside the predicted L1 branch.
        if l2_label not in valid_subdomains and valid_subdomains:
            valid_indices = [i for i, cls in enumerate(le_l2.classes_) if cls in valid_subdomains]
            if valid_indices:
                best_idx = max(valid_indices, key=lambda i: l2_probs[i])
                l2_label = le_l2.inverse_transform([best_idx])[0]
                l2_conf = float(l2_probs[best_idx])
        
        result['level1'] = l1_label
        result['level2'] = l2_label
        result['level1_confidence'] = l1_conf
        result['level2_confidence'] = l2_conf
        result['level1_result'] = {
            'label': l1_label,
            'confidence': l1_conf,
        }
        result['level2_result'] = {
            'label': l2_label,
            'confidence': l2_conf,
        }
        result['confidence'] = l1_conf
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


def get_confidence_color(confidence: float) -> str:
    """Get color class based on confidence level."""
    if confidence >= 0.85:
        return "confidence-high"
    elif confidence >= 0.75:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_message_with_metadata(msg_dict: dict):
    """Display a message with expanded query and topic tag inline."""
    with st.chat_message(msg_dict["role"]):
        st.markdown(msg_dict["message"])
        
        # If user message, show expanded query and topic info inline
        if msg_dict["role"] == "user":
            # Only show expanded query if not marked as general
            if msg_dict.get("expanded_query"):
                st.markdown("---")
                
                # Expanded query
                st.markdown("**🔍 Expanded:**")
                st.markdown(
                    f'<div class="expanded-query">{msg_dict["expanded_query"]}</div>',
                    unsafe_allow_html=True
                )
            
            # Topic classification
            if msg_dict.get("level1_topic"):
                if not msg_dict.get("expanded_query"):
                    st.markdown("---")
                
                topic_tag = f"{msg_dict['level1_topic']} > {msg_dict['level2_topic']}"
                confidence = msg_dict.get("confidence", 0.0)
                l1_confidence = msg_dict.get("level1_confidence", confidence)
                l2_confidence = msg_dict.get("level2_confidence", 0.0)
                conf_color = get_confidence_color(l1_confidence)
                confidence_percent = round(confidence * 100, 1)
                l1_confidence_percent = round(l1_confidence * 100, 1)
                l2_confidence_percent = round(l2_confidence * 100, 1)
                
                st.markdown("**🏷️ Topic:**")
                st.markdown(
                    f'<div class="topic-tag">{topic_tag}</div><br/>'
                    f'<span class="{conf_color}">Overall Confidence: {confidence_percent}%</span><br/>'
                    f'<span class="{conf_color}">L1: {l1_confidence_percent}%</span><br/>'
                    f'<span class="{conf_color}">L2: {l2_confidence_percent}%</span>',
                    unsafe_allow_html=True
                )


def generate_bot_response(expanded_query: str) -> str:
    """Generate a Gemini response for the expanded query."""
    try:
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "❌ API key not found"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"Answer this question concisely and accurately: {expanded_query}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Could not generate response: {str(e)}"


def main():
    """Main Streamlit application."""
    st.title("💬 Query Expander Pipeline")
    st.markdown("""
    A real-time pipeline that:
    1. **Expands** short/ambiguous queries using Gemini API
    2. **Classifies** messages into a 2-level topic hierarchy 
    """)
    
    # Initialize components
    conv_manager = initialize_conversation_manager()
    query_expander = initialize_query_expander()
    classifier_models = initialize_topic_classifier()
    
    # Check if classifier is ready
    classifier_status = classifier_models
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show component status
        st.subheader("Component Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if query_expander:
                st.success("✅ Query Expander")
            else:
                st.error("❌ Query Expander")
        
        with col2:
            if classifier_status["ready"]:
                st.success("✅ Classifier")
            else:
                st.error("❌ Classifier")
        
        # If classifier not ready, show helpful message
        if not classifier_status.get("ready"):
            st.warning(
                f"⚠️ **Classifier not loaded**\n\n"
                f"Error: {classifier_status.get('error', 'Unknown error')}"
            )
        
        # Show classifier details if ready
        if classifier_status.get("ready"):
            st.subheader("Classifier Info")
            l1_classes = len(classifier_models['le_l1'].classes_) if classifier_models.get('le_l1') else 0
            l2_classes = len(classifier_models['le_l2'].classes_) if classifier_models.get('le_l2') else 0
            st.info(
                f"- **Status:** ✅ Ready\n"
                f"- **Level 1 Classes:** {l1_classes}\n"
                f"- **Level 2 Classes:** {l2_classes}"
            )
        else:
            st.error(f"❌ {classifier_status.get('error', 'Classifier not available')}")
        
        # Conversation stats
        st.subheader("Conversation Stats")
        stats = conv_manager.get_conversation_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", stats["total_messages"])
        with col2:
            st.metric("User Messages", stats["user_messages"])
        with col3:
            st.metric("Bot Messages", stats["assistant_messages"])
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            conv_manager.clear_session()
            st.rerun()
    
    # Main chat area
    # Display conversation history
    messages = conv_manager.get_all_messages()
    
    if not messages:
        st.info("👋 Welcome! Start a conversation to see the pipeline in action.")
    else:
        for msg_dict in messages:
            display_message_with_metadata(msg_dict)
    
    # Input area at bottom
    st.markdown("---")
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([0.9, 0.1])
        
        with col1:
            user_input = st.text_input(
                "Your message:",
                placeholder="Type your question or statement here...",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("📤 Send", use_container_width=True)
    
    # Process user input
    if submit_button and user_input:
        # Check if expander is available
        if not query_expander:
            st.error("❌ Query Expander not initialized. Check your GOOGLE_API_KEY in .env file.")
            return

        if classifier_status.get("error"):
            st.warning(f"⚠️ {classifier_status['error']}")
        
        # Show processing message
        with st.spinner("🔄 Processing your message..."):
            try:
                # Get the last 20 prior messages for Gemini context.
                context_messages = conv_manager.get_last_n_messages(n=20)

                # Add the new user message after capturing context so it does not
                # consume one slot from the Gemini history window.
                conv_manager.add_message("user", user_input)
                
                # Step 1: Expand query
                print(f"DEBUG: About to expand query: {user_input}")
                expanded_query, is_general = query_expander.expand_query(
                    user_input, 
                    context_messages
                )
                print(f"DEBUG: Query expanded to: {expanded_query}, is_general={is_general}")
                
                # Step 2: Generate bot response
                print(f"DEBUG: Generating bot response for: {expanded_query}")
                bot_response = generate_bot_response(expanded_query)
                print(f"DEBUG: Bot response generated: {bot_response[:50]}...")
                
                # Step 3: Classify (use classifier if available)
                level1_topic = None
                level2_topic = None
                confidence = 0.0
                level1_confidence = 0.0
                level2_confidence = 0.0
                classification_error = None
                
                if classifier_status.get("ready"):
                    classification_text = f"{expanded_query} [SEP] {bot_response}"
                    classification_result = predict_topic(
                        classification_text,
                        classifier_models
                    )
                    
                    if classification_result.get("success"):
                        level1_topic = classification_result.get("level1", "General")
                        level2_topic = classification_result.get("level2", "Miscellaneous/General Knowledge")
                        confidence = classification_result.get("confidence", 0.0)
                        level1_confidence = classification_result.get("level1_confidence", confidence)
                        level2_confidence = classification_result.get("level2_confidence", 0.0)
                    else:
                        classification_error = classification_result.get("error")
                else:
                    classification_error = classifier_status.get("error") or "Classifier not available"
                
                # Step 4: Aggregate results
                aggregated = ResponseAggregator.aggregate(
                    original_query=user_input,
                    expanded_query=expanded_query,
                    level1_topic=level1_topic,
                    level2_topic=level2_topic,
                    confidence=confidence,
                    level1_confidence=level1_confidence,
                    level2_confidence=level2_confidence,
                    bot_answer=bot_response,
                    is_general=is_general,
                    classification_error=classification_error
                )
                
                # Update message with metadata
                messages = conv_manager.get_all_messages()
                last_user_msg_idx = len(messages) - 1
                if last_user_msg_idx >= 0 and messages[last_user_msg_idx]["role"] == "user":
                    # Update in database
                    conn = __import__("sqlite3").connect("conversations.db")
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE conversations 
                        SET expanded_query = ?, level1_topic = ?, level2_topic = ?, confidence = ?, level1_confidence = ?, level2_confidence = ?
                        WHERE id = ?
                    """, (
                        expanded_query,
                        level1_topic,
                        level2_topic,
                        confidence,
                        level1_confidence,
                        level2_confidence,
                        messages[last_user_msg_idx]["id"]
                    ))
                    conn.commit()
                    conn.close()

                    st.session_state.messages[last_user_msg_idx]["expanded_query"] = expanded_query
                    st.session_state.messages[last_user_msg_idx]["level1_topic"] = level1_topic
                    st.session_state.messages[last_user_msg_idx]["level2_topic"] = level2_topic
                    st.session_state.messages[last_user_msg_idx]["confidence"] = confidence
                    st.session_state.messages[last_user_msg_idx]["level1_confidence"] = level1_confidence
                    st.session_state.messages[last_user_msg_idx]["level2_confidence"] = level2_confidence

                if classification_error:
                    st.error(f"❌ Topic classification unavailable: {classification_error}")
                
                # Add bot response
                print(f"DEBUG: Adding bot response to conversation")
                conv_manager.add_message("assistant", bot_response)
                print(f"DEBUG: Bot message added, calling rerun()")
                
                # Rerun to display updated conversation
                st.rerun()
                
            except Exception as e:
                print(f"DEBUG: Error in message processing: {str(e)}")
                import traceback
                traceback.print_exc()
                st.error(f"❌ Error processing message: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
