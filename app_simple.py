"""
Streamlit UI: Query Expander Testing
Simple app focused on testing query expansion with Gemini API
"""

import streamlit as st
import os
from dotenv import load_dotenv
from conversation_manager import ConversationManager
from query_expander import QueryExpander

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Query Expander Test",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .expanded-query {
        font-style: italic;
        color: #555;
        padding: 12px;
        background-color: #fff9e6;
        border-radius: 4px;
        border-left: 4px solid #ff7f0e;
        margin: 10px 0;
    }
    .original-query {
        padding: 12px;
        background-color: #f0f2f6;
        border-radius: 4px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .general-tag {
        display: inline-block;
        background-color: #e8f4f8;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        color: #0077b6;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_conversation_manager():
    """Initialize conversation manager."""
    return ConversationManager(db_path="conversations.db")


@st.cache_resource
def initialize_query_expander():
    """Initialize query expander with Gemini API."""
    try:
        return QueryExpander()
    except ValueError as e:
        st.error(f"❌ Failed to initialize Query Expander: {str(e)}")
        return None


def generate_bot_response(expanded_query: str) -> str:
    """Generate a bot response using Gemini for the expanded query."""
    try:
        import google.generativeai as genai
        
        # Configure API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "❌ API key not found"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Generate response
        prompt = f"Answer this question concisely: {expanded_query}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not generate response: {str(e)}"


def main():
    """Main app."""
    st.title("🔍 Query Expander Test")
    st.markdown("""
    Test the query expansion feature with Gemini API.
    See how ambiguous queries are expanded using conversation context.
    """)
    
    # Initialize session_id and messages if not already done
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize components
    conv_manager = initialize_conversation_manager()
    query_expander = initialize_query_expander()
    
    if not query_expander:
        st.error("❌ Query Expander initialization failed. Please check your .env file.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Info")
        st.success("✅ Query Expander Ready")
        
        # Conversation stats
        stats = conv_manager.get_conversation_stats()
        st.subheader("Conversation Stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", stats["total_messages"])
        with col2:
            st.metric("User Messages", stats["user_messages"])
        with col3:
            st.metric("Bot Messages", stats["assistant_messages"])
        
        # Clear button
        if st.button("🗑️ Clear Conversation"):
            conv_manager.clear_session()
            st.rerun()
    
    # Main chat area
    st.markdown("---")
    st.subheader("Conversation History")
    
    messages = conv_manager.get_all_messages()
    
    if not messages:
        st.info("👋 Start a conversation to see query expansion in action!")
    else:
        for msg_dict in messages:
            with st.chat_message(msg_dict["role"]):
                st.markdown(msg_dict["message"])
                
                # Show expansion info for user messages
                if msg_dict["role"] == "user" and msg_dict.get("expanded_query"):
                    st.markdown("---")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**Original:**")
                        st.markdown(
                            f'<div class="original-query">{msg_dict["message"]}</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown("**Expanded:**")
                        st.markdown(
                            f'<div class="expanded-query">{msg_dict["expanded_query"]}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Show general tag if applicable
                        if msg_dict.get("is_general", False):
                            st.markdown(
                                '<div class="general-tag">🏷️ GENERAL/CHITCHAT</div>',
                                unsafe_allow_html=True
                            )
    
    # Input area
    st.markdown("---")
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([0.9, 0.1])
        
        with col1:
            user_input = st.text_input(
                "Your message:",
                placeholder="Type a message (e.g., 'what about it?', 'tell me more')",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("📤 Send")
    
    # Process user input
    if submit_button and user_input:
        # Add user message to conversation
        conv_manager.add_message("user", user_input)
        
        with st.spinner("🔄 Expanding query..."):
            try:
                # Get context (last 20 messages, excluding the one we just added)
                context_messages = conv_manager.get_last_n_messages(n=20)
                if len(context_messages) > 0 and context_messages[-1]["message"] == user_input:
                    context_messages = context_messages[:-1]
                
                # Step 1: Expand query
                expanded_query, is_general = query_expander.expand_query(
                    user_input,
                    context_messages
                )
                
                # Update message with expansion info
                messages = conv_manager.get_all_messages()
                last_user_msg_idx = len(messages) - 1
                
                if last_user_msg_idx >= 0 and messages[last_user_msg_idx]["role"] == "user":
                    # Update in database
                    import sqlite3
                    conn = sqlite3.connect("conversations.db")
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE conversations 
                        SET expanded_query = ?
                        WHERE id = ?
                    """, (expanded_query, messages[last_user_msg_idx]["id"]))
                    conn.commit()
                    conn.close()
                    
                    # Update in session state
                    st.session_state.messages[last_user_msg_idx]["expanded_query"] = expanded_query
                    st.session_state.messages[last_user_msg_idx]["is_general"] = is_general
                
                # Step 2: Generate bot response for the expanded query
                bot_response = generate_bot_response(expanded_query)
                conv_manager.add_message("assistant", bot_response)
                
                # Rerun to display updated conversation
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
