"""
Conversation Manager: SQLite + Streamlit Session State Integration
Handles persistent storage and retrieval of conversation history.
"""

import sqlite3
import streamlit as st
from datetime import datetime
import os
import json


class ConversationManager:
    """Manages conversation history using SQLite database and Streamlit session state."""
    
    def __init__(self, db_path: str = "conversations.db"):
        """
        Initialize ConversationManager with database path.
        
        Args:
            db_path: Path to SQLite database file. Defaults to conversations.db
        """
        self.db_path = db_path
        self._initialize_db()
        self._initialize_session_state()
    
    def _initialize_db(self):
        """Create conversations table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                expanded_query TEXT,
                level1_topic TEXT,
                level2_topic TEXT,
                confidence REAL
            )
        """)
        cursor.execute("PRAGMA table_info(conversations)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "level1_confidence" not in existing_columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN level1_confidence REAL")
        if "level2_confidence" not in existing_columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN level2_confidence REAL")
        conn.commit()
        conn.close()
    
    def _initialize_session_state(self):
        """Initialize session state variables if they don't exist."""
        try:
            if "session_id" not in st.session_state:
                # Generate a unique session ID (using current timestamp + random ID)
                import uuid
                st.session_state.session_id = str(uuid.uuid4())
            
            if "messages" not in st.session_state:
                # Load existing messages from database for this session
                st.session_state.messages = self._load_messages_from_db()
        except AttributeError:
            # If session_state hasn't been initialized yet, skip for now
            pass
    
    def _load_messages_from_db(self):
        """Load all messages for current session from SQLite."""
        # Ensure session_id exists
        if "session_id" not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
             SELECT id, role, message, timestamp, expanded_query, level1_topic, 
                 level2_topic, confidence, level1_confidence, level2_confidence
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (st.session_state.session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            messages.append({
                "id": row["id"],
                "role": row["role"],
                "message": row["message"],
                "timestamp": row["timestamp"],
                "expanded_query": row["expanded_query"],
                "level1_topic": row["level1_topic"],
                "level2_topic": row["level2_topic"],
                "confidence": row["confidence"],
                "level1_confidence": row["level1_confidence"],
                "level2_confidence": row["level2_confidence"]
            })
        
        return messages
    
    def add_message(
        self, 
        role: str, 
        message: str,
        expanded_query: str = None,
        level1_topic: str = None,
        level2_topic: str = None,
        confidence: float = None,
        level1_confidence: float = None,
        level2_confidence: float = None
    ) -> int:
        """
        Add a message to both SQLite and session state.
        
        Args:
            role: "user" or "assistant"
            message: The message content
            expanded_query: Expanded version of user query (for user messages)
            level1_topic: Level 1 classification topic
            level2_topic: Level 2 classification subdomain
            confidence: Classification confidence score
            level1_confidence: Level 1 confidence score
            level2_confidence: Level 2 confidence score
        
        Returns:
            Message ID from database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations 
            (session_id, role, message, expanded_query, level1_topic, level2_topic, confidence, level1_confidence, level2_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            st.session_state.session_id,
            role,
            message,
            expanded_query,
            level1_topic,
            level2_topic,
            confidence,
            level1_confidence,
            level2_confidence
        ))
        
        conn.commit()
        message_id = cursor.lastrowid
        conn.close()
        
        # Also add to session state
        msg_dict = {
            "id": message_id,
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "expanded_query": expanded_query,
            "level1_topic": level1_topic,
            "level2_topic": level2_topic,
            "confidence": confidence,
            "level1_confidence": level1_confidence,
            "level2_confidence": level2_confidence
        }
        
        st.session_state.messages.append(msg_dict)
        
        return message_id
    
    def get_last_n_messages(self, n: int = 20):
        """
        Get the last N messages for context window.
        
        Args:
            n: Number of messages to retrieve (default 20 for 10 dialogue exchanges)
        
        Returns:
            List of message dictionaries ordered by timestamp DESC
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
             SELECT id, role, message, timestamp, expanded_query, level1_topic,
                 level2_topic, confidence, level1_confidence, level2_confidence
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (st.session_state.session_id, n))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Reverse to get oldest first
        messages = []
        for row in reversed(rows):
            messages.append({
                "id": row["id"],
                "role": row["role"],
                "message": row["message"],
                "timestamp": row["timestamp"],
                "expanded_query": row["expanded_query"],
                "level1_topic": row["level1_topic"],
                "level2_topic": row["level2_topic"],
                "confidence": row["confidence"],
                "level1_confidence": row["level1_confidence"],
                "level2_confidence": row["level2_confidence"]
            })
        
        return messages
    
    def get_all_messages(self):
        """Get all messages in the current session ordered by timestamp."""
        return st.session_state.messages
    
    def clear_session(self):
        """Clear all messages for current session from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM conversations WHERE session_id = ?
        """, (st.session_state.session_id,))
        conn.commit()
        conn.close()
        
        # Clear session state
        st.session_state.messages = []
    
    def get_conversation_stats(self):
        """Get statistics about the conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total messages
        cursor.execute("""
            SELECT COUNT(*) FROM conversations WHERE session_id = ?
        """, (st.session_state.session_id,))
        total_messages = cursor.fetchone()[0]
        
        # User vs assistant messages
        cursor.execute("""
            SELECT role, COUNT(*) as count FROM conversations 
            WHERE session_id = ?
            GROUP BY role
        """, (st.session_state.session_id,))
        
        role_counts = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return {
            "total_messages": total_messages,
            "user_messages": role_counts.get("user", 0),
            "assistant_messages": role_counts.get("assistant", 0)
        }
