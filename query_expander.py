"""
Query Expander: Google Gemini API Integration
Expands short/ambiguous user queries into full standalone questions using conversation history.
"""

import google.generativeai as genai
from typing import Tuple, List, Dict
import os


class QueryExpander:
    """Expands user queries using Gemini API with conversation context."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize QueryExpander with Google Gemini API client.
        
        Args:
            api_key: Google API key. If None, loads from GOOGLE_API_KEY env var
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in .env file or environment."
            )
        
        genai.configure(api_key=api_key)
        self.model = "gemini-2.5-flash"
        
        # System prompt (exactly as specified)
        self.system_prompt = """You are a query expansion assistant in a conversational chatbot.
Given the conversation history and the latest user message, rewrite
the user message as a complete, self-contained question or statement.
Resolve all pronouns, references, and ambiguity from history.
If the message is a greeting, chitchat, interruption or acknowledgement,
return it unchanged and append the tag [GENERAL] at the end.
Do NOT answer the question. Only rewrite it.
Return only the rewritten query. Nothing else."""
    
    def format_context(self, messages: List[Dict]) -> str:
        """
        Format message history as context string.
        
        Args:
            messages: List of message dictionaries with role and message keys
        
        Returns:
            Formatted context string
        """
        if not messages:
            return "No conversation history yet."
        
        context_lines = []
        for msg in messages:
            role = "User" if msg.get("role") == "user" else "Assistant"
            text = msg.get("message", "")
            context_lines.append(f"{role}: {text}")
        
        return "\n".join(context_lines)
    
    def expand_query(self, user_query: str, context_messages: List[Dict]) -> Tuple[str, bool]:
        """
        Expand user query using Gemini API with conversation context.
        
        Args:
            user_query: The current user message to expand
            context_messages: List of previous messages (last 20) for context
        
        Returns:
            Tuple of (expanded_query, is_general)
            - expanded_query: The expanded/rewritten query
            - is_general: True if query was tagged as [GENERAL] (skip classification)
        """
        # Format context from previous messages
        context = self.format_context(context_messages)
        
        # Build the user message for Gemini
        full_prompt = f"""{self.system_prompt}

Conversation history:
{context}

Latest user message to expand:
{user_query}

Rewrite the latest user message as a complete, self-contained statement. If it's general chitchat, append [GENERAL]."""
        
        # Call Gemini API
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(full_prompt)
        
        expanded_query = response.text.strip()
        
        # Check if response contains [GENERAL] tag
        is_general = "[GENERAL]" in expanded_query
        
        # Remove the [GENERAL] tag from the expanded query if present
        if is_general:
            expanded_query = expanded_query.replace("[GENERAL]", "").strip()
        
        return expanded_query, is_general
    
    def get_model_info(self) -> Dict:
        """Return information about the query expander model."""
        return {
            "model": self.model,
            "provider": "Google",
            "purpose": "Query expansion with context"
        }
