#!/usr/bin/env python
"""Debug script to check current Streamlit session and database state."""

import streamlit as st
import sqlite3
import os

# Initialize session  
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

st.title("Session Debug")

# Display current session_id
st.write(f"**Current Session ID:** {st.session_state.session_id}")

# Query database for this session
conn = sqlite3.connect("conversations.db")
cursor = conn.cursor()
cursor.execute(
    "SELECT COUNT(*) FROM conversations WHERE session_id = ?",
    (st.session_state.session_id,)
)
count = cursor.fetchone()[0]
conn.close()

st.write(f"**Messages in this session:** {count}")

# Show all sessions
conn = sqlite3.connect("conversations.db")
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT session_id FROM conversations")
sessions = cursor.fetchall()
conn.close()

st.write(f"**Total sessions in DB:** {len(sessions)}")
for i, session in enumerate(sessions, 1):
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session[0],))
    count = cursor.fetchone()[0]
    conn.close()
    marker = "← Current" if session[0] == st.session_state.session_id else ""
    st.write(f"{i}. {session[0][:20]}... ({count} messages) {marker}")
