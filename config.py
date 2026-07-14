"""
CowDoctor Configuration
=======================

This file contains all global configuration values used
throughout the project.

Every module should import values from this file instead
of hardcoding them.

Architecture
------------

Streamlit
    ↓
Conversation Manager
    ↓
Stage 1 LLM (Understanding)
    ↓
Disease Filtered RAG
    ↓
Retrieval Check
    ↓
Stage 2 LLM (Answer Generation)
"""

# ==========================================================
# Project Information
# ==========================================================

PROJECT_NAME = "CowDoctor AI"

VERSION = "2.0"

# ==========================================================
# Supported Diseases
# ==========================================================

SUPPORTED_DISEASES = [

    "FMD",

    "LSD",

    "Mastitis",

    "Foot Rot",

    "Ringworm"

]

# ==========================================================
# ChromaDB Configuration
# ==========================================================

CHROMA_DB_PATH = "chroma_db"

CHROMA_COLLECTION = "cattle_disease_knowledge"

# Number of chunks retrieved from RAG
RAG_TOP_K = 5

# ==========================================================
# OpenAI Models
# ==========================================================

# Stage 1
# Understanding Module
STAGE1_MODEL = "gpt-4.1-mini"

# Stage 2
# Answer Generation Module
STAGE2_MODEL = "gpt-4.1-mini"

# ==========================================================
# Application Settings
# ==========================================================

# Maximum conversation turns kept in chat history
MAX_CHAT_HISTORY = 20

# Default fallback message if RAG cannot retrieve information
NO_RAG_RESPONSE = (
    "I couldn't find enough verified information "
    "in the current knowledge base."
)

# Default message for temporary LLM/API failures
SYSTEM_ERROR_MESSAGE = (
    "Sorry, I'm having trouble processing your request "
    "right now. Please try again."
)