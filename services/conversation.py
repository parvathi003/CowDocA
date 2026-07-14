"""
Conversation Manager
====================

This module manages the conversation state for CowDoctor AI.

Responsibilities
----------------
- Remember the original question
- Remember the clarification question
- Remember whether clarification is pending
- Combine the original question with the user's follow-up
- Reset the conversation state

This module NEVER:
- Calls the LLM
- Calls RAG
- Detects diseases
- Answers questions
"""

# ==========================================================
# Initialize Conversation State
# ==========================================================

def initialize_conversation(session_state):
    """
    Initialize conversation variables if they
    do not already exist.
    """

    if "clarification_pending" not in session_state:
        session_state["clarification_pending"] = False

    if "original_question" not in session_state:
        session_state["original_question"] = None

    if "clarifying_question" not in session_state:
        session_state["clarifying_question"] = None


# ==========================================================
# Start Clarification
# ==========================================================

def start_clarification(
    session_state,
    original_question,
    clarifying_question
):
    """
    Save the clarification state.
    """

    session_state["clarification_pending"] = True
    session_state["original_question"] = original_question
    session_state["clarifying_question"] = clarifying_question


# ==========================================================
# Check Clarification Status
# ==========================================================

def is_waiting_for_clarification(session_state):
    """
    Returns True if clarification is pending.
    """

    return session_state.get(
        "clarification_pending",
        False
    )


# ==========================================================
# Get Clarifying Question
# ==========================================================

def get_clarifying_question(session_state):
    """
    Return the current clarification question.
    """

    return session_state.get(
        "clarifying_question",
        None
    )


# ==========================================================
# Build Combined Question
# ==========================================================

def build_clarification_input(
    session_state,
    followup_answer
):
    """
    Combine the original question and the
    user's follow-up answer.

    This combined text is sent back to Stage 1.
    """

    original_question = session_state.get(
        "original_question",
        ""
    )

    combined_question = f"""
Original Question:
{original_question}

Farmer Follow-up:
{followup_answer}
"""

    return combined_question.strip()


# ==========================================================
# Clear Clarification State
# ==========================================================

def clear_clarification(session_state):
    """
    Clear clarification information after
    Stage 1 successfully processes the follow-up.
    """

    session_state["clarification_pending"] = False
    session_state["original_question"] = None
    session_state["clarifying_question"] = None


# ==========================================================
# Reset Conversation
# ==========================================================

def reset_conversation(session_state):
    """
    Reset the entire conversation state.
    """

    clear_clarification(session_state)