"""
Conversation Manager
====================

This module manages the conversation state for CowDoc AI.

Responsibilities
----------------
- Remember the original question
- Remember the clarification question
- Remember whether clarification is pending
- Remember image upload state
- Remember image classification results
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

    # ------------------------------------------------------
    # Clarification State
    # ------------------------------------------------------

    if "clarification_pending" not in session_state:
        session_state["clarification_pending"] = False

    if "original_question" not in session_state:
        session_state["original_question"] = None

    if "clarifying_question" not in session_state:
        session_state["clarifying_question"] = None

    # ------------------------------------------------------
    # Image State
    # ------------------------------------------------------

    # Has the user uploaded an image?
    if "image_uploaded" not in session_state:
        session_state["image_uploaded"] = False

    # Uploaded image object/path
    if "uploaded_image" not in session_state:
        session_state["uploaded_image"] = None

    # Full classifier output
    if "classifier_result" not in session_state:
        session_state["classifier_result"] = None

    # Candidate diseases (Top-K)
    if "candidate_diseases" not in session_state:
        session_state["candidate_diseases"] = []

    # Waiting for image upload?
    if "waiting_for_image" not in session_state:
        session_state["waiting_for_image"] = False

    # User skipped image upload
    if "image_skipped" not in session_state:
        session_state["image_skipped"] = False


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
    Combine the original question,
    the assistant's clarification question,
    and the farmer's reply.
    """

    original_question = session_state.get(
        "original_question",
        ""
    )

    clarifying_question = session_state.get(
        "clarifying_question",
        ""
    )

    combined_question = f"""
Original Question:
{original_question}

Previous Assistant Question:
{clarifying_question}

Farmer Reply:
{followup_answer}
"""

    return combined_question.strip()

# ==========================================================
# Image State Helpers
# ==========================================================

def start_waiting_for_image(session_state):
    """
    Chatbot is requesting an image.
    """

    session_state["waiting_for_image"] = True
    session_state["image_skipped"] = False


def save_uploaded_image(
    session_state,
    uploaded_image
):
    """
    Store uploaded image.
    """

    session_state["uploaded_image"] = uploaded_image
    session_state["image_uploaded"] = True
    session_state["waiting_for_image"] = False
    session_state["image_skipped"] = False


def save_classifier_result(
    session_state,
    classifier_result
):
    """
    Save EfficientNet prediction results.
    """

    session_state["classifier_result"] = classifier_result

    session_state["candidate_diseases"] = [

        item["disease"]

        for item in classifier_result

    ]


def skip_image(session_state):
    """
    User chooses to continue without image.
    """

    session_state["image_skipped"] = True
    session_state["waiting_for_image"] = False


def has_uploaded_image(session_state):
    """
    Returns True if image exists.
    """

    return session_state.get(
        "image_uploaded",
        False
    )


def is_waiting_for_image(session_state):
    """
    Returns True if chatbot is waiting
    for an image.
    """

    return session_state.get(
        "waiting_for_image",
        False
    )


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

    session_state["image_uploaded"] = False
    session_state["uploaded_image"] = None
    session_state["classifier_result"] = None
    session_state["candidate_diseases"] = []
    session_state["waiting_for_image"] = False
    session_state["image_skipped"] = False