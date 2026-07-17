import streamlit as st

from services.conversation import (
    initialize_conversation,
    reset_conversation,
    save_uploaded_image,
    has_uploaded_image,
    is_waiting_for_image
)
from cow_image_classifier.predict import predict_disease

from llm.chatbot import get_chat_response



# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(

    page_title="CowDoc AI",

    page_icon="🐄",

    layout="wide"

)

with st.sidebar:

    st.title("🐄 CowDoc AI")

    st.write("Localized AI Assistant")

    st.write("For Smallholder Cattle Farmers")

    st.divider()

    st.subheader("Current Features")

    st.write("✅ AI Chatbot")
    st.write("✅ Optional Image Upload")
    st.write("✅ Image Classification")
    st.write("🚧 Voice Assistant")
    st.write("✅ Verified RAG")

    st.divider()

    st.info(
"""
### CowDoc AI

✔ Chat-based Diagnosis

✔ Upload Image Anytime

✔ Verified RAG Knowledge

✔ AI Reasoning

✔ Future Voice Assistant
"""
    )

    # ======================================================
    # New Conversation
    # ======================================================

    st.divider()

    if st.button(
        "🗑️ New Conversation",
        width="stretch"
    ):

        # Clear chat history
        st.session_state.messages = []

        # Clear image classifier result
        st.session_state.classifier_result = None

        # Clear candidate diseases
        st.session_state.candidate_diseases = []

        # Clear uploaded image
        st.session_state.uploaded_image = None
        # Reset clinical case memory
        st.session_state.case = {

            "summary": "",

            "important_information": [],

            "timeline": "",

            "treatments_tried": "",

            "hypothesis": "",

            "asked_questions": [],

            "image_evidence": None,

            "urgent": False,

            "status": "collecting_information",

            "clarification_rounds": 0

        }
        # Reset conversation manager
        reset_conversation(
            st.session_state
        )

        st.rerun()

# ==========================================================
# Session State
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "classifier_result" not in st.session_state:

    st.session_state.classifier_result = None

if "candidate_diseases" not in st.session_state:

    st.session_state.candidate_diseases = []

if "uploaded_image" not in st.session_state:

    st.session_state.uploaded_image = None
# ==========================================================
# Clinical Case Memory
# ==========================================================

if "case" not in st.session_state:

    st.session_state.case = {

        # Running understanding of the case
        "summary": "",

        # Important facts gathered so far
        "important_information": [],

        # Timeline of the illness
        "timeline": "",

        # Treatments already attempted
        "treatments_tried": "",

        # Current working hypothesis
        "hypothesis": "",

        # Questions already asked
        "asked_questions": [],

        # Image classifier evidence
        "image_evidence": None,

        # Sticky urgency flag
        "urgent": False,

        # Current status
        "status": "collecting_information",

        # Number of clarification rounds
        "clarification_rounds": 0

    }
# Initialize clarification state
initialize_conversation(
    st.session_state
)   

# ==========================================================
# Main Page
# ==========================================================

st.title("🐄 CowDoc AI")

st.caption(
    "AI Powered Cattle Disease Detection Assistant"
)

st.info(
"""
👋 Welcome to CowDoc AI.

• Describe your cow's symptoms naturally.

• Upload an image whenever you think it may help.

• If you don't have an image,
  we'll continue using symptoms only.
"""
)

st.divider()

# ==========================================================
# Conversation History
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        # -----------------------------
        # Text Message
        # -----------------------------

        if message.get("type", "text") == "text":

            st.markdown(

                message["content"]

            )

        # -----------------------------
        # Uploaded Image
        # -----------------------------

        elif message["type"] == "image":

            st.markdown("📷 Uploaded Image")

            if st.session_state.uploaded_image is not None:

                st.image(

                    st.session_state.uploaded_image,

                    width="stretch"

                )

st.divider()
# ==========================================================
# Image Upload
# ==========================================================

uploaded_image = st.file_uploader(

    "📷 Upload Image (Optional)",

    type=["jpg", "jpeg", "png"],

    key="cow_image"

)

if uploaded_image is not None:

    new_upload = (

        st.session_state.uploaded_image is None

        or

        uploaded_image.name != getattr(
            st.session_state.uploaded_image,
            "name",
            ""
        )

    )

    if new_upload:

        # ------------------------------------------
        # Save Uploaded Image
        # ------------------------------------------

        save_uploaded_image(

            st.session_state,

            uploaded_image

        )

        # ------------------------------------------
        # Store image as conversation item
        # ------------------------------------------

        st.session_state.messages.append(

            {

                "role": "user",

                "type": "image"

            }

        )

        # ------------------------------------------
        # Run Image Classifier
        # ------------------------------------------

        with st.spinner("Analyzing image..."):

            try:

                classifier_result = predict_disease(

                    uploaded_image

                )

                st.session_state.classifier_result = classifier_result

                # ------------------------------------------
                # Store image evidence
                # ------------------------------------------

                st.session_state.case["image_evidence"] = classifier_result

                st.session_state.candidate_diseases = [

                    prediction["disease"]

                    for prediction in classifier_result["predictions"]

                ]

            except Exception as e:

                st.error(

                    f"Image Classification Error\n\n{e}"

                )

                st.stop()

        # ------------------------------------------
        # AI acknowledgement
        # ------------------------------------------

        # ----------------------------------------------------------
        # AI acknowledgement after image upload
        # ----------------------------------------------------------

        has_text_message = any(

            message.get("type") == "text"

            and

            message["role"] == "user"

            for message in st.session_state.messages

        )

        if has_text_message:

            acknowledgement = (
                "📷 Thanks! I've received the image.\n\n"
                "I'll combine it with the symptoms you've described "
                "before making a diagnosis."
            )

        else:

            acknowledgement = (
                "📷 Thanks! I've received the image.\n\n"
                "Could you briefly describe what you've noticed "
                "in your cow? I'll use both your description and "
                "the image together."
            )

        st.session_state.messages.append(

            {

                "role": "assistant",

                "type": "text",

                "content": acknowledgement

            }

        )

        st.rerun()

# ==========================================================
# Waiting For Image
# ==========================================================

if is_waiting_for_image(st.session_state):

    st.info(
"""
The AI recommends uploading an image because it may improve
diagnostic confidence.

Uploading an image is OPTIONAL.

If you don't have one,
simply type **skip** and we'll continue using symptoms only.
"""
    )

st.divider()

# ==========================================================
# Chat Input
# ==========================================================

prompt = st.chat_input(

    "Describe your cow's symptoms..."

)

# ==========================================================
# Process Chat
# ==========================================================

if prompt:

    # ------------------------------------------------------
    # Save User Message
    # ------------------------------------------------------

    st.session_state.messages.append(

        {

            "role": "user",

            "type": "text",

            "content": prompt

        }

    )

    # ------------------------------------------------------
    # Generate AI Response
    # ------------------------------------------------------

    with st.spinner("CowDoc AI is analysing the information..."):

        try:

            response = get_chat_response(

                prompt,

                st.session_state

            )

        except Exception as e:

            response = (

                "Sorry, an unexpected error occurred.\n\n"

                f"{e}"

            )

    # ------------------------------------------------------
    # Save Assistant Message
    # ------------------------------------------------------

    st.session_state.messages.append(

        {

            "role": "assistant",

            "type": "text",

            "content": response

        }

    )

    st.rerun()

