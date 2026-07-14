import streamlit as st
from llm.chatbot import get_chat_response

from services.conversation import (
    initialize_conversation
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="CowDoc AI",
    page_icon="🐄",
    layout="wide"
)

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("🐄 CowDoc AI")

    st.write("Localized AI Assistant")

    st.write("For Smallholder Cattle Farmers")

    st.divider()

    st.subheader("Current Features")

    st.write("✅ AI Chatbot")

    st.write("🚧 Image Diagnosis")

    st.write("🚧 Voice Assistant")

    st.write("✅ RAG Knowledge Base")

    st.divider()

    st.info(
"""
### Project Status

✅ AI Chatbot

🚧 Image Classification

✅ RAG Knowledge Base

🚧 Voice Assistant
"""
    )

# ==========================================================
# Main Page
# ==========================================================

st.title("🐄 CowDoc AI")

st.caption(
    "AI Powered Cattle Disease Detection Assistant"
)

tab1, tab2, tab3 = st.tabs(
    [
        "📝 Text Diagnosis",
        "📷 Image Diagnosis",
        "🎤 Voice Assistant"
    ]
)

# ==========================================================
# Session State
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

initialize_conversation(
    st.session_state
)

# ==========================================================
# TAB 1 : TEXT DIAGNOSIS
# ==========================================================

with tab1:

    st.subheader("Describe Your Cow's Symptoms")

    st.write("""
You can describe your cow's symptoms in simple English.

### Example Questions

- My cow has fever.
- My cow has mouth ulcers.
- My cow has skin nodules.
- What is Foot and Mouth Disease?
- How can I prevent Mastitis?
""")

    st.divider()

    # ------------------------------------------------------
    # Display Previous Messages
    # ------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # ------------------------------------------------------
    # Chat Input
    # ------------------------------------------------------

    prompt = st.chat_input(
        "Describe your cow's symptoms..."
    )

    if prompt and prompt.strip():

        # User Message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):

            st.markdown(prompt)

        # Assistant Response

        with st.chat_message("assistant"):

            with st.spinner(
                "Understanding your question..."
            ):

                prompt = prompt.strip()

                answer = get_chat_response(

                    prompt,

                    st.session_state

                )

                st.markdown(answer)

        # Save Assistant Message

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# ==========================================================
# TAB 2 : IMAGE DIAGNOSIS
# ==========================================================

with tab2:

    st.subheader("📷 Cow Image Diagnosis")

    st.write("""
Upload a clear image of the affected area of your cow.

Supported image formats:

- JPG
- JPEG
- PNG
""")

    uploaded_file = st.file_uploader(
        "Choose a Cow Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Cow Image",
            use_container_width=True
        )

        st.success("✅ Image uploaded successfully.")

        st.info("""
Image Classification Workflow

Image
   ↓
EfficientNet-B0
   ↓
Disease Prediction
   ↓
CowDoc AI
   ↓
Stage 2
   ↓
Farmer-Friendly Answer
""")

    else:

        st.info(
            "Please upload an image to continue."
        )

# ==========================================================
# TAB 3 : VOICE ASSISTANT
# ==========================================================

with tab3:

    st.subheader("🎤 Voice Assistant")

    st.write("""
Future Features

• Malayalam Voice Support

• English Voice Support

• Speech-to-Text

• Text-to-Speech

• Voice Conversation with CowDoc AI
""")

    st.button(
        "🎤 Start Voice Assistant",
        disabled=True
    )

    st.info(
        "Voice Assistant will be implemented in the next phase."
    )

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "🐄 CowDoc AI | AI Powered Chatbot for Cattle Disease Detection"
)