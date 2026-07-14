print("Loaded NEW chatbot.py")
from openai import OpenAI
from dotenv import load_dotenv

import os
import json

from config import (
    STAGE1_MODEL,
    STAGE2_MODEL,
    NO_RAG_RESPONSE,
    SYSTEM_ERROR_MESSAGE
)

from rag.retrieve import retrieve_documents

from llm.stage1_prompt import get_stage1_prompt
from llm.stage2_prompt import get_stage2_prompt

from services.conversation import (
    start_clarification,
    is_waiting_for_clarification,
    build_clarification_input,
    clear_clarification
)

# ==========================================================
# Load Environment
# ==========================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ==========================================================
# Stage 1 JSON Schema
# ==========================================================

STAGE1_SCHEMA = {

    "name": "cowdoctor_understanding",

    "strict": True,

    "schema": {

        "type": "object",

        "properties": {

            "status": {

                "type": "string",

                "enum": [

                    "identified",

                    "clarification_needed",

                    "out_of_scope"

                ]

            },

            "disease": {

                "type": [

                    "string",

                    "null"

                ]

            },

            "normalized_query": {

                "type": "string"

            },

            "intent_type": {

                "type": [

                    "string",

                    "null"

                ]

            },

            "clarifying_question": {

                "type": [

                    "string",

                    "null"

                ]

            },

            "out_of_scope_note": {

                "type": [

                    "string",

                    "null"

                ]

            },

            "urgent": {

                "type": "boolean"

            }

        },

        "required": [

            "status",

            "disease",

            "normalized_query",

            "intent_type",

            "clarifying_question",

            "out_of_scope_note",

            "urgent"

        ],

        "additionalProperties": False

    }

}
# ==========================================================
# Stage 1
# Understanding Module
# ==========================================================

def understand_farmer_question(user_question):
    """
    Stage 1

    Understand the farmer's question and return
    structured JSON.
    """

    try:

        response = client.responses.create(

            model=STAGE1_MODEL,

            input=[

                {
                    "role": "system",

                    "content": get_stage1_prompt()

                },

                {
                    "role": "user",

                    "content": user_question

                }

            ],

            text={

                "format": {

                    "type": "json_schema",

                    "name": STAGE1_SCHEMA["name"],

                    "strict": True,

                    "schema": STAGE1_SCHEMA["schema"]

                }

            }

        )

    except Exception as e:

        print(f"Stage 1 rror: {e}")

        return {

            "status": "clarification_needed",

            "disease": None,

            "normalized_query": "",

            "intent_type": None,

            "clarifying_question":
                "Sorry, I couldn't understand your question. Could you describe your cow's symptoms again?",

            "out_of_scope_note": None,

            "urgent": False

        }

    try:

        result = json.loads(
            response.output_text
        )

    except json.JSONDecodeError:

        return {

            "status": "clarification_needed",

            "disease": None,

            "normalized_query": "",

            "intent_type": None,

            "clarifying_question":
                "Sorry, I couldn't understand your question. Could you describe your cow's symptoms again?",

            "out_of_scope_note": None,

            "urgent": False

        }

    required_fields = [

    "status",

    "disease",

    "normalized_query",

    "intent_type",

    "clarifying_question",

    "out_of_scope_note",

    "urgent"

    ]

    for field in required_fields:

     if field not in result:

        raise ValueError(
            f"Missing field: {field}"
        )

    return result
# ==========================================================
# Build RAG Context
# ==========================================================

def build_context(rag_results):
    """
    Convert retrieved documents into a single
    context string for Stage 2.
    """

    context = ""

    for document in rag_results["documents"]:

        context += document

        context += "\n\n"

    return context


# ==========================================================
# Build Source List
# ==========================================================

def build_sources(rag_results):
    """
    Build a readable list of sources.
    """

    sources = []

    for metadata in rag_results["sources"]:

        source = metadata.get(
            "source",
            "Unknown Source"
        )

        if source not in sources:

            sources.append(source)

    if len(sources) == 0:

        return "No source available."

    output = "Source:\n"

    for source in sources:

        output += f"- {source}\n"

    return output.strip()


# ==========================================================
# Stage 2
# Answering Module
# ==========================================================

def generate_final_answer(

    user_question,

    stage1_output,

    rag_results

):
    """
    Generate the final answer using ONLY
    verified RAG knowledge.
    """

    context = build_context(
        rag_results
    )

    sources = build_sources(
        rag_results
    )

    user_prompt = f"""
Farmer Question

{user_question}


Stage 1 Output

{json.dumps(stage1_output, indent=2)}


Retrieved Knowledge

{context}


Verified Sources

{sources}
"""

    try:

        response = client.responses.create(

            model=STAGE2_MODEL,

            input=[

                {
                    "role": "system",

                    "content": get_stage2_prompt()

                },

                {
                    "role": "user",

                    "content": user_prompt

                }

            ]

        )

        return response.output_text

    except Exception as e:

        print(f"Stage 2 Error: {e}")

        return SYSTEM_ERROR_MESSAGE
# ==========================================================
# Main Chat Function
# ==========================================================

def get_chat_response(
    user_question,
    session_state
):
    """
    Complete CowDoctor pipeline.

    User
        ↓
    Stage 1
        ↓
    Clarification / Out of Scope / Identified
        ↓
    RAG
        ↓
    Stage 2
        ↓
    Final Answer
    """

    # ------------------------------------------------------
    # Handle Clarification Response
    # ------------------------------------------------------

    if is_waiting_for_clarification(session_state):

        user_question = build_clarification_input(

            session_state,

            user_question

        )

        clear_clarification(session_state)

    # ------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------

    stage1_output = understand_farmer_question(
        user_question
    )

    status = stage1_output.get("status")

    # ------------------------------------------------------
    # Clarification Needed
    # ------------------------------------------------------

    if status == "clarification_needed":

        question = stage1_output.get(
            "clarifying_question",
            "Could you please describe the symptoms in more detail?"
        )

        start_clarification(

            session_state,

            user_question,

            question

        )

        return question

    # ------------------------------------------------------
    # Out of Scope
    # ------------------------------------------------------

    if status == "out_of_scope":

        return stage1_output.get(

            "out_of_scope_note",

            "Sorry, CowDoctor currently supports only the diseases in its knowledge base."

        )

    # ------------------------------------------------------
    # Disease Identified
    # ------------------------------------------------------

    disease = stage1_output.get("disease")

    if disease is None:

        return (
            "I couldn't determine which disease you're asking about. "
            "Could you provide a little more detail?"
        )

    # ------------------------------------------------------
    # Retrieve Documents
    # ------------------------------------------------------

    rag_results = retrieve_documents(

        disease=disease,

        question=stage1_output["normalized_query"]

    )

    # ------------------------------------------------------
    # Retrieval Check
    # ------------------------------------------------------

    if not rag_results["documents"]:

       return NO_RAG_RESPONSE

    # ------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------

    final_answer = generate_final_answer(

        user_question,

        stage1_output,

        rag_results

    )

    return final_answer


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    class DummySession(dict):
        pass

    session = DummySession()

    while True:

        question = input("\nFarmer: ")

        if question.lower() == "exit":
            break

        answer = get_chat_response(

            question,

            session

        )

        print("\nCowDoctor:\n")

        print(answer)