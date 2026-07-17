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
    clear_clarification,
    start_waiting_for_image
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

                    "image_recommended",

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

            "image_reason": {

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

            },

            "updated_summary": {

                "type": "string"

            },

            "new_information": {

                "type": "array",

                "items": {

                    "type": "string"

                }

            },

            "timeline": {

                "type": "string"

            },

            "treatments_tried": {

                "type": "string"

            }

        },

        "required": [

            "status",

            "disease",

            "normalized_query",

            "intent_type",

            "clarifying_question",

            "image_reason",

            "out_of_scope_note",

            "urgent",

            "updated_summary",

            "new_information",

            "timeline",

            "treatments_tried"

        ],

        "additionalProperties": False

    }

}
# ==========================================================
# Build Image Context
# ==========================================================

def build_image_context(classifier_result):
    """
    Convert image classifier output into readable text
    for Stage 1.

    Returns None if no image is available.
    """

    if not classifier_result:
        return None

    lines = []

    top = classifier_result["top_prediction"]

    lines.append("An image has been uploaded.")
    lines.append("")
    lines.append(
        f"Top Prediction: {top['disease']} "
        f"({top['confidence']}%)"
    )
    lines.append("")
    lines.append("Other Candidate Diseases:")

    for prediction in classifier_result["predictions"][1:]:

        lines.append(
            f"- {prediction['disease']} "
            f"({prediction['confidence']}%)"
        )

    return "\n".join(lines)
# ==========================================================
# Build Clinical Case Context
# ==========================================================

def build_case_context(session_state):

    case = session_state.case

    context = f"""
Current Clinical Case

Case Summary:
{case["summary"]}

Important Information:
{chr(10).join("- " + item for item in case["important_information"]) if case["important_information"] else "None"}

Timeline:
{case["timeline"] or "Unknown"}

Treatments Tried:
{case["treatments_tried"] or "None"}

Current Hypothesis:
{case["hypothesis"] or "Unknown"}

Clarification Rounds:
{case["clarification_rounds"]}

Urgent:
{case["urgent"]}

Status:
{case["status"]}
"""

    if case["image_evidence"]:

        top = case["image_evidence"]["top_prediction"]

        context += f"""

Image Evidence

Top Prediction:
{top["disease"]}

Confidence:
{top["confidence"]}%
"""

    else:

        context += "\n\nImage Evidence:\nNone"

    return context

# ==========================================================
# Update Clinical Case
# ==========================================================

def update_case(session_state, stage1_output):

    case = session_state.case

    # Update summary
    summary = stage1_output.get("updated_summary", "").strip()

    if summary:
        case["summary"] = summary

    # Add new important information
    for item in stage1_output.get("new_information", []):

        if item not in case["important_information"]:

            case["important_information"].append(item)

    # Update timeline
    timeline = stage1_output.get("timeline", "").strip()

    if timeline:
        case["timeline"] = timeline

    # Update treatments
    treatment = stage1_output.get("treatments_tried", "").strip()

    if treatment:
        case["treatments_tried"] = treatment

    # Update hypothesis
    disease = stage1_output.get("disease")

    if disease:
        case["hypothesis"] = disease

    # Sticky urgent flag
    if stage1_output.get("urgent"):

        case["urgent"] = True

    # Increase clarification rounds
    if stage1_output.get("status") == "clarification_needed":

        case["clarification_rounds"] += 1
# ==========================================================
# Stage 1
# Understanding Module
# ==========================================================

def understand_farmer_question(
    farmer_question,
    case_context=None,
    image_context=None
):
    """
    Stage 1

    Understand the farmer's question and return
    structured JSON.
    """

    # ------------------------------------------------------
    # Build User Content
    # ------------------------------------------------------

    user_content = ""

    if case_context:

        user_content += f"""
    Current Clinical Case

    {case_context}

    ------------------------------------------------
    """

    user_content += f"""

    Latest Farmer Reply

    {farmer_question}
    """

    if image_context:

        user_content += f"""

    ------------------------------------------------

    Image Information

    {image_context}
    """

    # ------------------------------------------------------
    # Call Stage 1 LLM
    # ------------------------------------------------------

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
                    "content": user_content
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

        print(f"Stage 1 Error: {e}")

        return {

            "status": "clarification_needed",

            "disease": None,

            "normalized_query": "",

            "intent_type": None,

            "clarifying_question":
                "Sorry, I couldn't understand your question. Could you describe your cow's symptoms again?",

            "image_reason": None,

            "out_of_scope_note": None,

            "urgent": False,

            "updated_summary": "",

            "new_information": [],

            "timeline": "",

            "treatments_tried": ""

        }

    # ------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------

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

            "image_reason": None,

            "out_of_scope_note": None,

            "urgent": False,

            "updated_summary": "",

            "new_information": [],

            "timeline": "",

            "treatments_tried": ""

        }

    # ------------------------------------------------------
    # Validate Required Fields
    # ------------------------------------------------------

    required_fields = [

        "status",

        "disease",

        "normalized_query",

        "intent_type",

        "clarifying_question",

        "image_reason",

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

    # ------------------------------------------------------
    # Build Image Context
    # ------------------------------------------------------

    image_context = build_image_context(
        session_state.get("classifier_result")
    )
    case_context = build_case_context(
       session_state
)
    # ------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------

    stage1_output = understand_farmer_question(
        farmer_question=user_question,
        case_context=case_context,
        image_context=image_context
    )
    print("\n========== STAGE 1 OUTPUT ==========")
    print(json.dumps(stage1_output, indent=4))
    print("====================================")
    # ------------------------------------------------------
    # Update Clinical Case Memory
    # ------------------------------------------------------

    update_case(
        session_state,
        stage1_output
    )
    status = stage1_output.get("status")
    # ------------------------------------------------------
    # Memory Recall
    # ------------------------------------------------------

    if stage1_output.get("intent_type") == "memory_recall":

            case = session_state.case

            response = "Here's what you've told me so far:\n\n"

            if case["summary"]:
                response += f"Summary:\n{case['summary']}\n\n"

            if case["important_information"]:
                response += "Confirmed information:\n"

                for item in case["important_information"]:
                    response += f"• {item}\n"

            if case["hypothesis"]:
                response += f"\nCurrent identified disease: {case['hypothesis']}"

            return response
# ------------------------------------------------------
# Image Recommended
# ------------------------------------------------------

    if status == "image_recommended":

        question = stage1_output.get(

            "clarifying_question",

            "Please upload a clear image of the affected area. "
            "If you do not have one, type 'skip'."

        )

        start_waiting_for_image(session_state)

        return question
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
    print("\n========== RAG ==========")
    print("Disease:", disease)
    print("Query:", stage1_output["normalized_query"])
    print("Documents:", len(rag_results["documents"]))
    print("====================================\n")
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