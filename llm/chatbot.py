from openai import OpenAI
from dotenv import load_dotenv
import os

from rag.retrieve import retrieve_documents
from llm.prompt import SYSTEM_PROMPT

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
# ==========================================================
# OpenAI Model
# ==========================================================

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini"
)
# ==========================================================
# Supported Diseases
# ==========================================================

SUPPORTED_DISEASES = {
    "foot and mouth disease": "FMD",
    "fmd": "FMD",

    "lumpy skin disease": "LSD",
    "lsd": "LSD",

    "mastitis": "Mastitis",

    "foot rot": "FootRot",
    "footrot": "FootRot",

    "ringworm": "Ringworm"
}

# ==========================================================
# Symptoms Dictionary
# ==========================================================

SYMPTOMS = {

    "fever": [
        "fever",
        "high temperature",
        "temperature"
    ],

    "loss_of_appetite": [
        "not eating",
        "loss of appetite",
        "reduced appetite"
    ],

    "mouth_ulcers": [
        "mouth ulcer",
        "mouth ulcers",
        "mouth sores",
        "blisters"
    ],

    "drooling": [
        "drooling",
        "salivation",
        "saliva"
    ],

    "skin_nodules": [
        "skin nodules",
        "nodules",
        "raised lumps",
        "lumps"
    ],

    "limping": [
        "limping",
        "lameness"
    ],

    "foot_swelling": [
        "foot swelling",
        "swollen hoof",
        "hoof swelling"
    ],

    "udder_swelling": [
        "udder swelling",
        "swollen udder"
    ],

    "abnormal_milk": [
        "milk clots",
        "flakes in milk",
        "abnormal milk"
    ],

    "hair_loss": [
        "hair loss",
        "bald patches"
    ],

    "round_skin_patches": [
        "round patches",
        "ring shaped patches",
        "grey patches"
    ]
}
# ==========================================================
# Detect Disease Name
# ==========================================================

def detect_disease(question):

    question = question.lower()

    for keyword, disease in SUPPORTED_DISEASES.items():

        if keyword in question:
            return disease

    return None


# ==========================================================
# Detect Symptoms
# ==========================================================

def detect_symptoms(question):

    question = question.lower()

    detected = []

    for symptom, keywords in SYMPTOMS.items():

        for keyword in keywords:

            if keyword in question:

                detected.append(symptom)
                break

    return detected


# ==========================================================
# Check if Question is Vague
# ==========================================================

def is_vague_question(question):

    question = question.lower().strip()

    vague_inputs = [

        "my cow is sick",
        "my cow is ill",
        "cow is sick",
        "cow is ill",
        "help",
        "please help",
        "my animal is sick",
        "my animal is ill"

    ]

    if question in vague_inputs:
        return True

    disease = detect_disease(question)

    symptoms = detect_symptoms(question)

    if disease:
        return False

    if len(symptoms) == 0:
        return True

    return False


# ==========================================================
# Check Unsupported Disease
# ==========================================================

def unsupported_disease(question):

    question = question.lower()

    unsupported = [

        "bird flu",
        "anthrax",
        "rabies",
        "brucellosis",
        "black quarter",
        "blackleg",
        "tuberculosis"

    ]

    for disease in unsupported:

        if disease in question:

            return disease

    return None
# ==========================================================
# Need Follow-up Questions?
# ==========================================================

def need_followup(symptoms):

    symptom_set = set(symptoms)

    # Strong indicators for a specific disease
    if {"mouth_ulcers", "drooling"} <= symptom_set:
        return False          # Likely FMD

    if {"skin_nodules", "fever"} <= symptom_set:
        return False          # Likely LSD

    if {"udder_swelling", "abnormal_milk"} <= symptom_set:
        return False          # Likely Mastitis

    if {"limping", "foot_swelling"} <= symptom_set:
        return False          # Likely Foot Rot

    if {"hair_loss", "round_skin_patches"} <= symptom_set:
        return False          # Likely Ringworm

    # Not enough information
    if len(symptom_set) <= 2:
        return True

    return False


# ==========================================================
# Generate Follow-up Questions
# ==========================================================

def get_followup_questions(symptoms):

    symptom_set = set(symptoms)

    questions = []

    if "fever" in symptom_set:

        if "mouth_ulcers" not in symptom_set:
            questions.append(
                "Does your cow have mouth ulcers or excessive drooling?"
            )

        if "skin_nodules" not in symptom_set:
            questions.append(
                "Does your cow have skin nodules or raised lumps?"
            )

        if "limping" not in symptom_set:
            questions.append(
                "Is your cow limping or having difficulty walking?"
            )

        if "udder_swelling" not in symptom_set:
            questions.append(
                "Is the udder swollen or is the milk abnormal?"
            )

        if "hair_loss" not in symptom_set:
            questions.append(
                "Are there circular skin patches or hair loss?"
            )

    if "limping" in symptom_set:

        if "foot_swelling" not in symptom_set:

            questions.append(
                "Is there swelling between the hooves or a foul smell from the foot?"
            )

    if "udder_swelling" in symptom_set:

        if "abnormal_milk" not in symptom_set:

            questions.append(
                "Does the milk contain clots or flakes?"
            )

    return questions


# ==========================================================
# Build Context for LLM
# ==========================================================

def build_context(results):

    context = ""

    for doc in results["documents"]:

        context += doc + "\n\n"

    return context
# ==========================================================
# Source Mapping
# ==========================================================

SOURCE_MAPPING = {

    "Foot-and-Mouth Disease in Cattle .pdf": {
        "organization": "National Dairy Development Board (NDDB)"
    },

    "Guidelines for prevention of Lumpy Skin Disease.pdf": {
        "organization": "National Dairy Development Board (NDDB)"
    },

    "Lumpy Skin disease in cattle.pdf": {
        "organization": "National Dairy Development Board (NDDB)"
    },

    "Common animal diseases and their management (1).pdf": {
        "organization": "Vikaspedia"
    },

    "Incidences of mastitis among bovines and its management .pdf": {
        "organization": "Indian Veterinary Research Institute (IVRI)"
    },

    "Footrot Disease Infographic FINAL.pdf": {
        "organization": "University Veterinary Publication"
    }

}

# ==========================================================
# Build Verified Sources
# ==========================================================

def build_sources(results):

    source_list = []

    for meta in results["sources"]:

        pdf = meta["source"]

        info = SOURCE_MAPPING.get(
            pdf,
            {
                "organization": "Verified Veterinary Source"
            }
        )

        formatted_source = (
            f"Organization: {info['organization']}\n"
            f"Document: {pdf}"
        )

        if formatted_source not in source_list:
            source_list.append(formatted_source)

    return "\n\n".join(source_list)


# ==========================================================
# Generate RAG Response
# ==========================================================

def generate_rag_response(user_question):

    # Retrieve documents
    results = retrieve_documents(user_question)

    # Build context
    context = build_context(results)

    # Build verified sources
    verified_sources = build_sources(results)

    # Create prompt for LLM
    user_prompt = f"""
User Question:

{user_question}


Retrieved Knowledge:

{context}


Verified Knowledge Sources:

{verified_sources}
"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        temperature=0.2,

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ]

    )

    return response.choices[0].message.content
# ==========================================================
# Main Chat Function
# ==========================================================

def get_chat_response(user_question):

    # Remove extra spaces
    user_question = user_question.strip()

    # ------------------------------------------------------
    # Check Unsupported Disease
    # ------------------------------------------------------

    disease = unsupported_disease(user_question)

    if disease:

        return f"""
❌ Sorry.

CowDoc AI currently does not contain verified information about **{disease.title()}**.

Currently supported diseases are:

• Foot-and-Mouth Disease (FMD)

• Lumpy Skin Disease (LSD)

• Mastitis

• Foot Rot

• Ringworm

Please consult a veterinarian for diseases outside this knowledge base.
"""

    # ------------------------------------------------------
    # Check Vague Question
    # ------------------------------------------------------

    if is_vague_question(user_question):

        return """
I need a little more information before I can help.

Please describe your cow's symptoms.

For example:

• Fever
• Mouth ulcers
• Excessive drooling
• Skin nodules
• Swollen udder
• Milk clots
• Limping
• Hair loss

The more symptoms you describe, the more accurate I can be.
"""

    # ------------------------------------------------------
    # Detect Disease
    # ------------------------------------------------------

    detected_disease = detect_disease(user_question)

    # ------------------------------------------------------
    # Detect Symptoms
    # ------------------------------------------------------

    detected_symptoms = detect_symptoms(user_question)

    # ------------------------------------------------------
    # Need Follow-up Questions?
    # ------------------------------------------------------

    if detected_disease is None:

        if need_followup(detected_symptoms):

            questions = get_followup_questions(detected_symptoms)

            response = (
                "Based on the symptoms you described, I need a little more "
                "information before identifying the most likely disease.\n\n"
            )

            for i, question in enumerate(questions, start=1):

                response += f"{i}. {question}\n"

            response += (
                "\nPlease answer these questions so I can provide a better response."
            )

            return response

    # ------------------------------------------------------
    # Retrieve Information using RAG
    # ------------------------------------------------------

    return generate_rag_response(user_question)