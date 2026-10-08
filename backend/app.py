from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import re
import json


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# GROQ
# ============================================================

API_KEY = os.getenv("GROQ_API_KEY", "").strip()

client = None

if API_KEY:
    client = Groq(api_key=API_KEY)

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# FILES
# ============================================================

UPLOAD_FOLDER = "uploads"

CHAT_HISTORY_FILE = "chat_history.json"
LEARNER_MODEL_FILE = "learner_model.json"
QUESTION_HISTORY_FILE = "question_history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# MEMORY
# ============================================================

knowledge_base = []


# ============================================================
# LEARNER MODEL
# ============================================================

def load_learner_model():

    if not os.path.exists(LEARNER_MODEL_FILE):
        return {}

    try:

        with open(
            LEARNER_MODEL_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, dict):
                return data

            return {}

    except Exception as error:

        print(
            f"Could not load learner model: {error}"
        )

        return {}


learner_model = load_learner_model()


def save_learner_model():

    try:

        with open(
            LEARNER_MODEL_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                learner_model,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        print(
            f"Could not save learner model: {error}"
        )


# ============================================================
# QUESTION HISTORY
# ============================================================

def load_question_history():

    if not os.path.exists(QUESTION_HISTORY_FILE):
        return []

    try:

        with open(
            QUESTION_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except Exception as error:

        print(
            f"Could not load question history: {error}"
        )

        return []


question_history = load_question_history()


def save_question_history():

    try:

        with open(
            QUESTION_HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                question_history[-500:],
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        print(
            f"Could not save question history: {error}"
        )


# ============================================================
# CHAT HISTORY
# ============================================================

def load_chat_history():

    if not os.path.exists(CHAT_HISTORY_FILE):
        return []

    try:

        with open(
            CHAT_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except Exception as error:

        print(
            f"Could not load chat history: {error}"
        )

        return []


chat_history = load_chat_history()


def save_chat_history():

    try:

        with open(
            CHAT_HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                chat_history[-100:],
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        print(
            f"Could not save chat history: {error}"
        )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def normalize_question(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PDF CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=1200
):

    words = text.split()

    chunks = []

    current_chunk = []

    current_length = 0

    for word in words:

        current_chunk.append(word)

        current_length += len(word) + 1

        if current_length >= chunk_size:

            chunks.append(
                " ".join(current_chunk)
            )

            current_chunk = []

            current_length = 0

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


# ============================================================
# INDEX PDF
# ============================================================

def index_pdf(
    file_path,
    filename
):

    reader = PdfReader(file_path)

    added_chunks = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        raw_text = page.extract_text() or ""

        text = clean_text(raw_text)

        if not text:
            continue

        chunks = create_chunks(text)

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            knowledge_base.append({

                "source": filename,

                "page": page_number,

                "chunk": chunk_number,

                "text": chunk

            })

            added_chunks += 1

    return added_chunks


# ============================================================
# LOAD EXISTING PDFs
# ============================================================

def load_existing_pdfs():

    knowledge_base.clear()

    files = os.listdir(
        UPLOAD_FOLDER
    )

    pdf_files = [

        file

        for file in files

        if file.lower().endswith(".pdf")

    ]

    print()
    print("==============================")
    print("LOADING SAVED PDF FILES")
    print("==============================")

    for filename in pdf_files:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        try:

            chunks = index_pdf(
                file_path,
                filename
            )

            print(
                f"Loaded: {filename} | {chunks} chunks"
            )

        except Exception as error:

            print(
                f"Could not load {filename}: {error}"
            )

    print(
        f"Total knowledge chunks: {len(knowledge_base)}"
    )

    print("==============================")
    print()


# ============================================================
# RAG SEARCH
# ============================================================

def retrieve_context(
    question,
    top_k=5
):

    if not knowledge_base:
        return []

    documents = [

        item["text"]

        for item in knowledge_base

    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:

        document_vectors = vectorizer.fit_transform(
            documents
        )

        question_vector = vectorizer.transform(
            [question]
        )

    except ValueError:

        return []

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    ranked_indexes = similarities.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:

        score = float(
            similarities[index]
        )

        if score <= 0:
            continue

        item = knowledge_base[index].copy()

        item["score"] = round(
            score,
            4
        )

        results.append(item)

    return results


# ============================================================
# FIND TOPIC MASTERY
# ============================================================

def get_topic_mastery(topic):

    if topic not in learner_model:
        return 0

    return learner_model[topic].get(
        "mastery",
        0
    )


# ============================================================
# GET WEAK TOPICS
# ============================================================

def find_weak_topics():

    weak_topics = []

    for topic, data in learner_model.items():

        mastery = data.get(
            "mastery",
            0
        )

        if mastery < 60:

            weak_topics.append({

                "topic": topic,

                "mastery": mastery,

                "attempts": data.get(
                    "attempts",
                    0
                ),

                "correct": data.get(
                    "correct",
                    0
                ),

                "total": data.get(
                    "total",
                    0
                )

            })

    weak_topics.sort(
        key=lambda item: item["mastery"]
    )

    return weak_topics


# ============================================================
# CHOOSE ADAPTIVE DIFFICULTY
# ============================================================

def choose_adaptive_difficulty(mastery):

    if mastery < 40:

        return "Beginner"

    elif mastery < 70:

        return "Intermediate"

    else:

        return "Advanced"


# ============================================================
# QUESTION HISTORY CHECK
# ============================================================

def get_previous_questions(topic):

    normalized_topic = topic.lower().strip()

    previous = []

    for item in question_history:

        if item.get(
            "topic",
            ""
        ).lower().strip() == normalized_topic:

            previous.append(
                item.get(
                    "question",
                    ""
                )
            )

    return previous


# ============================================================
# SAVE GENERATED QUESTIONS
# ============================================================

def save_generated_questions(
    topic,
    questions
):

    for question in questions:

        question_text = question.get(
            "question",
            ""
        ).strip()

        if not question_text:
            continue

        normalized = normalize_question(
            question_text
        )

        already_exists = False

        for item in question_history:

            old_question = normalize_question(
                item.get(
                    "question",
                    ""
                )
            )

            if normalized == old_question:

                already_exists = True
                break

        if not already_exists:

            question_history.append({

                "topic": topic,

                "question": question_text,

                "difficulty": question.get(
                    "difficulty",
                    "Intermediate"
                ),

                "source": question.get(
                    "source",
                    ""
                ),

                "page": question.get(
                    "page",
                    0
                )

            })

    save_question_history()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return "EduNova AI Backend is Running"


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    saved_files = [

        file

        for file in os.listdir(
            UPLOAD_FOLDER
        )

        if file.lower().endswith(".pdf")

    ]

    return jsonify({

        "status": "ready",

        "groq_configured": bool(
            API_KEY
        ),

        "model": GROQ_MODEL,

        "knowledge_chunks": len(
            knowledge_base
        ),

        "saved_pdfs": saved_files,

        "chat_count": len(
            chat_history
        ),

        "learner_topics": len(
            learner_model
        ),

        "saved_questions": len(
            question_history
        )

    })


# ============================================================
# ASK AI
# ============================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    data = request.get_json() or {}

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return jsonify({

            "answer":
            "Please enter a question."

        }), 400

    if client is None:

        return jsonify({

            "answer":
            "Groq API key is not configured."

        }), 500

    retrieved = retrieve_context(
        question,
        top_k=5
    )

    if retrieved:

        context_parts = []

        for item in retrieved:

            context_parts.append(
                f"""
SOURCE:
{item['source']}

PAGE:
{item['page']}

CONTENT:
{item['text']}
"""
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are EduNova AI,
a personalized learning tutor.

Use the uploaded course material
when it is relevant.

If the material does not contain
enough information, you may use
general knowledge.

Explain clearly for a college student.

Do not invent information.

If uploaded material is used,
include a Sources section with
filename and page number.

UPLOADED MATERIAL:

{context}

STUDENT QUESTION:

{question}
"""

        grounded = True

    else:

        prompt = f"""
You are EduNova AI,
a helpful personalized learning tutor.

There is no relevant uploaded material.

Answer using general knowledge.

Explain simply for a college student.

Student question:

{question}
"""

        grounded = False

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content":
                    "You are EduNova AI, "
                    "a helpful personalized "
                    "learning tutor."
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.2

        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        sources = []

        for item in retrieved:

            source = {

                "source":
                item["source"],

                "page":
                item["page"],

                "chunk":
                item["chunk"],

                "score":
                item["score"]

            }

            if source not in sources:

                sources.append(source)

        chat_history.append({

            "question":
            question,

            "answer":
            answer,

            "grounded":
            grounded,

            "sources":
            sources

        })

        save_chat_history()

        return jsonify({

            "answer":
            answer,

            "provider":
            "groq",

            "model":
            GROQ_MODEL,

            "grounded":
            grounded,

            "sources":
            sources

        })

    except Exception as error:

        print(
            "\nGROQ ERROR:",
            repr(error)
        )

        return jsonify({

            "answer":
            str(error),

            "provider":
            "groq",

            "model":
            GROQ_MODEL,

            "grounded":
            False,

            "sources":
            []

        }), 500


# ============================================================
# UPLOAD PDF
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "file" not in request.files:

        return jsonify({

            "success": False,

            "error":
            "No file uploaded"

        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({

            "success": False,

            "error":
            "No file selected"

        }), 400

    filename = file.filename

    if not filename.lower().endswith(".pdf"):

        return jsonify({

            "success": False,

            "error":
            "Please upload a PDF file."

        }), 400

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:

        file.save(file_path)

        global knowledge_base

        knowledge_base = [

            item

            for item in knowledge_base

            if item["source"] != filename

        ]

        chunks_added = index_pdf(
            file_path,
            filename
        )

        return jsonify({

            "success": True,

            "message":
            "PDF indexed successfully",

            "source":
            filename,

            "chunks":
            chunks_added,

            "total_knowledge_chunks":
            len(knowledge_base)

        }), 200

    except Exception as error:

        print(
            "\nPDF ERROR:",
            repr(error)
        )

        return jsonify({

            "success": False,

            "error":
            "Could not read the PDF.",

            "details":
            str(error)

        }), 500


# ============================================================
# CHAT HISTORY
# ============================================================

@app.route(
    "/history",
    methods=["GET"]
)
def history():

    return jsonify({

        "history":
        chat_history

    })


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(
    topic,
    count,
    difficulty=None,
    adaptive=False
):

    retrieved = retrieve_context(
        topic,
        top_k=8
    )

    if not retrieved:

        return None, "Topic not found in uploaded material."

    context_parts = []

    for item in retrieved:

        context_parts.append(
            f"""
SOURCE: {item['source']}
PAGE: {item['page']}
CONTENT: {item['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    previous_questions = get_previous_questions(
        topic
    )

    previous_text = "\n".join(
        previous_questions[-30:]
    )

    if not previous_text:

        previous_text = "No previous questions."

    difficulty_instruction = ""

    if difficulty:

        difficulty_instruction = f"""
Generate questions primarily at this difficulty:

{difficulty}
"""

    if adaptive:

        adaptive_instruction = """
This is an adaptive quiz.

Adjust the difficulty according to
the learner's mastery.

The requested difficulty has already
been calculated by EduNova AI.
"""

    else:

        adaptive_instruction = ""

    prompt = f"""
You are EduNova AI,
an adaptive learning assessment generator.

Create {count} multiple-choice questions.

TOPIC:
{topic}

{difficulty_instruction}

{adaptive_instruction}

Use ONLY the uploaded course material.

Each question must contain:

question
options
correct_answer
explanation
difficulty
source
page

Difficulty must be one of:

Beginner
Intermediate
Advanced

IMPORTANT:

Do not repeat or closely rephrase
previously asked questions.

Previous questions:

{previous_text}

Return ONLY valid JSON.

Format:

{{
    "topic": "{topic}",
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "correct_answer": "Option A",
            "explanation": "Short explanation",
            "difficulty": "Beginner",
            "source": "filename.pdf",
            "page": 1
        }}
    ]
}}

UPLOADED COURSE MATERIAL:

{context}
"""

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content":
                    "You generate source-grounded "
                    "educational quizzes."
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.2

        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        answer = re.sub(
            r"^```json\s*",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"^```\s*",
            "",
            answer
        )

        answer = re.sub(
            r"\s*```$",
            "",
            answer
        )

        quiz_data = json.loads(
            answer
        )

        questions = quiz_data.get(
            "questions",
            []
        )

        if not isinstance(
            questions,
            list
        ):

            return None, "Invalid quiz format."

        save_generated_questions(
            topic,
            questions
        )

        return quiz_data, None

    except json.JSONDecodeError:

        return None, "AI returned invalid quiz JSON."

    except Exception as error:

        print(
            "\nQUIZ ERROR:",
            repr(error)
        )

        return None, str(error)


# ============================================================
# NORMAL QUIZ
# ============================================================

@app.route(
    "/quiz",
    methods=["POST"]
)
def quiz():

    data = request.get_json() or {}

    topic = data.get(
        "topic",
        ""
    ).strip()

    try:

        count = int(
            data.get(
                "count",
                5
            )
        )

    except:

        count = 5

    count = max(
        1,
        min(count, 10)
    )

    if not topic:

        return jsonify({

            "success": False,

            "error":
            "Please enter a topic."

        }), 400

    if client is None:

        return jsonify({

            "success": False,

            "error":
            "Groq API key is not configured."

        }), 500

    quiz_data, error = generate_quiz(
        topic,
        count
    )

    if error:

        return jsonify({

            "success": False,

            "error":
            error

        }), 500

    return jsonify({

        "success": True,

        "quiz":
        quiz_data,

        "grounded":
        True

    })


# ============================================================
# LEARNER MODEL UPDATE
# ============================================================

@app.route(
    "/learner/update",
    methods=["POST"]
)
def update_learner():

    data = request.get_json() or {}

    topic = data.get(
        "topic",
        ""
    ).strip()

    try:

        score = int(
            data.get(
                "score",
                0
            )
        )

        total = int(
            data.get(
                "total",
                0
            )
        )

    except:

        return jsonify({

            "success": False,

            "message":
            "Score and total must be numbers."

        }), 400

    if not topic:

        return jsonify({

            "success": False,

            "message":
            "Topic is required."

        }), 400

    if total <= 0:

        return jsonify({

            "success": False,

            "message":
            "Total questions must be greater than 0."

        }), 400

    if score < 0 or score > total:

        return jsonify({

            "success": False,

            "message":
            "Invalid score."

        }), 400

    if topic not in learner_model:

        learner_model[topic] = {

            "attempts": 0,

            "correct": 0,

            "total": 0,

            "mastery": 0

        }

    learner_model[topic]["attempts"] += 1

    learner_model[topic]["correct"] += score

    learner_model[topic]["total"] += total

    mastery = (

        learner_model[topic]["correct"]

        /

        learner_model[topic]["total"]

    ) * 100

    learner_model[topic]["mastery"] = round(
        mastery
    )

    save_learner_model()

    return jsonify({

        "success": True,

        "topic":
        topic,

        "mastery":
        learner_model[topic]["mastery"],

        "attempts":
        learner_model[topic]["attempts"],

        "correct":
        learner_model[topic]["correct"],

        "total":
        learner_model[topic]["total"]

    })


# ============================================================
# LEARNER PROGRESS
# ============================================================

@app.route(
    "/learner/progress",
    methods=["GET"]
)
def learner_progress():

    return jsonify({

        "success": True,

        "learner_model":
        learner_model

    })


# ============================================================
# WEAK TOPICS
# ============================================================

@app.route(
    "/learner/weak-topics",
    methods=["GET"]
)
def get_weak_topics():

    weak_topics = find_weak_topics()

    return jsonify({

        "success": True,

        "weak_topics":
        weak_topics

    })


# ============================================================
# ADAPTIVE QUIZ
# ============================================================

@app.route(
    "/adaptive-quiz",
    methods=["POST"]
)
def adaptive_quiz():

    data = request.get_json() or {}

    topic = data.get(
        "topic",
        ""
    ).strip()

    try:

        count = int(
            data.get(
                "count",
                5
            )
        )

    except:

        count = 5

    count = max(
        1,
        min(count, 10)
    )

    if not topic:

        return jsonify({

            "success": False,

            "error":
            "Please enter a topic."

        }), 400

    if client is None:

        return jsonify({

            "success": False,

            "error":
            "Groq API key is not configured."

        }), 500

    mastery = get_topic_mastery(
        topic
    )

    difficulty = choose_adaptive_difficulty(
        mastery
    )

    quiz_data, error = generate_quiz(

        topic,

        count,

        difficulty,

        adaptive=True

    )

    if error:

        return jsonify({

            "success": False,

            "error":
            error

        }), 500

    return jsonify({

        "success": True,

        "adaptive": True,

        "topic":
        topic,

        "mastery":
        mastery,

        "selected_difficulty":
        difficulty,

        "quiz":
        quiz_data

    })


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

@app.route(
    "/learner/recommendations",
    methods=["GET"]
)
def learner_recommendations():

    weak_topics = find_weak_topics()

    recommendations = []

    for item in weak_topics:

        mastery = item["mastery"]

        if mastery < 40:

            action = "Start with basic concepts and examples."

        elif mastery < 60:

            action = "Revise the topic and take an adaptive quiz."

        else:

            action = "Practice more questions to improve mastery."

        recommendations.append({

            "topic":
            item["topic"],

            "mastery":
            mastery,

            "recommendation":
            action

        })

    if not recommendations:

        recommendations.append({

            "topic":
            "All current topics",

            "mastery":
            100,

            "recommendation":
            "Keep practicing and try advanced questions."

        })

    return jsonify({

        "success": True,

        "recommendations":
        recommendations

    })


# ============================================================
# QUESTION HISTORY
# ============================================================

@app.route(
    "/learner/questions",
    methods=["GET"]
)
def learner_questions():

    return jsonify({

        "success": True,

        "questions":
        question_history

    })


# ============================================================
# DIAGNOSTIC QUIZ
# ============================================================

@app.route(
    "/diagnostic-quiz",
    methods=["POST"]
)
def diagnostic_quiz():

    data = request.get_json() or {}

    try:

        count = int(
            data.get(
                "count",
                5
            )
        )

    except:

        count = 5

    count = max(
        1,
        min(count, 10)
    )

    if client is None:

        return jsonify({

            "success": False,

            "error":
            "Groq API key is not configured."

        }), 500

    if not knowledge_base:

        return jsonify({

            "success": False,

            "error":
            "No learning material is available."

        }), 404

    context_parts = []

    selected_chunks = knowledge_base[:10]

    for item in selected_chunks:

        context_parts.append(
            f"""
SOURCE: {item['source']}
PAGE: {item['page']}
CONTENT: {item['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are EduNova AI.

Create {count} diagnostic multiple-choice
questions for a new learner.

The questions should cover different
concepts from the uploaded material.

Do not assume that the learner has
previous knowledge.

Each question must contain:

question
options
correct_answer
explanation
difficulty
topic
source
page

Difficulty must be:

Beginner
Intermediate
Advanced

Return ONLY valid JSON.

Format:

{{
    "questions": [
        {{
            "question": "Question",
            "options": [
                "A",
                "B",
                "C",
                "D"
            ],
            "correct_answer": "A",
            "explanation": "Explanation",
            "difficulty": "Beginner",
            "topic": "Topic",
            "source": "file.pdf",
            "page": 1
        }}
    ]
}}

UPLOADED MATERIAL:

{context}
"""

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content":
                    "You create source-grounded "
                    "diagnostic assessments."
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.2

        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        answer = re.sub(
            r"^```json\s*",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"^```\s*",
            "",
            answer
        )

        answer = re.sub(
            r"\s*```$",
            "",
            answer
        )

        diagnostic_data = json.loads(
            answer
        )

        return jsonify({

            "success": True,

            "diagnostic":
            diagnostic_data

        })

    except Exception as error:

        print(
            "\nDIAGNOSTIC ERROR:",
            repr(error)
        )

        return jsonify({

            "success": False,

            "error":
            str(error)

        }), 500


# ============================================================
# STARTUP
# ============================================================

print()
print("==============================")
print("LOADING SAVED CHAT HISTORY")
print("==============================")

print(
    f"Loaded chats: {len(chat_history)}"
)

print("==============================")
print()


print("==============================")
print("LOADING LEARNER MODEL")
print("==============================")

print(
    f"Loaded learner topics: {len(learner_model)}"
)

print("==============================")
print()


print("==============================")
print("LOADING QUESTION HISTORY")
print("==============================")

print(
    f"Loaded questions: {len(question_history)}"
)

print("==============================")
print()


load_existing_pdfs()


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )