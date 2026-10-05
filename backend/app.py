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

load_dotenv()

app = Flask(__name__)
CORS(app)

# ==============================
# GROQ
# ==============================

API_KEY = os.getenv("GROQ_API_KEY", "").strip()

client = None

if API_KEY:
    client = Groq(api_key=API_KEY)

GROQ_MODEL = "openai/gpt-oss-20b"


# ==============================
# FOLDERS & FILES
# ==============================

UPLOAD_FOLDER = "uploads"
CHAT_HISTORY_FILE = "chat_history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================
# MEMORY
# ==============================

knowledge_base = []


# ==============================
# CHAT HISTORY FUNCTIONS
# ==============================

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


def save_chat_history():

    try:

        history_to_save = chat_history[-100:]

        with open(
            CHAT_HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history_to_save,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        print(
            f"Could not save chat history: {error}"
        )


chat_history = load_chat_history()


# ==============================
# TEXT CLEANING
# ==============================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==============================
# PDF CHUNKING
# ==============================

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


# ==============================
# INDEX PDF
# ==============================

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


# ==============================
# LOAD SAVED PDFs
# ==============================

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


# ==============================
# RAG SEARCH
# ==============================

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


# ==============================
# HOME
# ==============================

@app.route("/")
def home():

    return "EduNova AI Backend is Running"


# ==============================
# STATUS
# ==============================

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

        "groq_configured": bool(API_KEY),

        "model": GROQ_MODEL,

        "knowledge_chunks": len(
            knowledge_base
        ),

        "saved_pdfs": saved_files,

        "chat_count": len(
            chat_history
        )

    })


# ==============================
# ASK AI
# ==============================

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


    # ==============================
    # SEARCH UPLOADED MATERIAL
    # ==============================

    retrieved = retrieve_context(
        question,
        top_k=5
    )


    # ==============================
    # IF COURSE MATERIAL FOUND
    # ==============================

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

The student has asked a question.

Relevant uploaded course material
is provided below.

Use the uploaded course material
when it is relevant to the question.

If the course material does not contain
enough information, you may answer using
your general knowledge.

Explain the answer simply and clearly
for a college student.

Do not invent information.

If you use the uploaded course material,
include a Sources section at the end
with the filename and page number.

UPLOADED COURSE MATERIAL:

{context}

STUDENT QUESTION:

{question}
"""

        grounded = True

    else:

        # ==============================
        # GENERAL AI MODE
        # ==============================

        prompt = f"""
You are EduNova AI,
a helpful personalized learning tutor.

There is no relevant uploaded course
material available for this question.

Answer the student's question using
your general knowledge.

Explain the concept in simple,
student-friendly language.

For programming, data structures,
AI, machine learning, mathematics,
DBMS and other academic topics:

- Give a clear definition
- Explain the concept simply
- Give an example when useful
- Use bullet points when helpful
- Keep the answer easy to understand
- Avoid unnecessary complexity

Student question:

{question}
"""

        grounded = False


    # ==============================
    # GROQ REQUEST
    # ==============================

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


        # ==============================
        # SOURCES
        # ==============================

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


        # ==============================
        # SAVE CHAT
        # ==============================

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


        # ==============================
        # RESPONSE
        # ==============================

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


# ==============================
# UPLOAD PDF
# ==============================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "file" not in request.files:

        return jsonify({
            "success": False,
            "error": "No file uploaded"
        }), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400


    filename = file.filename


    if not filename.lower().endswith(".pdf"):

        return jsonify({
            "success": False,
            "error": "Please upload a PDF file."
        }), 400


    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    try:

        # ==============================
        # SAVE PDF
        # ==============================

        file.save(file_path)


        # ==============================
        # REMOVE OLD CHUNKS
        # ==============================

        global knowledge_base

        knowledge_base = [

            item

            for item in knowledge_base

            if item["source"] != filename

        ]


        # ==============================
        # READ AND INDEX PDF
        # ==============================

        chunks_added = index_pdf(
            file_path,
            filename
        )


        print(
            f"PDF uploaded successfully: {filename}"
        )

        print(
            f"Chunks added: {chunks_added}"
        )


        # ==============================
        # SUCCESS RESPONSE
        # ==============================

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


# ==============================
# CHAT HISTORY
# ==============================

@app.route(
    "/history",
    methods=["GET"]
)
def history():

    return jsonify({

        "history":
        chat_history

    })


# ==============================
# QUIZ
# ==============================

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

    except (TypeError, ValueError):

        count = 5


    if count < 1:
        count = 1

    if count > 10:
        count = 10


    if not topic:

        return jsonify({

            "error":
            "Please enter a topic."

        }), 400


    if client is None:

        return jsonify({

            "error":
            "Groq API key is not configured."

        }), 500


    # ==============================
    # SEARCH COURSE MATERIAL
    # ==============================

    retrieved = retrieve_context(
        topic,
        top_k=8
    )


    if not retrieved:

        return jsonify({

            "error":
            "I couldn't find this topic "
            "in the uploaded course material."

        }), 404


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


    # ==============================
    # QUIZ PROMPT
    # ==============================

    prompt = f"""
You are EduNova AI,
an adaptive learning assessment generator.

Create {count} multiple-choice questions
about:

TOPIC:
{topic}

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


    # ==============================
    # GENERATE QUIZ
    # ==============================

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",

                    "content":
                    "You generate "
                    "source-grounded "
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


        # ==============================
        # REMOVE MARKDOWN CODE BLOCK
        # ==============================

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


        # ==============================
        # PARSE JSON
        # ==============================

        quiz_data = json.loads(
            answer
        )


        return jsonify({

            "success": True,

            "quiz":
            quiz_data,

            "grounded":
            True

        }), 200


    except json.JSONDecodeError as error:

        print(
            "\nQUIZ JSON ERROR:",
            repr(error)
        )

        return jsonify({

            "success": False,

            "error":
            "AI returned an invalid quiz format.",

            "details":
            str(error)

        }), 500


    except Exception as error:

        print(
            "\nQUIZ ERROR:",
            repr(error)
        )

        return jsonify({

            "success": False,

            "error":
            str(error)

        }), 500


# ==============================
# STARTUP
# ==============================

print()
print("==============================")
print("LOADING SAVED CHAT HISTORY")
print("==============================")

print(
    f"Loaded chats: {len(chat_history)}"
)

print("==============================")
print()


load_existing_pdfs()


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )