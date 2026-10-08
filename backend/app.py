from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from pptx import Presentation
from moviepy import VideoFileClip
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


# General AI chat model
GROQ_MODEL = "openai/gpt-oss-20b"

# More capable model for structured quiz generation
QUIZ_MODEL = "openai/gpt-oss-120b"

# Speech-to-text model
WHISPER_MODEL = "whisper-large-v3-turbo"


# ============================================================
# FILES
# ============================================================

UPLOAD_FOLDER = "uploads"

CHAT_HISTORY_FILE = "chat_history.json"
LEARNER_MODEL_FILE = "learner_model.json"
QUESTION_HISTORY_FILE = "question_history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

PDF_EXTENSIONS = (
    ".pdf",
)

PPTX_EXTENSIONS = (
    ".pptx",
)

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
)


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

    text = str(text or "")

    text = text.replace(
        "\x00",
        " "
    )

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

    text = str(text or "").lower()

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
# TEXT CHUNKING
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

        text = clean_text(
            raw_text
        )

        if not text:
            continue

        chunks = create_chunks(
            text
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            knowledge_base.append({

                "source": filename,

                "page": page_number,

                "slide": 0,

                "timestamp": "",

                "start": 0,

                "end": 0,

                "chunk": chunk_number,

                "text": chunk

            })

            added_chunks += 1

    return added_chunks


# ============================================================
# INDEX POWERPOINT PPTX
# ============================================================

def index_pptx(
    file_path,
    filename
):

    presentation = Presentation(
        file_path
    )

    added_chunks = 0

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):

        slide_text_parts = []

        # ----------------------------------------------------
        # EXTRACT TEXT FROM SLIDE SHAPES
        # ----------------------------------------------------

        for shape in slide.shapes:

            if hasattr(
                shape,
                "text"
            ):

                text = clean_text(
                    shape.text
                )

                if text:

                    slide_text_parts.append(
                        text
                    )

        # ----------------------------------------------------
        # EXTRACT TABLE CONTENT
        # ----------------------------------------------------

        for shape in slide.shapes:

            if not hasattr(
                shape,
                "has_table"
            ):
                continue

            if not shape.has_table:
                continue

            table_text = []

            for row in shape.table.rows:

                row_values = []

                for cell in row.cells:

                    cell_text = clean_text(
                        cell.text
                    )

                    if cell_text:

                        row_values.append(
                            cell_text
                        )

                if row_values:

                    table_text.append(
                        " | ".join(
                            row_values
                        )
                    )

            if table_text:

                slide_text_parts.append(
                    "Table: "
                    + " ; ".join(table_text)
                )

        slide_text = " ".join(
            slide_text_parts
        )

        if not slide_text:
            continue

        # ----------------------------------------------------
        # CHUNK SLIDE TEXT
        # ----------------------------------------------------

        chunks = create_chunks(
            slide_text,
            chunk_size=1200
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            knowledge_base.append({

                "source": filename,

                "page": 0,

                "slide": slide_number,

                "timestamp": "",

                "start": 0,

                "end": 0,

                "chunk": chunk_number,

                "text": chunk

            })

            added_chunks += 1

    return added_chunks


# ============================================================
# FORMAT VIDEO TIMESTAMP
# ============================================================

def format_timestamp(seconds):

    try:
        seconds = int(float(seconds or 0))
    except Exception:
        seconds = 0

    hours = seconds // 3600

    minutes = (
        (seconds % 3600)
        // 60
    )

    secs = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# INDEX VIDEO
# ============================================================

def index_video(
    file_path,
    filename
):

    print(
        f"Processing VIDEO: {filename}"
    )

    video = None

    audio_path = os.path.join(
        UPLOAD_FOLDER,
        "_temp_audio.mp3"
    )

    try:

        # ----------------------------------------------------
        # 1. OPEN VIDEO
        # ----------------------------------------------------

        video = VideoFileClip(
            file_path
        )

        if video.audio is None:

            print(
                f"No audio found in {filename}"
            )

            return 0

        # ----------------------------------------------------
        # 2. EXTRACT AUDIO
        # ----------------------------------------------------

        video.audio.write_audiofile(
            audio_path,
            logger=None
        )

        # Close video after extracting audio
        video.close()
        video = None

        # ----------------------------------------------------
        # 3. CHECK GROQ
        # ----------------------------------------------------

        if client is None:

            raise ValueError(
                "Groq API key is not configured."
            )

        # ----------------------------------------------------
        # 4. TRANSCRIBE WITH WHISPER
        # ----------------------------------------------------

        with open(
            audio_path,
            "rb"
        ) as audio_file:

            transcription = (
                client.audio.transcriptions.create(

                    file=audio_file,

                    model=WHISPER_MODEL,

                    response_format="verbose_json",

                    timestamp_granularities=[
                        "segment"
                    ]
                )
            )

        # ----------------------------------------------------
        # 5. GET SEGMENTS
        # ----------------------------------------------------

        segments = getattr(
            transcription,
            "segments",
            None
        )

        if segments is None:

            segments = []

        added_chunks = 0

        # ----------------------------------------------------
        # 6. CREATE TIMESTAMPED KNOWLEDGE CHUNKS
        # ----------------------------------------------------

        for segment_number, segment in enumerate(
            segments,
            start=1
        ):

            if isinstance(
                segment,
                dict
            ):

                start_time = segment.get(
                    "start",
                    0
                )

                end_time = segment.get(
                    "end",
                    0
                )

                text = segment.get(
                    "text",
                    ""
                )

            else:

                start_time = getattr(
                    segment,
                    "start",
                    0
                )

                end_time = getattr(
                    segment,
                    "end",
                    0
                )

                text = getattr(
                    segment,
                    "text",
                    ""
                )

            text = clean_text(
                text
            )

            if not text:
                continue

            timestamp = format_timestamp(
                start_time
            )

            knowledge_base.append({

                "source": filename,

                "page": 0,

                "slide": 0,

                "timestamp": timestamp,

                "start": float(
                    start_time or 0
                ),

                "end": float(
                    end_time or 0
                ),

                "chunk": segment_number,

                "text": text

            })

            added_chunks += 1

        print(
            f"Loaded VIDEO: "
            f"{filename} | "
            f"{added_chunks} timestamp chunks"
        )

        return added_chunks

    except Exception as error:

        print(
            f"VIDEO ERROR for "
            f"{filename}: "
            f"{repr(error)}"
        )

        return 0

    finally:

        if video is not None:

            try:
                video.close()
            except Exception:
                pass

        if os.path.exists(
            audio_path
        ):

            try:

                os.remove(
                    audio_path
                )

            except Exception:
                pass


# ============================================================
# LOAD EXISTING LEARNING MATERIAL
# ============================================================

def load_existing_learning_material():

    knowledge_base.clear()

    files = os.listdir(
        UPLOAD_FOLDER
    )

    supported_files = [

        file

        for file in files

        if file.lower().endswith(
            PDF_EXTENSIONS
            + PPTX_EXTENSIONS
            + VIDEO_EXTENSIONS
        )

    ]

    print()
    print("==============================")
    print("LOADING SAVED LEARNING MATERIAL")
    print("==============================")

    for filename in supported_files:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        try:

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            if filename.lower().endswith(
                PDF_EXTENSIONS
            ):

                chunks = index_pdf(
                    file_path,
                    filename
                )

                print(
                    f"Loaded PDF: "
                    f"{filename} | "
                    f"{chunks} chunks"
                )

            # ------------------------------------------------
            # PPTX
            # ------------------------------------------------

            elif filename.lower().endswith(
                PPTX_EXTENSIONS
            ):

                chunks = index_pptx(
                    file_path,
                    filename
                )

                print(
                    f"Loaded PPTX: "
                    f"{filename} | "
                    f"{chunks} chunks"
                )

            # ------------------------------------------------
            # VIDEO
            # ------------------------------------------------

            elif filename.lower().endswith(
                VIDEO_EXTENSIONS
            ):

                chunks = index_video(
                    file_path,
                    filename
                )

                print(
                    f"Loaded VIDEO: "
                    f"{filename} | "
                    f"{chunks} chunks"
                )

        except Exception as error:

            print(
                f"Could not load "
                f"{filename}: {error}"
            )

    print(
        f"Total knowledge chunks: "
        f"{len(knowledge_base)}"
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

        document_vectors = (
            vectorizer.fit_transform(
                documents
            )
        )

        question_vector = (
            vectorizer.transform(
                [question]
            )
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

        item = knowledge_base[
            index
        ].copy()

        item["score"] = round(
            score,
            4
        )

        results.append(
            item
        )

    return results


# ============================================================
# TOPIC MASTERY
# ============================================================

def get_topic_mastery(topic):

    if topic not in learner_model:
        return 0

    return learner_model[topic].get(
        "mastery",
        0
    )


# ============================================================
# WEAK TOPICS
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
# ADAPTIVE DIFFICULTY
# ============================================================

def choose_adaptive_difficulty(mastery):

    if mastery < 40:

        return "Beginner"

    elif mastery < 70:

        return "Intermediate"

    return "Advanced"


# ============================================================
# PREVIOUS QUESTIONS
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

        if not isinstance(
            question,
            dict
        ):
            continue

        question_text = str(
            question.get(
                "question",
                ""
            )
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
                ),

                "slide": question.get(
                    "slide",
                    0
                ),

                "timestamp": question.get(
                    "timestamp",
                    ""
                )

            })

    save_question_history()


# ============================================================
# SAFE AI JSON CLEANER
# ============================================================

def clean_ai_json(text):

    if text is None:

        raise ValueError(
            "AI returned no response."
        )

    text = str(text).strip()

    if not text:

        raise ValueError(
            "AI returned an empty response."
        )

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    first_object = text.find("{")

    last_object = text.rfind("}")

    if (
        first_object != -1
        and last_object != -1
        and last_object > first_object
    ):

        text = text[
            first_object:last_object + 1
        ]

    if (
        not text.startswith("{")
        and not text.startswith("[")
    ):

        raise ValueError(
            "AI response does not contain valid JSON."
        )

    return text.strip()


# ============================================================
# VALIDATE QUIZ
# ============================================================

def validate_quiz_data(
    quiz_data,
    requested_count
):

    if not isinstance(
        quiz_data,
        dict
    ):

        return False, (
            "Quiz response is not a JSON object."
        )

    questions = quiz_data.get(
        "questions"
    )

    if not isinstance(
        questions,
        list
    ):

        return False, (
            "Quiz questions are missing."
        )

    if len(questions) == 0:

        return False, (
            "No quiz questions were generated."
        )

    valid_questions = []

    for question in questions:

        if not isinstance(
            question,
            dict
        ):
            continue

        question_text = str(
            question.get(
                "question",
                ""
            )
        ).strip()

        options = question.get(
            "options"
        )

        correct_answer = str(
            question.get(
                "correct_answer",
                ""
            )
        ).strip()

        explanation = str(
            question.get(
                "explanation",
                ""
            )
        ).strip()

        difficulty = str(
            question.get(
                "difficulty",
                "Intermediate"
            )
        ).strip()

        source = str(
            question.get(
                "source",
                ""
            )
        ).strip()

        page = question.get(
            "page",
            0
        )

        slide = question.get(
            "slide",
            0
        )

        timestamp = str(
            question.get(
                "timestamp",
                ""
            )
        ).strip()

        if not question_text:
            continue

        if not isinstance(
            options,
            list
        ):
            continue

        if len(options) != 4:
            continue

        options = [

            str(option).strip()

            for option in options

        ]

        if any(
            not option
            for option in options
        ):
            continue

        if len(set(options)) != 4:
            continue

        if correct_answer not in options:

            option_map = {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 3
            }

            upper_answer = correct_answer.upper()

            if upper_answer in option_map:

                correct_answer = options[
                    option_map[upper_answer]
                ]

            else:

                continue

        if difficulty not in [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]:

            difficulty = "Intermediate"

        try:

            page = int(page)

        except:

            page = 0

        try:

            slide = int(slide)

        except:

            slide = 0

        question["question"] = question_text
        question["options"] = options
        question["correct_answer"] = correct_answer
        question["explanation"] = explanation
        question["difficulty"] = difficulty
        question["source"] = source
        question["page"] = page
        question["slide"] = slide
        question["timestamp"] = timestamp

        valid_questions.append(
            question
        )

    if not valid_questions:

        return False, (
            "AI generated questions, "
            "but none passed validation."
        )

    unique_questions = []

    seen = set()

    for question in valid_questions:

        normalized = normalize_question(
            question["question"]
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        unique_questions.append(
            question
        )

    if not unique_questions:

        return False, (
            "No unique questions were generated."
        )

    quiz_data["questions"] = unique_questions[
        :requested_count
    ]

    return True, None


# ============================================================
# GROQ STRUCTURED REQUEST
# ============================================================

def call_groq_json(
    prompt,
    system_message,
    model=QUIZ_MODEL
):

    if client is None:

        raise ValueError(
            "Groq API key is not configured."
        )

    # --------------------------------------------------------
    # ATTEMPT 1: JSON MODE
    # --------------------------------------------------------

    try:

        print(
            f"Trying JSON mode with {model}..."
        )

        response = client.chat.completions.create(

            model=model,

            messages=[

                {
                    "role": "system",
                    "content": system_message
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.1,

            max_tokens=5000,

            response_format={
                "type": "json_object"
            }

        )

        if not response.choices:

            raise ValueError(
                "Groq returned no choices."
            )

        answer = response.choices[0].message.content

        if answer is None:

            raise ValueError(
                "Groq returned empty content."
            )

        answer = str(answer).strip()

        if not answer:

            raise ValueError(
                "Groq returned an empty response."
            )

        print()
        print("RAW AI RESPONSE:")
        print(answer)
        print()

        return answer

    except Exception as error:

        print()
        print(
            "JSON MODE ERROR:",
            repr(error)
        )
        print()

    # --------------------------------------------------------
    # ATTEMPT 2: NORMAL TEXT MODE
    # --------------------------------------------------------

    try:

        print(
            f"Retrying {model} without JSON mode..."
        )

        response = client.chat.completions.create(

            model=model,

            messages=[

                {
                    "role": "system",
                    "content":
                    system_message
                    + "\nReturn ONLY valid JSON."
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.1,

            max_tokens=5000

        )

        if not response.choices:

            raise ValueError(
                "Groq returned no choices."
            )

        answer = response.choices[0].message.content

        if answer is None:

            raise ValueError(
                "Groq returned empty content."
            )

        answer = str(answer).strip()

        if not answer:

            raise ValueError(
                "Groq returned an empty response."
            )

        print()
        print("RAW AI RETRY RESPONSE:")
        print(answer)
        print()

        return answer

    except Exception as error:

        print()
        print(
            "NORMAL MODE ERROR:",
            repr(error)
        )
        print()

        raise ValueError(
            f"AI generation failed: {error}"
        )


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

        return None, (
            "Topic not found in uploaded material."
        )

    context_parts = []

    for item in retrieved:

        if item.get("timestamp"):

            location = (
                f"TIMESTAMP: "
                f"{item['timestamp']}"
            )

        elif item.get("slide", 0):

            location = (
                f"SLIDE: {item['slide']}"
            )

        else:

            location = (
                f"PAGE: {item['page']}"
            )

        context_parts.append(

            f"SOURCE: {item['source']}\n"
            f"{location}\n"
            f"CONTENT: {item['text']}"

        )

    context = "\n\n".join(
        context_parts
    )

    previous_questions = get_previous_questions(
        topic
    )

    previous_text = "\n".join(
        previous_questions[-30:]
    )

    if not previous_text:

        previous_text = (
            "No previous questions."
        )

    difficulty_instruction = ""

    if difficulty:

        difficulty_instruction = f"""
Required difficulty: {difficulty}

All generated questions must use this
difficulty unless the question cannot
reasonably be created at that level.
"""

    adaptive_instruction = ""

    if adaptive:

        adaptive_instruction = """
This is an adaptive quiz.

The backend has already selected the
difficulty.

Do not change the selected difficulty.
"""

    prompt = f"""
Create exactly {count} multiple-choice questions
for the topic "{topic}".

{difficulty_instruction}

{adaptive_instruction}

SOURCE RULE:
Use ONLY the uploaded material below.

Do not use outside knowledge.

REPETITION RULE:
Do not repeat or closely rephrase previous questions.

PREVIOUS QUESTIONS:
{previous_text}

Each question must contain:

question
options
correct_answer
explanation
difficulty
source
page
slide
timestamp

Rules:

- Exactly 4 options.
- All 4 options must be different.
- correct_answer must exactly match one option.
- explanation must be short.
- difficulty must be Beginner, Intermediate, or Advanced.
- source must be an uploaded filename.
- For PDF material, page must be the real page number.
- For PPTX material, slide must be the real slide number.
- For video material, timestamp must be the real timestamp.
- Do not invent page, slide, or timestamp values.
- Generate exactly {count} questions.
- Return JSON only.

JSON:

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
      "explanation": "Short explanation.",
      "difficulty": "Beginner",
      "source": "filename.pdf",
      "page": 1,
      "slide": 0,
      "timestamp": ""
    }}
  ]
}}

UPLOADED MATERIAL:

{context}
"""

    system_message = """
You are EduNova AI's quiz generator.

Generate source-grounded educational
multiple-choice questions.

Follow the requested JSON structure exactly.

Never write Markdown.

Never write anything outside the JSON.
"""

    for attempt in range(2):

        print()
        print(
            f"Generating quiz for '{topic}' "
            f"(attempt {attempt + 1}/2)"
        )

        try:

            raw_response = call_groq_json(
                prompt,
                system_message,
                QUIZ_MODEL
            )

            cleaned_json = clean_ai_json(
                raw_response
            )

            quiz_data = json.loads(
                cleaned_json
            )

            valid, validation_error = validate_quiz_data(
                quiz_data,
                count
            )

            if not valid:

                print(
                    "QUIZ VALIDATION ERROR:",
                    validation_error
                )

                if attempt == 0:
                    continue

                return None, validation_error

            questions = quiz_data[
                "questions"
            ]

            if len(questions) < count:

                message = (
                    f"AI generated only "
                    f"{len(questions)} questions "
                    f"instead of {count}."
                )

                print(
                    "QUIZ COUNT ERROR:",
                    message
                )

                if attempt == 0:
                    continue

                return None, message

            save_generated_questions(
                topic,
                questions
            )

            return quiz_data, None

        except json.JSONDecodeError as error:

            print(
                "QUIZ JSON ERROR:",
                repr(error)
            )

            if attempt == 1:

                return None, (
                    "AI returned invalid quiz JSON. "
                    "Please try again."
                )

        except Exception as error:

            print(
                "QUIZ GENERATION ERROR:",
                repr(error)
            )

            if attempt == 1:

                return None, (
                    "Quiz generation failed. "
                    "Please try again."
                )

    return None, (
        "Quiz generation failed. "
        "Please try again."
    )


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

        if file.lower().endswith(
            PDF_EXTENSIONS
            + PPTX_EXTENSIONS
            + VIDEO_EXTENSIONS
        )

    ]

    saved_pdfs = [

        file

        for file in saved_files

        if file.lower().endswith(
            PDF_EXTENSIONS
        )

    ]

    saved_pptx = [

        file

        for file in saved_files

        if file.lower().endswith(
            PPTX_EXTENSIONS
        )

    ]

    saved_videos = [

        file

        for file in saved_files

        if file.lower().endswith(
            VIDEO_EXTENSIONS
        )

    ]

    return jsonify({

        "status": "ready",

        "groq_configured": bool(
            API_KEY
        ),

        "model": GROQ_MODEL,

        "quiz_model": QUIZ_MODEL,

        "whisper_model": WHISPER_MODEL,

        "knowledge_chunks": len(
            knowledge_base
        ),

        "saved_learning_material": saved_files,

        "saved_pdfs": saved_pdfs,

        "saved_pptx": saved_pptx,

        "saved_videos": saved_videos,

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

            # ------------------------------------------------
            # VIDEO CONTEXT
            # ------------------------------------------------

            if item.get("timestamp"):

                location_text = (
                    f"TIMESTAMP: "
                    f"{item['timestamp']}"
                )

            # ------------------------------------------------
            # PDF CONTEXT
            # ------------------------------------------------

            elif item.get("page", 0):

                location_text = (
                    f"PAGE: {item['page']}"
                )

            # ------------------------------------------------
            # PPTX CONTEXT
            # ------------------------------------------------

            elif item.get("slide", 0):

                location_text = (
                    f"SLIDE: {item['slide']}"
                )

            else:

                location_text = (
                    "LOCATION: Unknown"
                )

            context_parts.append(

                f"""
SOURCE:
{item['source']}

{location_text}

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

Clearly distinguish uploaded-material
information from general knowledge.

Explain clearly for a college student.

Do not invent information.

SOURCE CITATION RULES:

1. If the answer uses a PDF,
   cite the exact filename and page.

2. If the answer uses a PowerPoint,
   cite the exact filename and slide.

3. If the answer uses a video,
   cite the exact filename and timestamp.

4. Never invent a page number.

5. Never invent a slide number.

6. Never invent a timestamp.

7. If the uploaded material is not
   enough to answer the question,
   clearly say that.

8. If general knowledge is used,
   clearly identify it as general knowledge.

Use these citation formats:

For PDF:
Sources:
- filename.pdf — Page 3

For PowerPoint:
Sources:
- filename.pptx — Slide 5

For video:
Sources:
- filename.mp4 — 00:07:10

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

There is no relevant uploaded material
for this question.

Clearly state that the uploaded learning
material does not cover this question.

Then answer using general knowledge.

Do not create fake citations.

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

            temperature=0.2,

            max_tokens=3000

        )

        if not response.choices:

            raise ValueError(
                "AI returned no response."
            )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer:

            raise ValueError(
                "AI returned an empty response."
            )

        answer = answer.strip()

        # ----------------------------------------------------
        # REMOVE DUPLICATE SOURCES
        # ----------------------------------------------------

        sources = []

        seen_sources = set()

        for item in retrieved:

            source_key = (

                item.get(
                    "source",
                    ""
                ),

                item.get(
                    "page",
                    0
                ),

                item.get(
                    "slide",
                    0
                ),

                item.get(
                    "timestamp",
                    ""
                )

            )

            if source_key in seen_sources:
                continue

            seen_sources.add(
                source_key
            )

            source = {

                "source":
                item["source"],

                "page":
                item.get(
                    "page",
                    0
                ),

                "slide":
                item.get(
                    "slide",
                    0
                ),

                "timestamp":
                item.get(
                    "timestamp",
                    ""
                ),

                "chunk":
                item["chunk"],

                "score":
                item["score"]

            }

            sources.append(
                source
            )

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
            "AI response failed. Please try again.",

            "provider":
            "groq",

            "model":
            GROQ_MODEL,

            "grounded":
            False,

            "sources":
            [],

            "error":
            str(error)

        }), 500


# ============================================================
# UPLOAD PDF / PPTX / VIDEO
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

    supported_extensions = (
        PDF_EXTENSIONS
        + PPTX_EXTENSIONS
        + VIDEO_EXTENSIONS
    )

    if not filename.lower().endswith(
        supported_extensions
    ):

        return jsonify({

            "success": False,

            "error":
            "Please upload a PDF, PPTX, or video file."

        }), 400

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:

        file.save(
            file_path
        )

        global knowledge_base

        knowledge_base = [

            item

            for item in knowledge_base

            if item["source"] != filename

        ]

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if filename.lower().endswith(
            PDF_EXTENSIONS
        ):

            chunks_added = index_pdf(
                file_path,
                filename
            )

            file_type = "PDF"

        # ----------------------------------------------------
        # POWERPOINT
        # ----------------------------------------------------

        elif filename.lower().endswith(
            PPTX_EXTENSIONS
        ):

            chunks_added = index_pptx(
                file_path,
                filename
            )

            file_type = "PowerPoint"

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        else:

            chunks_added = index_video(
                file_path,
                filename
            )

            file_type = "Video"

        return jsonify({

            "success": True,

            "message":
            f"{file_type} indexed successfully",

            "source":
            filename,

            "file_type":
            file_type,

            "chunks":
            chunks_added,

            "total_knowledge_chunks":
            len(knowledge_base)

        }), 200

    except Exception as error:

        print(
            "\nFILE UPLOAD ERROR:",
            repr(error)
        )

        return jsonify({

            "success": False,

            "error":
            "Could not read the uploaded file.",

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

            action = (
                "Start with basic concepts and examples."
            )

        elif mastery < 60:

            action = (
                "Revise the topic and take an adaptive quiz."
            )

        else:

            action = (
                "Practice more questions to improve mastery."
            )

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

        if item.get("timestamp"):

            location = (
                f"TIMESTAMP: "
                f"{item['timestamp']}"
            )

        elif item.get("slide", 0):

            location = (
                f"SLIDE: {item['slide']}"
            )

        else:

            location = (
                f"PAGE: {item['page']}"
            )

        context_parts.append(

            f"SOURCE: {item['source']}\n"
            f"{location}\n"
            f"CONTENT: {item['text']}"

        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
Create exactly {count} diagnostic
multiple-choice questions for a new learner.

Cover different concepts from the
uploaded learning material.

Use ONLY the uploaded material.

Each question must contain:

question
options
correct_answer
explanation
difficulty
topic
source
page
slide
timestamp

Rules:

- Exactly 4 options.
- All options must be different.
- correct_answer must match one option.
- difficulty must be Beginner,
  Intermediate, or Advanced.
- Do not invent page, slide, or timestamp numbers.
- Return JSON only.
- Do not use Markdown.

JSON:

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
      "page": 1,
      "slide": 0,
      "timestamp": ""
    }}
  ]
}}

UPLOADED MATERIAL:

{context}
"""

    system_message = """
You are EduNova AI's diagnostic assessment generator.

Create source-grounded diagnostic questions.

Return ONLY valid JSON.

Do not use Markdown.
"""

    try:

        raw_response = call_groq_json(
            prompt,
            system_message,
            QUIZ_MODEL
        )

        cleaned_json = clean_ai_json(
            raw_response
        )

        diagnostic_data = json.loads(
            cleaned_json
        )

        questions = diagnostic_data.get(
            "questions",
            []
        )

        if not isinstance(
            questions,
            list
        ):

            raise ValueError(
                "Diagnostic questions are missing."
            )

        if not questions:

            raise ValueError(
                "No diagnostic questions were generated."
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


# ============================================================
# LOAD PDF + PPTX + VIDEO MATERIAL
# ============================================================

load_existing_learning_material()


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )