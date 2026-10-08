const API_URL = "http://127.0.0.1:5000";


async function askEduNova(question) {

    question = question.trim();

    if (question === "") {
        return {
            success: false,
            answer: "Please enter a question."
        };
    }

    try {

        const response = await fetch(
            API_URL + "/ask",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question: question,
                    provider: "groq"
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {

            return {
                success: false,
                answer:
                    data.answer ||
                    "Something went wrong."
            };
        }

        return {
            success: true,
            answer: data.answer,
            provider: data.provider || "groq",
            model: data.model || "",
            grounded: data.grounded || false,
            source: data.source || null
        };

    } catch (error) {

        console.error(
            "EduNova AI connection error:",
            error
        );

        return {
            success: false,
            answer:
                "EduNova AI could not connect to the backend. Please make sure Flask is running."
        };
    }
