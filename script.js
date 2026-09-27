console.log("EduNova AI Dashboard Loaded");
function sendMessage(){

    let input = document.getElementById("userInput");
    let chatBox = document.getElementById("chatBox");

    if(input.value.trim() === ""){
        return;
    }

    let userMessage = document.createElement("div");
    userMessage.className = "user-message";
    userMessage.textContent = input.value;
    chatBox.appendChild(userMessage);

    let aiMessage = document.createElement("div");
    aiMessage.className = "ai-message";
    aiMessage.textContent = "I'm EduNova AI. I'll help you with studies, placements, AI, DBMS, Java, and more.";
    chatBox.appendChild(aiMessage);

    input.value = "";

    chatBox.scrollTop = chatBox.scrollHeight;
}