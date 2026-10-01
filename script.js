/* =====================================================
   EduNova AI
   Login / User Name JavaScript
===================================================== */


/* =====================================================
   CONTINUE TO DASHBOARD
===================================================== */

function continueToDashboard() {

    // Get name from login page
    const nameInput =
        document.getElementById("studentName");


    // Make sure the input exists
    if (!nameInput) {

        alert("Name field not found.");

        return;
    }


    // Get entered name
    const studentName =
        nameInput.value.trim();


    // Check name
    if (studentName === "") {

        alert("Please enter your name.");

        nameInput.focus();

        return;
    }


    // Save only the name
    localStorage.setItem(
        "studentName",
        studentName
    );


    // Open dashboard
    window.location.href =
        "index.html";
}


/* =====================================================
   SHOW USER NAME ON DASHBOARD
===================================================== */

function loadStudentName() {

    const studentName =
        localStorage.getItem(
            "studentName"
        );


    const welcomeText =
        document.getElementById(
            "welcomeText"
        );


    if (
        studentName &&
        welcomeText
    ) {

        welcomeText.textContent =
            "Welcome, " +
            studentName +
            " 👋";
    }

}


/* =====================================================
   LOGOUT
===================================================== */

function logoutStudent() {

    // Remove saved name
    localStorage.removeItem(
        "studentName"
    );


    // Go back to login
    window.location.href =
        "login.html";
}


/* =====================================================
   PAGE LOAD
===================================================== */

document.addEventListener(
    "DOMContentLoaded",
    function () {

        loadStudentName();

    }
);