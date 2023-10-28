function copytxt() {
    navigator.clipboard.writeText("thanagorn2005@gmail.com");
    let myAlert = document.querySelectorAll('.toast');
    let bsAlert = new bootstrap.Toast(myAlert);
    bsAlert.show();
}
document.getElementById("cpybtn").onclick = function () {
    var myAlert = document.getElementById("Copied");
    var bsAlert = new bootstrap.Toast(myAlert);
    bsAlert.show();
}

//alert
window.onload = (event) => {
    let myAlert = document.getElementById("MobileAlert");
    var bsAlert = new bootstrap.Toast(myAlert);
    bsAlert.show();//show it
};

// Collapsible
var coll = document.getElementsByClassName("collapsible");

for (let i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function () {
        this.classList.toggle("active");

        var content = this.nextElementSibling;

        if (content.style.maxHeight) {
            content.style.maxHeight = null;
        } else {
            content.style.maxHeight = content.scrollHeight + "px";
        }

    });
}

function getTime() {
    let today = new Date();
    hours = today.getHours();
    minutes = today.getMinutes();

    if (hours < 10) {
        hours = "0" + hours;
    }

    if (minutes < 10) {
        minutes = "0" + minutes;
    }

    let time = hours + ":" + minutes;
    return time;
}

// Gets the first message
function firstBotMessage() {
    let firstMessage = "How's it going?"
    document.getElementById("botStarterMessage").innerHTML = '<p class="botText"><span>' + firstMessage + '</span></p>';

    let time = getTime();

    $("#chat-timestamp").append(time);
    document.getElementById("userInput").scrollIntoView(false);

}

firstBotMessage();

// Retrieves the response
function getHardResponse(userText) {
    let botResponse = getBotResponse(userText);
    let botHtml = '<p class="botText"><span>' + botResponse + '</span></p>';
    $("#chatbox").append(botHtml);

    document.getElementById("chat-bar-bottom").scrollIntoView(true);
}

//Gets the text text from the input box and processes it
function getResponse() {
    let userText = $("#textInput").val();

    if (userText == "") {
        userText = "สวัสดี";
    }

    let userHtml = '<p class="userText"><span>' + userText + '</span></p>';

    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document.getElementById("chat-bar-bottom").scrollIntoView(true);

    // Send user message to the Flask app
    $.ajax({
        type: 'POST',
        url: '/chat',
        data: { user_message: userText },
        success: function (data) {
            let botResponse = data.bot_response;
            let botHtml = '<p class="botText"><span>' + botResponse + '</span></p>';
            $("#chatbox").append(botHtml);
            document.getElementById("chat-bar-bottom").scrollIntoView(true);
        }
    });

    setTimeout(() => {
        getHardResponse(userText);
    }, 1000)

    
}

// Function to handle user input
function sendUserMessage() {
    let userText = $("#textInput").val();

    if (userText == "") {
        userText = "Text something!";
    }

    let userHtml = '<p class="userText"><span>' + userText + '</span></p>';

    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document.getElementById("chat-bar-bottom").scrollIntoView(true);

    // Send user message to the Flask app and receive the response
    getResponse(userText);
}

// Function to handle predefined button clicks
function buttonSendText(sampleText) {
    let userHtml = '<p class="userText"><span>' + sampleText + '</span></p>';

    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document.getElementById("chat-bar-bottom").scrollIntoView(true);

    // Send the predefined message to the Flask app and receive the response
    getResponse(sampleText);
}

function sendButton() {
    getResponse();
}

function heartButton() {
    buttonSendText("Heart clicked!")
}

// Press enter to send a message
$("#textInput").keypress(function (e) {
    if (e.which == 13) {
        getResponse();
    }
});