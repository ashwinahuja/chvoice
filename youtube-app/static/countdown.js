var seconds = 10;
      
function countdown() {
    seconds = seconds - 1;
    if (seconds < 0) {
        var Url = window.location;
        // Redirection link 
        window.location = Url["origin"] + "/youtube-dl" ;
    } else {
        // Update remaining seconds
        document.getElementById("countdown-number").innerHTML = seconds;
        console.log(seconds)
        // Count down using javascript
        window.setTimeout("countdown()", 1000);
    }
}

// Run countdown function
countdown();