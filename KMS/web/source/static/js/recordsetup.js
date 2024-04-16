navigator
.mediaDevices
.getUserMedia({audio: true})
.then(stream => { handlerFunction(stream) });

var myEle = document.getElementById("resultanswer");

function handlerFunction(stream) {
rec = new MediaRecorder(stream);
rec.ondataavailable = e => {
    audioChunks.push(e.data);
    if (rec.state == "inactive") {
        let blob = new Blob(audioChunks, {type: 'audio/mp4a-latm'});
        sendData(blob);
    }
}
}

function sendData(data) {
var form = new FormData();
form.append('file', data, 'data.m4a');
form.append('title', 'data.m4a');
//Chrome inspector shows that the post data includes a file and a title.
$.ajax({
    type: 'POST',
    url: '/save-record',
    data: form,
    cache: false,
    processData: false,
    contentType: false
})
.done(function(data) {
    console.log(data);
    window.location.href = "http://localhost:5000/";
});
}

recordButton.onclick = e => {
console.log('Recording are started..');
recordButton.disabled = true;
stopButton.disabled = false;
audioChunks = [];
rec.start();
};

stopButton.onclick = e => {
console.log("Recording are stopped.");
recordButton.disabled = false;
stopButton.disabled = true;
rec.stop();
};


resultanswer.onclick = e => {
    resultanswer.disabled = true;
    stopnarrative();
}
narration.onclick = e => {
    stopnarrative();
}

function stopnarrative(){
    speechSynthesis.cancel();
}

if (myEle) {
    narration.disabled = false;
    var text = resultanswer.textContent;
    var msg = new SpeechSynthesisUtterance();
    var voices = window.speechSynthesis.getVoices();
    setTimeout(() => {
        console.log(window.speechSynthesis.getVoices());    
        var voices = window.speechSynthesis.getVoices();
        msg.voice = voices[192];
        msg.rate = 1.5;
        msg.pitch = 1;
        msg.text = text;
        speechSynthesis.speak(msg);
        
    }, 1000);

};
