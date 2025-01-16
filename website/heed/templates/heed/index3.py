<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audio Recorder</title>
  <style>
    #recordButton {
      padding: 10px 20px;
      background-color: red;
      color: white;
      border: none;
      cursor: pointer;
    }
    #recordButton.recording {
      background-color: green;
    }
    #audioPlayer {
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <button id="recordButton">Start Recording</button>
  <button id="deleteButton" style="display: none;">Delete Recording</button>
  <audio id="audioPlayer" controls style="display: none;"></audio>

  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script>
    // Get references to DOM elements
    const recordButton = document.getElementById("recordButton");
    const deleteButton = document.getElementById("deleteButton");
    const audioPlayer = document.getElementById("audioPlayer");
    
    let mediaRecorder;
    let audioBlob;
    let audioUrl;
    let audioFile;

    // Function to start recording
    function startRecording() {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
          .then(stream => {
            // Set up MediaRecorder
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (event) => {
              audioBlob = event.data;
              audioUrl = URL.createObjectURL(audioBlob);
              audioFile = new File([audioBlob], "recorded_audio.wav", { type: "audio/wav" });

              // Create audio player and show it
              audioPlayer.src = audioUrl;
              audioPlayer.style.display = 'block';

              // Enable the delete button
              deleteButton.style.display = 'inline-block';

              // Send the audio to the backend
              sendAudioToBackend(audioFile);
            };

            // Start recording for 2 seconds
            mediaRecorder.start();
            recordButton.textContent = "Recording...";
            recordButton.classList.add("recording");

            // Stop recording after 2 seconds
            setTimeout(() => {
              mediaRecorder.stop();
              recordButton.textContent = "Start Recording";
              recordButton.classList.remove("recording");
            }, 2000);
          })
          .catch(err => {
            console.error("Error accessing the microphone: ", err);
          });
      } else {
        alert("Your browser does not support audio recording.");
      }
    }

    // Function to send audio to the backend using AJAX
    function sendAudioToBackend(audioFile) {
      const formData = new FormData();
      formData.append("audio", audioFile);

      $.ajax({
        url: "/upload_audio",  // Replace with your backend endpoint
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
          console.log("Audio file successfully uploaded to the backend.");
        },
        error: function(err) {
          console.error("Error uploading audio: ", err);
        }
      });
    }

    // Function to delete the recorded file from the browser
    function deleteRecording() {
      audioPlayer.style.display = 'none';
      deleteButton.style.display = 'none';
      audioPlayer.src = '';  // Clear the audio source
    }

    // Event listeners
    recordButton.addEventListener("click", startRecording);
    deleteButton.addEventListener("click", deleteRecording);
  </script>
</body>
</html>

