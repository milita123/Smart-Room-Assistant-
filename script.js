function fetchDetections() {
    fetch('/detections')
        .then(response => response.json())
        .then(data => {
            const detectionsDiv = document.getElementById('detections');
            if (data.length > 0) {
                detectionsDiv.innerHTML = 'Detected: ' + data.join(', ');
            } else {
                detectionsDiv.innerHTML = 'No objects detected.';
            }
        });
}

// Fetch every 1 second
setInterval(fetchDetections, 1000);