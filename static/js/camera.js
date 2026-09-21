// Camera and Face Recognition functionality
let video, canvas, ctx;
let stream = null;
let capturedImages = [];
let selectedStudentId = null;

document.addEventListener('DOMContentLoaded', function () {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    if (canvas) {
        ctx = canvas.getContext('2d');
    }

    const startBtn = document.getElementById('startCamera');
    const stopBtn = document.getElementById('stopCamera');
    const captureBtn = document.getElementById('capturePhoto');
    const recognizeBtn = document.getElementById('recognizeFace');
    const trainBtn = document.getElementById('trainModel');
    const studentSelect = document.getElementById('studentSelect');

    // Load students
    loadStudents();

    // Event listeners
    if (startBtn) {
        startBtn.addEventListener('click', startCamera);
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', stopCamera);
    }
    if (captureBtn) {
        captureBtn.addEventListener('click', captureForTraining);
    }
    if (recognizeBtn) {
        recognizeBtn.addEventListener('click', recognizeFace);
    }
    if (trainBtn) {
        trainBtn.addEventListener('click', trainModel);
    }
    if (studentSelect) {
        studentSelect.addEventListener('change', function () {
            selectedStudentId = this.value;
            capturedImages = [];
            updateCaptureProgress(0);
        });
    }
});

let CAPTURE_LIMIT = 100;

async function loadStudents() {
    // Fetch target count from server
    try {
        const configResp = await fetch('/face/config');
        const configData = await configResp.json();
        CAPTURE_LIMIT = configData.face_samples_count || 100;
        console.log(`DEBUG: Capture limit set to ${CAPTURE_LIMIT} from server`);
        
        // Update instructions UI if it exists
        const instructions = document.querySelectorAll('aside ul li');
        instructions.forEach(li => {
            if (li.textContent.includes('photos')) {
                li.innerHTML = `Select a student and click <strong>Capture</strong> to take ${CAPTURE_LIMIT} photos.`;
            }
        });
        const progressHeader = document.querySelector('.flex-between span#captureCount');
        if (progressHeader && progressHeader.nextSibling) {
            progressHeader.nextSibling.textContent = `/${CAPTURE_LIMIT}`;
        }
    } catch (e) {
        console.error("Failed to fetch fresh config:", e);
    }
    try {
        const response = await fetch('/students/api/all');
        const students = await response.json();

        const select = document.getElementById('studentSelect');
        if (select) {
            students.forEach(student => {
                const option = document.createElement('option');
                option.value = student.id;
                option.textContent = `${student.name} (${student.student_id})`;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading students:', error);
    }
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' },
            audio: false
        });

        video.srcObject = stream;

        // Update UI
        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('stopCamera').style.display = 'inline-flex';
        document.getElementById('capturePhoto').style.display = 'inline-flex';
        document.getElementById('recognizeFace').style.display = 'inline-flex';

        showNotification('Camera started successfully', 'success');
    } catch (error) {
        console.error('Error accessing camera:', error);
        showNotification('Error accessing camera. Please check permissions.', 'error');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }

    // Update UI
    document.getElementById('startCamera').style.display = 'inline-flex';
    document.getElementById('stopCamera').style.display = 'none';
    document.getElementById('capturePhoto').style.display = 'none';
    document.getElementById('recognizeFace').style.display = 'none';

    showNotification('Camera stopped', 'success');
}

async function captureForTraining() {
    if (!selectedStudentId) {
        showNotification('Please select a student first', 'error');
        return;
    }

    if (capturedImages.length >= CAPTURE_LIMIT) {
        showNotification(`Maximum ${CAPTURE_LIMIT} photos captured`, 'warning');
        return;
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    ctx.drawImage(video, 0, 0);

    // Get image data
    const imageData = canvas.toDataURL('image/jpeg');
    capturedImages.push(imageData);

    // Update progress
    updateCaptureProgress(capturedImages.length);

    // Auto-capture multiple photos
    if (capturedImages.length < CAPTURE_LIMIT) {
        setTimeout(captureForTraining, 200); // Capture every 200ms
    } else {
        // Send to server
        await savePhotos();
    }
}

async function savePhotos() {
    try {
        const response = await fetch(`/face/capture-photos/${selectedStudentId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                images: capturedImages
            })
        });

        const result = await response.json();

        if (result.success) {
            showNotification(`Successfully captured ${result.count} photos!`, 'success');
            capturedImages = [];
            updateCaptureProgress(0);
        } else {
            showNotification(result.message || 'Error saving photos', 'error');
        }
    } catch (error) {
        console.error('Error saving photos:', error);
        showNotification('Error saving photos', 'error');
    }
}

function updateCaptureProgress(count) {
    const progressDiv = document.getElementById('captureProgress');
    const countSpan = document.getElementById('captureCount');
    const progressBar = document.getElementById('progressBar');

    if (count > 0) {
        progressDiv.style.display = 'block';
        countSpan.textContent = count;
        progressBar.style.width = (count / CAPTURE_LIMIT * 100) + '%';
    } else {
        progressDiv.style.display = 'none';
    }
}

async function recognizeFace() {
    const btn = document.getElementById('recognizeFace');
    const originalText = btn.innerHTML;

    // Set loading state
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner" style="width:20px; height:20px; border-width:2px; margin:0;"></div> Thinking...';

    // Set canvas size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Capture current frame
    ctx.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg');

    try {
        const response = await fetch('/face/recognize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData
            })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('Recognition result:', result);

        const resultDiv = document.getElementById('recognitionResult');
        const contentDiv = document.getElementById('resultContent');

        if (!resultDiv || !contentDiv) {
            console.error('UI elements not found');
            return;
        }

        if (result.success && result.students && result.students.length > 0) {
            let html = '';
            let names = [];
            result.students.forEach(student => {
                names.push(student.name);
                html += `
                    <div style="margin-top: 0.5rem; padding: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 0.5rem;">
                        <strong>${student.name}</strong> (${student.student_id})<br>
                        Roll No: ${student.roll_no} | Dept: ${student.department}<br>
                        Confidence: ${student.confidence}%<br>
                        ${student.attendance_marked ? '✅ Attendance marked' : '⚠️ ' + (student.message || 'Already marked')}
                    </div>
                `;
            });
            contentDiv.innerHTML = html;
            resultDiv.className = 'alert alert-success';
            resultDiv.style.display = 'block';

            // ADD SUCCESS POPUP
            console.log("Recognition Success:", names.join(', '));
            showNotification(`Recognized: ${names.join(', ')}`, 'success');
        } else if (result.is_spoof) {
            const msg = result.message || '⚠️ Spoof attempt detected! Please present a live human face.';
            contentDiv.innerHTML = `<div style="font-weight: 600; color: #f87171; padding: 0.25rem 0;">${msg}</div>`;
            resultDiv.className = 'alert alert-error';
            resultDiv.style.display = 'block';

            console.log("Spoof Attempt Blocked:", msg);
            showNotification(msg, 'error');
        } else {
            const msg = result.message || 'No face recognized. Try better lighting.';
            contentDiv.innerHTML = msg;
            resultDiv.className = 'alert alert-warning';
            resultDiv.style.display = 'block';

            // ADD WARNING POPUP
            console.log("Recognition Failed:", msg);
            showNotification(msg, 'warning');
        }
    } catch (error) {
        console.error('Detailed Recognition Error:', error);
        showNotification('Recognition Error: ' + error.message, 'error');
    } finally {
        // Reset button state
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

async function trainModel() {
    const statusDiv = document.getElementById('trainingStatus');
    statusDiv.innerHTML = '<div class="spinner"></div> Training model... This may take a moment.';

    try {
        const response = await fetch('/face/train-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                student_id: selectedStudentId
            })
        });

        const result = await response.json();

        if (result.success) {
            statusDiv.innerHTML = `<div class="alert alert-success">${result.message}</div>`;
            showNotification('Model trained successfully!', 'success');
        } else {
            statusDiv.innerHTML = `<div class="alert alert-error">${result.message}</div>`;
            showNotification(result.message, 'error');
        }
    } catch (error) {
        console.error('Error training model:', error);
        statusDiv.innerHTML = '<div class="alert alert-error">Error training model</div>';
        showNotification('Error training model', 'error');
    }
}
