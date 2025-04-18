// Main JavaScript file for the Plant Disease Detection App

document.addEventListener('DOMContentLoaded', function () {
    // File upload elements
    const uploadForm = document.getElementById('upload-form');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const uploadButton = document.getElementById('upload-button');
    const resultContainer = document.getElementById('result-container');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');

    // Chat elements
    const chatContainer = document.getElementById('chat-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatLoader = document.getElementById('chat-loader');

    // Upload area logic
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFileSelect(fileInput.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            showError('Please select an image file (PNG, JPG, JPEG).');
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreviewContainer.classList.remove('d-none');
            uploadButton.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', function (e) {
            e.preventDefault();

            if (!fileInput.files.length) {
                showError('Please select an image to upload.');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            loader.style.display = 'block';
            resultContainer.classList.add('d-none');
            errorMessage.classList.add('d-none');
            uploadButton.disabled = true;

            fetch('/api/upload', {
                method: 'POST',
                body: formData,
            })
                .then((response) => {
                    if (!response.ok) {
                        return response.json().then((data) => {
                            throw new Error(data.error || 'Error uploading image');
                        });
                    }
                    return response.json();
                })
                .then((data) => {
                    document.getElementById('prediction-name').textContent = formatDiseaseName(data.prediction);
                    document.getElementById('confidence-value').textContent = (data.confidence * 100).toFixed(2) + '%';

                    const chatButton = document.getElementById('chat-button');
                    if (chatButton) {
                        chatButton.setAttribute('href', `/chat?disease_id=${data.id}`);
                    }

                    const confidenceBadge = document.getElementById('confidence-badge');
                    if (confidenceBadge) {
                        if (data.confidence > 0.8) {
                            confidenceBadge.className = 'badge bg-success';
                        } else if (data.confidence > 0.6) {
                            confidenceBadge.className = 'badge bg-warning text-dark';
                        } else {
                            confidenceBadge.className = 'badge bg-danger';
                        }
                    }

                    resultContainer.classList.remove('d-none');
                })
                .catch((error) => {
                    showError(error.message);
                })
                .finally(() => {
                    loader.style.display = 'none';
                    uploadButton.disabled = false;
                });
        });
    }

    // Chatbot logic using Gemini
    if (chatForm && chatContainer) {
        chatForm.addEventListener('submit', function (e) {
            e.preventDefault();

            const message = userInput.value.trim();
            if (!message) return;

            addMessageToChat('user', message);
            userInput.value = '';

            const diseaseName = document.getElementById('disease-name')?.textContent || '';

            chatLoader.classList.remove('d-none');
            sendButton.disabled = true;

            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionStorage.getItem('session_id') || null,
                    disease: diseaseName
                }),
            })
                .then((response) => {
                    if (!response.ok) {
                        return response.json().then((data) => {
                            throw new Error(data.error || 'Error getting Gemini response');
                        });
                    }
                    return response.json();
                })
                .then((data) => {
                    if (data.session_id) {
                        sessionStorage.setItem('session_id', data.session_id);
                    }
                    addMessageToChat('bot', data.response);
                })
                .catch((error) => {
                    addMessageToChat('bot', 'Sorry, I encountered an error: ' + error.message);
                })
                .finally(() => {
                    chatLoader.classList.add('d-none');
                    sendButton.disabled = false;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                });
        });
    }

    function addMessageToChat(sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender === 'user' ? 'user-message' : 'bot-message'}`;

        if (sender === 'bot') {
            content = content
                .replace(/# (.*?)(\n|$)/g, '<h4>$1</h4>')
                .replace(/## (.*?)(\n|$)/g, '<h5>$1</h5>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/- (.*?)(\n|$)/g, '<li>$1</li>')
                .replace(/\n\n/g, '<br><br>')
                .replace(/\n/g, '<br>');

            if (content.includes('<li>')) {
                content = content.replace(/<li>(.*?)<\/li>/g, function (match) {
                    return '<ul>' + match + '</ul>';
                });
                content = content.replace(/<\/ul><ul>/g, '');
            }
        }

        messageDiv.innerHTML = content;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
    }

    function formatDiseaseName(name) {
        return name.replace(/_/g, ' ');
    }

    // Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
