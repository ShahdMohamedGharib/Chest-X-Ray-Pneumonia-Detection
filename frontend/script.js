// JavaScript for Chest X-Ray Classification Frontend

const API_URL = 'http://127.0.0.1:8000';

// DOM Elements
const imageInput = document.getElementById('imageInput');
const uploadBox = document.getElementById('uploadBox');
const results = document.getElementById('results');
const loading = document.getElementById('loading');
const navBtns = document.querySelectorAll('.nav-btn');
const sections = document.querySelectorAll('.section');

// Navigation
navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetSection = btn.dataset.section;

        // Update active button
        navBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update active section
        sections.forEach(s => s.classList.remove('active'));
        document.getElementById(targetSection).classList.add('active');
    });
});

// Upload Box Click
uploadBox.addEventListener('click', () => {
    imageInput.click();
});

// Drag and Drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#1e40af';
    uploadBox.style.background = '#eff6ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#2563eb';
    uploadBox.style.background = '#ffffff';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#2563eb';
    uploadBox.style.background = '#ffffff';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageUpload(files[0]);
    }
});

// File Input Change
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageUpload(file);
    }
});

// Handle Image Upload
async function handleImageUpload(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('الرجاء رفع صورة فقط!');
        return;
    }

    // Show loading
    results.classList.add('hidden');
    loading.classList.remove('hidden');

    // Display original image
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('originalImage').src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send to API
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert('حدث خطأ أثناء التحليل. تأكد من تشغيل الـ API على المنفذ 8000');
        loading.classList.add('hidden');
    }
}

// Display Results
function displayResults(data) {
    // Hide loading
    loading.classList.add('hidden');

    // Show results
    results.classList.remove('hidden');

    // Display preprocessed image
    if (data.visualizations && data.visualizations.preprocessed) {
        document.getElementById('preprocessedImage').src = data.visualizations.preprocessed;
    }

    // Display Grad-CAM for all models
    const gradcamContainer = document.getElementById('gradcamContainer');
    gradcamContainer.innerHTML = '';
    
    const modelNames = {
        'improved_cnn': 'ImprovedCNN',
        'resnet18': 'ResNet18',
        'resnet50': 'ResNet50',
        'densenet121': 'DenseNet121',
        'efficientnet_b0': 'EfficientNet-B0'
    };
    
    if (data.visualizations && data.visualizations.gradcam) {
        for (const [modelKey, gradcamImage] of Object.entries(data.visualizations.gradcam)) {
            const modelName = modelNames[modelKey] || modelKey;
            const gradcamCard = document.createElement('div');
            gradcamCard.className = 'gradcam-card';
            
            if (gradcamImage) {
                gradcamCard.innerHTML = `
                    <h4>${modelName}</h4>
                    <img src="${gradcamImage}" alt="Grad-CAM ${modelName}">
                `;
            } else {
                gradcamCard.innerHTML = `
                    <h4>${modelName}</h4>
                    <p style="text-align: center; color: var(--text-muted); padding: 20px;">Grad-CAM غير متاح</p>
                `;
            }
            
            gradcamContainer.appendChild(gradcamCard);
        }
    }

    // Display Ensemble Prediction
    if (data.ensemble_prediction) {
        const ensemble = data.ensemble_prediction;

        document.getElementById('ensembleClass').textContent = ensemble.class;
        document.getElementById('ensembleClass').style.color =
            ensemble.class === 'NORMAL' ? '#10b981' : '#ef4444';

        document.getElementById('ensembleConfidence').textContent =
            `${(ensemble.confidence * 100).toFixed(1)}%`;

        // Display probabilities
        const probsHtml = `
            <div class="prob-item">
                <div class="prob-label">NORMAL</div>
                <div class="prob-value">${(ensemble.probabilities.NORMAL * 100).toFixed(1)}%</div>
            </div>
            <div class="prob-item">
                <div class="prob-label">PNEUMONIA</div>
                <div class="prob-value">${(ensemble.probabilities.PNEUMONIA * 100).toFixed(1)}%</div>
            </div>
        `;
        document.getElementById('ensembleProbabilities').innerHTML = probsHtml;
    }

    // Display Individual Predictions
    if (data.individual_predictions) {
        const container = document.getElementById('individualPredictions');
        container.innerHTML = '';

        const modelNames = {
            'improved_cnn': 'ImprovedCNN',
            'resnet18': 'ResNet18',
            'resnet50': 'ResNet50',
            'densenet121': 'DenseNet121',
            'efficientnet_b0': 'EfficientNet-B0'
        };

        for (const [modelKey, prediction] of Object.entries(data.individual_predictions)) {
            const modelName = modelNames[modelKey] || modelKey;
            const classColor = prediction.class === 'NORMAL' ? '#10b981' : '#ef4444';

            const modelCard = document.createElement('div');
            modelCard.className = 'model-prediction';
            modelCard.innerHTML = `
                <h4>${modelName}</h4>
                <div class="class" style="color: ${classColor}">${prediction.class}</div>
                <div class="confidence">${(prediction.confidence * 100).toFixed(1)}%</div>
                <div class="probs">
                    NORMAL: ${(prediction.probabilities.NORMAL * 100).toFixed(1)}% | 
                    PNEUMONIA: ${(prediction.probabilities.PNEUMONIA * 100).toFixed(1)}%
                </div>
            `;
            container.appendChild(modelCard);
        }
    }

    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Load Models Info
async function loadModelsInfo() {
    try {
        const response = await fetch(`${API_URL}/models-info`);
        const data = await response.json();
        
        // Update models section with dynamic info
        const modelsSection = document.getElementById('models');
        if (modelsSection && data.models) {
            // This can be used to dynamically update model cards if needed
            console.log('✓ Models info loaded:', data);
        }
    } catch (error) {
        console.warn('⚠️ Could not load models info:', error);
    }
}

// Check API Health on Load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        console.log('✓ API متصل:', data);
        
        // Load models info
        await loadModelsInfo();
    } catch (error) {
        console.warn('⚠️ API غير متصل. تأكد من تشغيل: uvicorn api.app:app --reload');
    }
});
