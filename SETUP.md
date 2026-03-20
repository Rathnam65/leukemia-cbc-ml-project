# Leukemia CBC ML Project - Setup Guide

## Prerequisites
- Python 3.8+
- pip package manager

## Installation

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. (Optional) Setup OCR for Scanned PDFs
If you want to use OCR for scanned PDF documents:

**Windows:**
```bash
pip install pdf2image
# Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Add Tesseract to PATH or update pytesseract path in app.py
```

**macOS:**
```bash
brew install tesseract
pip install pdf2image
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils
pip install pdf2image
```

## Running the Project

### 1. Train the Model
```bash
python train_model.py
```
This will:
- Load the CBC dataset
- Train a Random Forest classifier
- Save the model to `model/leukemia_model.pkl`

### 2. Start the Flask Backend
```bash
python app.py
```
The API will be available at: `http://127.0.0.1:5000`

### 3. Open the Frontend
Open `../frontend/index.html` in your web browser

## API Endpoints

### Manual Input (JSON)
**POST** `/predict`
```json
{
  "WBC": 63987,
  "RBC": 15.36,
  "Hb": 22.5,
  "Platelets": 2493
}
```

### File Upload
**POST** `/predict` (multipart/form-data)
- Supports: CSV, PDF, DOCX
- Key: `files` (multiple files allowed)

## Supported File Formats

### CSV
Must contain columns: `WBC`, `RBC`, `Hb`, `Platelets`

Example:
```csv
WBC,RBC,Hb,Platelets
63987,15.36,22.5,2493
45000,4.5,14.2,250000
```

### PDF
Automatically extracts CBC values from text or tables.
Works with:
- Standard lab PDFs with readable text
- Tables containing CBC data
- Multiple formats of field names (WBC, White Blood Cells, etc.)
- Scanned PDFs (with OCR enabled)

### DOCX
Extracts text and searches for CBC values in various formats

## Field Name Variations Supported

**WBC:**
- WBC, WBC Count, White Blood Cells, WBC_Count

**RBC:**
- RBC, RBC Count, Red Blood Cells, RBC_Count

**Hemoglobin:**
- Hb, HB, Hemoglobin, Hemoglobin_Level

**Platelets:**
- Platelets, PLT, Platelet Count, Platelet_Count

## Expected Value Ranges

| Parameter | Normal Range | Unit |
|-----------|-------------|------|
| WBC | 4.5K - 11K | cells/µL |
| RBC | 4.5 - 5.9 | millions/µL |
| Hemoglobin | 12 - 17.5 | g/dL |
| Platelets | 150K - 400K | cells/µL |

## Troubleshooting

### PDF extraction not working
1. Ensure PDF is readable (not corrupted)
2. Check if PDF contains text (not just images)
3. Enable OCR for scanned PDFs (see setup above)
4. Try CSV format as alternative

### Model not found error
- Make sure to run `train_model.py` first
- Check that `model/leukemia_model.pkl` exists

### Connection refused
- Ensure Flask app is running on port 5000
- Check firewall settings
- Verify no other service is using port 5000

### OCR not working
- Install Tesseract for your OS (see Prerequisites)
- Update pytesseract path if needed in app.py

## Notes
- The model predictions should be validated by medical professionals
- This tool is for screening purposes only, not a diagnostic tool
- Always consult qualified healthcare providers for medical decisions
