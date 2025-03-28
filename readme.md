# Code Summarizer


## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/adityadaharwal963/Code_Summarizer_GenAI.git
cd Code_Summarizer_GenAI
```

### 2. Create a New Conda Environment
```bash
conda env create -f environment.yml
```

### 3. Verify Installation
```bash
conda info --envs
```

### 4. Activate the Environment
```bash
conda activate my_env
```

### 5. Update the Environment (if needed)
```bash
conda env update -f environment.yml
```

---

### Using venv (Alternative)

#### 1. Create a Virtual Environment
```bash
python -m venv my_env
```

#### 2. Activate the Virtual Environment
- **Linux/Mac:**
```bash
source my_env/bin/activate
```
- **Windows:**
```bash
my_env\Scripts\activate
```

#### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

---
---

## API Setup

### 6. Get Google Gemini API Key
- Obtain the API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 7. Configure Environment Variables
- Create a `.env` file in the project root.
- Copy the contents of `sample-env.txt` and paste them into `.env`, updating the values as needed.

---

## Run the Application

### 8. Start the Flask Server
```bash
flask run
```
- The server will be live at `http://127.0.0.1:5000`.

---

## API Endpoints

### 1. `/api/summary` (POST)

- **Description:**  
  Generates machine learning (ML) code and returns the phases with descriptions and code.

- **Request Body:**
```json
{
  "prompt": "string"
}
```

- **Response:**
```json
{
  "phases": [
    {
      "phase": "Phase Name",
      "description": "Description of the phase",
      "code": "Generated code"
    }
  ]
}
```

---

### 2. `/api/generate` (POST)

- **Description:**  
  Generates output based on the provided input.

- **Request Body:**
```json
{
  "prompt": "string"
}
```

- **Response:**
```json
{
  "output": "Generated result based on input"
}
```

---

### 3. `/api/models` (GET)

- **Description:**  
  Returns available models.

- **Request Body:**  
  (No body required for `GET` requests)

- **Response:**
```json
{
  "models": [
    "model_1",
    "model_2",
    "model_3"
  ]
}
```


