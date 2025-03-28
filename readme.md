# Code Summarizer


## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository_link>
cd <repository_name>
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


