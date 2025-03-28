import os
import json
from google import genai
from google.genai import types

def summary(code_text):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    prompt = """As an AI language model, your task is to analyze machine learning scripts and convert them into a structured JSON format. For each script provided, you should:

1. **Identify Key Phases:** Determine the main stages in the code, such as "Data Loading," "Preprocessing," "Model Training," and "Evaluation."
2. **Extract Code Snippets:** For each identified phase, extract relevant code snippets. If a phase is complex, divide it into sub-phases accordingly.
3. **Provide Non-Technical Descriptions:** Offer simple explanations for each phase.
4. **Output Strictly JSON:** Return only the JSON breakdown in the specified format.

**JSON Structure:**
{
  "phases": [
    {
      "phase": "Phase Name",
      "description": "Brief explanation of the phase.",
      "code": [
        "Relevant code snippet 1",
        "Relevant code snippet 2"
      ],
      "sub_phases": [
        {
          "sub_phase": "Sub-Phase Name",
          "description": "Brief explanation of the sub-phase.",
          "code": [
            "Relevant code snippet 1",
            "Relevant code snippet 2"
          ]
        }
      ]
    }
  ]
}
}
Ensure that:
- The response strictly follows the above JSON structure.
- No additional text appears before or after the JSON output.
- The response should be a valid JSON that can be parsed directly.
"""

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
        types.Content(role="user", parts=[types.Part.from_text(text=code_text)])
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    # Extract the raw JSON (removing any markdown formatting if present)
    raw_text = response.text.strip()

    # If wrapped in triple backticks, remove them
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:-3].strip()

    # Ensure it's valid JSON before returning
    try:
        json_data = json.loads(raw_text)
        return json_data
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from Gemini"}

