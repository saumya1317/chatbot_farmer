import os
import os
import google.generativeai as genai

# Configure the Gemini API with your key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Choose a model (flash is lighter and free)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Send a text prompt
response = model.generate_content("Explain how photosynthesis works")

# Print the response
print(response.text)
