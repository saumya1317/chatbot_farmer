# Farmer's Assistant

A multilingual AI assistant for farmers that provides sustainable farming advice and plant disease detection.

## Features

- 🌾 Sustainable farming advice
- 🌿 Plant disease detection from images
- 📄 PDF document analysis
- 🌐 Multilingual support (English & Hindi)
- 🔊 Text-to-speech functionality
- 💬 WhatsApp-like chat interface

## Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your API key:
```
GOOGLE_API_KEY=your_api_key_here
```

5. Run the app locally:
```bash
streamlit run app.py
```

## Deployment

### Deploying to Streamlit Cloud

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Add your environment variables (GOOGLE_API_KEY)
7. Click "Deploy"

### Environment Variables

For deployment, you'll need to set these environment variables:
- `GOOGLE_API_KEY`: Your Google AI API key

## Usage

1. Upload farming-related PDF documents for detailed advice
2. Upload plant/leaf images for disease detection
3. Ask questions about farming practices
4. Switch between English and Hindi languages
5. Use text-to-speech to listen to responses
6. Copy responses to clipboard

## Contributing

Feel free to submit issues and enhancement requests! 