# AutoML Agent

AutoML Agent is a lightweight Streamlit application that helps users upload a CSV file, detect the likely machine learning problem, clean the data, and train a simple baseline model.

## Features
- Upload CSV datasets
- Detect classification vs. regression automatically
- Clean missing values and normalize basic data types
- Train a baseline model and save it to the models directory
- View a summary of the analysis and training results in the UI

## Run locally
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## Notes
This project is intentionally lightweight and uses scikit-learn so it can run in a local development environment without needing a full MLOps stack.
