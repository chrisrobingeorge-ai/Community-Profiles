# Community Profile Extractor

This Streamlit app takes a Statistics Canada 2021 Census Profile CSV (community profile)
and extracts only the fields relevant to community planning for Growing Up Strong:
population, age, households, income, language, immigration, Indigenous identity,
mobility, commuting, etc.

## How to use (Streamlit Cloud)

1. Put these two files in the repo:
   - app.py
   - requirements.txt

2. Go to Streamlit Community Cloud and create a new app from this repo.

3. When the app is running in the browser:
   - Click "Browse files" to upload a Census Profile CSV you downloaded from Statistics Canada.
   - The app shows:
     - A raw preview of the file
     - A filtered report for only the fields we care about
     - Download buttons for:
       - Filtered CSV
       - Printable HTML report (you can Save as PDF in your browser)

No local install is required.