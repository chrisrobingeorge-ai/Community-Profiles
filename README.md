# Community Profiles (Statistics Canada) — Streamlit App
Generate **narratively written community profiles** using **Statistics Canada** CSV files.  
Built with **Streamlit**, **Pandas**, and **Jinja2**, with export to **Markdown** and **Word (.docx)**.

## ✨ Features
- Reads one master CSV or multiple CSVs from `data/raw/`
- Choose a community by name or enter a **GeoUID**
- 24-section narrative aligned to common Census Profile topics
- Mapping-driven: adjust `metadata/fields.yaml` to your exact column headers
- Download **.md** and **.docx**
- “Data Health Check” shows mapping coverage by section

## 🗂 Project structure
``
