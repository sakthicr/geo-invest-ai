# 🗺️ Geo-Invest AI: Location Intelligence Specialist

### *A Multi-Modal Agentic RAG and Metaheuristic Framework for Low-Cost Retail Site Selection*

**Geo-Invest AI** is a premium location intelligence prototype designed to analyze natural language franchise proposals and suggest optimized geographic layout coordinates with dynamic, high-fidelity feasibility metrics. 

By combining **Pydantic AI (utilizing Gemini models)**, **OpenStreetMap Nominatim Geocoding**, **XGBoost tabular models**, and a **lightweight PyTorch EfficientNet feature simulator**, the framework helps franchise planners find the absolute best location nodes while avoiding competitive saturation and self-cannibalization.

---

## 🌟 Key Features

* **🤖 Pydantic AI Dynamic Agent Integration**: Uses Gemini models (`gemini-2.5-flash` ➔ `gemini-1.5-flash` ➔ `gemini-2.5-pro`) with a robust fallback system to handle spikes in model demand.
* **📍 Dynamic Keyless Geocoding (OSM Nominatim)**: Automatically parses location keywords from custom prompt inputs (e.g. *Perumalpuram, Tirunelveli* or *South Madurai*) and translates them into precise latitudes and longitudes.
* **🌾 Theme-Adaptive Recommendation Engine**: Classifies any city/town dynamically into 4 thematic profiles (**Historical/Temple**, **Coastal/Tourism**, **Tech/IT Hub**, and **Default**) and synthesizes culturally relevant, highly feasible business ideas (e.g. *traditional filter coffee hubs for temple routes, student co-working cafes for university hubs*).
* **💰 Transparent Profit Metric Labeling**: Every map popup and site recommendation card explicitly standardizes and labels estimated profits as **monthly** metrics (`/ mo`) to prevent abrupt visual metrics.
* **🎨 Premium Neon-Teal Visual Aesthetics**: The responsive UI is styled with glassmorphism panels, Google Fonts modern typography (Outfit/Inter), glowing neon indicators, and interactive help popovers mapping Leaflet tiles on a sleek **CartoDB Dark Matter** viewport.

---

## 🛠️ Tech Stack

* **Frontend**: HTML5, Javascript, TailwindCSS (Vanilla Grid utilities), Leaflet Maps (CartoDB Dark Matter), Lucide Icons.
* **Backend**: FastAPI, Uvicorn, Python 3.11.
* **Streamlit Framework**: Folium mapping with dynamic sidebar widgets.
* **Core Agent Engine**: Pydantic AI, Gemini models, XGBoost, Scikit-learn, PyTorch (Optional visual scouter).

---

## 🚀 How to Run Locally

### 1. Prerequisite Setup
Ensure you have Python 3.11 installed, clone the repository, and install the production dependencies:
```bash
# Clone the repository
git clone https://github.com/sakthicr/geo-invest-ai.git
cd geo-invest-ai

# Activate your virtual environment and install packages
# (Assuming standard Windows virtual environment setup)
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Run the FastAPI Web Server (Interactive Dashboard)
FastAPI serves the premium neon-teal GIS layout dashboard at your root path:
```bash
.\.venv\Scripts\python.exe -u server.py
```
Open your browser and navigate to **`http://127.0.0.1:8000`** to view the live dashboard!

### 3. Run the Streamlit Interface
If you prefer running the containerized Streamlit analytics panel:
```bash
.\.venv\Scripts\streamlit.exe run app.py
```
Navigate to **`http://localhost:8501`** in your browser.

---

## 🔑 API Key Security & Keyless Fallback

### How it Works:
* **No hardcoded keys**: Your personal API Key is never pushed to GitHub or saved on the server.
* **Visitor-Side Encryption**: Keys are entered in the client-side UI and saved privately inside the browser's `localStorage`, only sent in temporary memory during active POST queries.
* **Keyless Geocoding Failsafe**: If no key is entered (or the key is invalid), the backend **automatically triggers the high-fidelity OpenStreetMap Nominatim geocoder** and thematic recommender, adapting the map and suggested business names (e.g. `Tirunelveli Traditional Filter Coffee & Snack Lounge` centering coordinates precisely inside Tirunelveli) completely keyless!

---

## 🐳 Containerized Deployments

We provide a **`Dockerfile`** that supports dynamic port mapping, making it ready to deploy in 1 click on:
* **Streamlit Community Cloud** (100% Free - Main file: `app.py`)
* **Render** (Docker Web Service - Main command: `server:app`)
* **Railway** / **Google Cloud Run**
