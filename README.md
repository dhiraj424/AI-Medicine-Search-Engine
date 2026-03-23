**Pharma-Insight: AI Semantic Search Engine**

**Link** https://ai-medicine-search-engine-4o4phefifchq9wstzjwbk9.streamlit.app/

**Objective**

An NLP-based retrieval system that understands medical symptoms using Semantic Search instead of traditional keyword matching. Built for real-time pharmaceutical data analysis and patient safety.

**Technical Stack**

Model: Sentence-BERT (all-MiniLM-L6-v2) for text vectorization.

Logic: Cosine Similarity for ranking and retrieving relevant medications.

Data: Preprocessed pharmaceutical datasets using Pandas.

Deployment: Interactive dashboard built with Streamlit.

**Core Features**

Contextual Search: Matches natural language symptoms (e.g., "shaking with cold") to relevant drugs (e.g., Paracetamol).

Safety Protocol: Automated alerts for Alcohol Interactions and Common Side Effects.

Commercial Insights: Displays Market Price and Manufacturer details for every result.

Risk Management: High-visibility warnings for unsafe medication combinations.

**How to Run**

Install dependencies: pip install -r requirements.txt

Launch the application: streamlit run app.py

**Disclaimer**:
This is a technical portfolio project for demonstration purposes. It is not a clinical tool. Consult a licensed physician for medical advice.
