import streamlit as st
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load data and model using caching for performance
@st.cache_resource
def load_data():
    with open('medicine_model.pkl', 'rb') as f:
        data = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return data['df'], data['embeddings'], model

df, drug_embeddings, model = load_data()

# 2. UI Configuration
st.set_page_config(page_title="Pharmaceutical Search Engine", layout="wide")
st.title("AI-Powered Pharmaceutical Retrieval System")
st.markdown("---")

# User Input Section
query = st.text_input("Enter symptoms or medical condition:", placeholder="e.g., severe headache, muscle pain")

if query:
    # Vectorizing the user query
    query_vec = model.encode([query])
    
    # Calculating Semantic Similarity
    sim = cosine_similarity(query_vec, drug_embeddings).flatten()
    
    # Retrieving Top 3 matches
    indices = sim.argsort()[-3:][::-1]
    
    st.subheader("Search Results")
    
    for idx in indices:
        row = df.iloc[idx]
        score = sim[idx]
        
        # Professional Expander for each result
        with st.expander(f"Product: {row['Drug_Name']} (Confidence: {score:.2%})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Primary Indications:** {row['Uses']}")
                
                # Safety Protocol: Alcohol Interaction Analysis
                alcohol_status = str(row.get('Alcohol_Interaction', 'Information Not Available'))
                
                if any(word in alcohol_status.lower() for word in ['unsafe', 'caution', 'avoid']):
                    st.error(f"Safety Warning: {alcohol_status}")
                else:
                    st.info(f"Interaction Status: {alcohol_status}")
                
                # Check for Usage Instructions
                if 'How_To_Use' in df.columns:
                    st.write(f"**Administration Instructions:** {row['How_To_Use']}")
            
            with col2:
                # Commercial Details
                price = row.get('Selling_Price', 'N/A')
                st.metric("Market Price", f"INR {price}")
                st.write(f"**Manufacturer:** {row.get('Manufacturer', 'Data Unavailable')}")

# Footer Disclaimer
st.sidebar.markdown("---")
st.sidebar.write("**Disclaimer:** This is a technical demonstration for portfolio purposes. Please consult a qualified healthcare professional for medical advice.")