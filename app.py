# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime
import base64
import re
import pickle

# Import necessary libraries for embeddings, clustering, and summarization
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from hdbscan import HDBSCAN
import nltk
nltk.download('punkt', download_dir=r'C:\Users\JLBKMVE\AppData\Roaming\nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# For summarization
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize session state variables if needed
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None
if 'topic_model' not in st.session_state:
    st.session_state['topic_model'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Sidebar - Data Upload
st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_excel(uploaded_file)
    st.session_state['df'] = df.copy()
    st.write("Data preview:")
    st.write(df.head())
else:
    st.warning("Please upload an Excel file to proceed.")

# Create tabs for each step
tab1, tab2, tab3 = st.tabs(["Semantic Search", "Clustering", "Summarization"])

# Device configuration for PyTorch
device = 'cpu'

# Semantic Search Tab
with tab1:
    st.header("Semantic Search")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        # Prepare data

        # Handle 'Title' and 'Description' columns optionally
        if 'Title' in df.columns:
            df['Title'] = df['Title'].fillna('')
        else:
            df['Title'] = ''
        if 'Description' in df.columns:
            df['Description'] = df['Description'].fillna('')
        else:
            df['Description'] = ''
        df['text_for_embedding'] = df['Title'] + " " + df['Description']

        # Load or generate embeddings
        embeddings_file = 'embeddings.pkl'
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            st.write("Generating embeddings...")
            model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
            embeddings = model.encode(df['text_for_embedding'].tolist(), show_progress_bar=True, device=device)
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
        st.session_state['embeddings'] = embeddings

        # Input query
        query = st.text_input("Enter your search query:")
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

        if st.button("Search"):
            if query:
                # Perform semantic search
                model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
                query_embedding = model.encode([query], device=device)
                similarities = cosine_similarity(query_embedding, embeddings)
                above_threshold_indices = np.where(similarities[0] > similarity_threshold)[0]
                results = df.iloc[above_threshold_indices].copy()
                results['similarity_score'] = similarities[0][above_threshold_indices]
                st.write("Search Results:")
                # Dynamically select columns to display
                columns_to_display = []
                if 'Title' in results.columns:
                    columns_to_display.append('Title')
                if 'Description' in results.columns:
                    columns_to_display.append('Description')
                columns_to_display.append('similarity_score')
                st.write(results[columns_to_display])
            else:
                st.warning("Please enter a query to search.")
    else:
        st.warning("Please upload a data file to proceed.")

# Clustering Tab
with tab2:
    st.header("Clustering")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        # Prepare text for clustering
        # Handle 'Title' and 'Description' columns optionally
        if 'Title' in df.columns:
            df['Title'] = df['Title'].fillna('')
        else:
            df['Title'] = ''
        if 'Description' in df.columns:
            df['Description'] = df['Description'].fillna('')
        else:
            df['Description'] = ''
        df['text'] = df['Title'] + ' ' + df['Description']
        texts = df['text'].tolist()

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        texts_cleaned = []
        for text in texts:
            word_tokens = word_tokenize(text, language='english')
            filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
            texts_cleaned.append(filtered_text)

        # Generate embeddings for clustering
        st.write("Generating embeddings for clustering...")
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
        embeddings = sentence_model.encode(texts_cleaned, show_progress_bar=True, device=device)

        # Perform clustering
        if st.button("Run Clustering"):
            st.write("Performing clustering...")
            hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
            topic_model = BERTopic(embedding_model=sentence_model, hdbscan_model=hdbscan_model)
            topics, _ = topic_model.fit_transform(texts_cleaned, embeddings)
            df['Topic'] = topics
            st.session_state['topic_model'] = topic_model
            st.session_state['df'] = df.copy()
            st.write("Clustering Results:")
            # Dynamically select columns to display
            columns_to_display = []
            if 'Title' in df.columns:
                columns_to_display.append('Title')
            if 'Description' in df.columns:
                columns_to_display.append('Description')
            columns_to_display.append('Topic')
            st.write(df[columns_to_display])

            # Optionally display visualizations
            st.write("Visualizing Topics...")

            st.subheader("Intertopic Distance Map")
            fig1 = topic_model.visualize_topics()
            st.plotly_chart(fig1)

            st.subheader("Topic Document Visualization")
            fig2 = topic_model.visualize_documents(texts_cleaned, embeddings=embeddings)
            st.plotly_chart(fig2)

            st.subheader("Topic Hierarchy Visualization")
            fig3 = topic_model.visualize_hierarchy()
            st.plotly_chart(fig3)

            # Attempt to visualize topics per class
            st.subheader("Topics per Class Visualization")
            available_columns = df.columns.tolist()
            if len(available_columns) > 0:
                class_column = st.selectbox("Select a column to use as classes for Topics per Class visualization:", available_columns)
                if class_column:
                    try:
                        classes = df[class_column].astype(str).tolist()
                        topics_per_class = topic_model.topics_per_class(texts_cleaned, classes=classes)
                        fig4 = topic_model.visualize_topics_per_class(topics_per_class)
                        st.plotly_chart(fig4)
                    except Exception as e:
                        st.warning("Could not generate Topics per Class visualization.")
            else:
                st.warning("No columns found in data to use for Topics per Class visualization.")
    else:
        st.warning("Please upload a data file to proceed.")

# Summarization Tab
with tab3:
    st.header("Summarization")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        if 'Topic' in df.columns:
            # Select Impact Area and Objectives
            impact_areas = ['Nutrition', 'Poverty Reduction', 'Gender', 'Climate Change', 'Environmental Biodiversity']
            impact_area = st.selectbox("Select Impact Area", impact_areas)

            # Define objectives for each impact area
            impact_area_objectives = {
                'Nutrition': "Your objectives related to Nutrition...",
                'Poverty Reduction': "Your objectives related to Poverty Reduction...",
                'Gender': "Your objectives related to Gender...",
                'Climate Change': "Your objectives related to Climate Change...",
                'Environmental Biodiversity': "Your objectives related to Environmental Biodiversity..."
            }
            objectives = impact_area_objectives.get(impact_area, "")

            if st.button("Generate Summaries"):
                # Define system prompt
                system_prompt = """
You are an expert summarizer skilled in creating concise and relevant summaries of lengthy texts. Your goal is to produce summaries that are aligned with specific objectives provided for each task. When summarizing, focus on the following guidelines:

1. **Clarity and Precision**: Ensure the summary is clear, precise, and easy to understand.
2. **Objective Alignment**: Tailor the summary to address the specific objectives provided in the prompt. Highlight key points and insights related to these objectives.
3. **Coherence and Flow**: Maintain a logical flow and coherence in the summary, even if the original text is divided into multiple sections.
4. **Conciseness**: Strive to be concise, including only the most relevant information to meet the objectives.

For each task, you will receive a text along with specific objectives. Summarize the text according to these guidelines, ensuring that the objectives are explicitly addressed in your summary.
"""

                # Initialize LLM
                openai_api_key = os.environ.get('OPENAI_API_KEY')
                if not openai_api_key:
                    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                else:
                    llm = ChatOpenAI(api_key=openai_api_key, model_name='gpt-4')
                    summaries = []
                    grouped = df.groupby('Topic')
                    progress_bar = st.progress(0)
                    total_topics = len(grouped)
                    for idx, (topic, group) in enumerate(grouped):
                        all_text = " ".join(group['text'].tolist())
                        # Prepare prompts
                        user_prompt = f"**Objectives**: {objectives}\n**Text to summarize**: {all_text}"
                        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                        human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                        chain = LLMChain(llm=llm, prompt=chat_prompt)
                        response = chain.run(user_prompt=user_prompt)
                        summary = response.strip()
                        summaries.append({'Topic': topic, 'Summary': summary})
                        progress_bar.progress((idx + 1) / total_topics)
                    progress_bar.empty()
                    summary_df = pd.DataFrame(summaries)
                    st.write("Summaries:")
                    st.write(summary_df)
                    # Optionally save summaries
                    csv = summary_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="summaries.csv">Download Summaries CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Please perform clustering first to generate topics.")
    else:
        st.warning("Please upload a data file to proceed.")
