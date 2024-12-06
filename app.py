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
import concurrent.futures  # Import for parallel processing

# Import necessary libraries for embeddings, clustering, and summarization
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from hdbscan import HDBSCAN
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# For summarization
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Device configuration for PyTorch
device = 'cpu'

# Initialize NLTK resources
def init_nltk_resources():
    nltk_data_dir = r'C:\Users\JLBKMVE\AppData\Roaming\nltk_data'
    nltk.data.path.append(nltk_data_dir)
    # Check if 'stopwords' is downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_dir)
    # Check if 'punkt' is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)

init_nltk_resources()

st.sidebar.title("Data Selection")
dataset_option = st.sidebar.selectbox('Select Dataset', ('PRMS 2022+2023 QAed', 'Upload my dataset'))

@st.cache_data
def load_default_dataset(default_dataset_path):
    if os.path.exists(default_dataset_path):
        df = pd.read_excel(default_dataset_path)
        return df
    else:
        st.error("Default dataset not found. Please ensure the file exists in the 'input' directory.")
        return None

@st.cache_data
def load_uploaded_dataset(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

def generate_embeddings(texts, model):
    with st.spinner('Calculating embeddings...'):
        embeddings = model.encode(texts, show_progress_bar=True, device=device)
    return embeddings

@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    return model

def load_or_compute_embeddings(df, using_default_dataset, uploaded_file_name=None):
    # Determine the embeddings filename
    embeddings_dir = os.path.dirname(__file__)
    if using_default_dataset:
        embeddings_file = os.path.join(embeddings_dir, 'PRMS_2022_2023_QAed.pkl')
    else:
        # For custom datasets, use timestamp and filename to ensure uniqueness
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use uploaded file name (without extension) in embeddings file name
        base_name = os.path.splitext(uploaded_file_name)[0] if uploaded_file_name else "custom_dataset"
        embeddings_file = os.path.join(embeddings_dir, f"{base_name}_{timestamp_str}.pkl")

    texts = (df.get('Title', '').fillna('') + " " + df.get('Description', '').fillna('')).tolist()

    # Check if embeddings are already computed and stored in session
    # If they are for the same dataset, we can reuse them
    if 'embeddings' in st.session_state and 'embeddings_file' in st.session_state:
        # If the session embeddings file name matches the one we would use for this dataset,
        # it means we've already computed and loaded the embeddings for this dataset.
        if st.session_state['embeddings_file'] == embeddings_file and len(st.session_state['embeddings']) == len(texts):
            return st.session_state['embeddings'], embeddings_file

    # If file exists and matches current dataset size, load it
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        if len(embeddings) == len(texts):
            st.write("Loading pre-calculated embeddings...")
            st.session_state['embeddings'] = embeddings
            st.session_state['embeddings_file'] = embeddings_file
            return embeddings, embeddings_file
        else:
            st.write("Pre-calculated embeddings do not match current data. Regenerating...")

    # Otherwise, generate embeddings
    st.write("Generating embeddings...")
    model = get_embedding_model()
    embeddings = generate_embeddings(texts, model)
    # Save embeddings
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    st.session_state['embeddings'] = embeddings
    st.session_state['embeddings_file'] = embeddings_file
    return embeddings, embeddings_file

if dataset_option == 'PRMS 2022+2023 QAed':
    default_dataset_path = os.path.join(os.path.dirname(__file__), 'input', 'export_data_table_results_20240312_160222CET.xlsx')
    df = load_default_dataset(default_dataset_path)
    if df is not None:
        st.session_state['df'] = df.copy()
        st.session_state['using_default_dataset'] = True
        st.write("Using default dataset:")

        # Load filter options
        filters_dir = os.path.join(os.path.dirname(__file__), 'filters')
        with open(os.path.join(filters_dir, 'regions.txt'), 'r') as f:
            regions_options = [line.strip() for line in f.readlines()]
        with open(os.path.join(filters_dir, 'countries.txt'), 'r') as f:
            countries_options = [line.strip() for line in f.readlines()]
        with open(os.path.join(filters_dir, 'centers.txt'), 'r') as f:
            centers_options = [line.strip() for line in f.readlines()]
        with open(os.path.join(filters_dir, 'impact_area.txt'), 'r') as f:
            impact_area_options = [line.strip() for line in f.readlines()]
        with open(os.path.join(filters_dir, 'sdg_target.txt'), 'r') as f:
            sdg_target_options = [line.strip() for line in f.readlines()]

        # Initialize filter selections
        if 'selected_regions' not in st.session_state:
            st.session_state['selected_regions'] = []
        if 'selected_countries' not in st.session_state:
            st.session_state['selected_countries'] = []
        if 'selected_centers' not in st.session_state:
            st.session_state['selected_centers'] = []
        if 'selected_impact_area' not in st.session_state:
            st.session_state['selected_impact_area'] = []
        if 'selected_sdg_targets' not in st.session_state:
            st.session_state['selected_sdg_targets'] = []

        col1, col2 = st.columns(2)
        with col1:
            st.session_state['selected_regions'] = st.multiselect("Regions", regions_options, default=st.session_state['selected_regions'])
        with col2:
            st.session_state['selected_countries'] = st.multiselect("Countries", countries_options, default=st.session_state['selected_countries'])

        col3, col4 = st.columns(2)
        with col3:
            st.session_state['selected_centers'] = st.multiselect("Primary Center", centers_options, default=st.session_state['selected_centers'])
        with col4:
            st.session_state['selected_impact_area'] = st.multiselect("Impact Area Target(s)", impact_area_options, default=st.session_state['selected_impact_area'])

        col5 = st.columns(1)
        with col5[0]:
            st.session_state['selected_sdg_targets'] = st.multiselect("SDG Target(s)", sdg_target_options, default=st.session_state['selected_sdg_targets'])

        filtered_df = df.copy()
        if st.session_state['selected_regions']:
            filtered_df = filtered_df[filtered_df['Regions'].isin(st.session_state['selected_regions'])]
        if st.session_state['selected_countries']:
            filtered_df = filtered_df[filtered_df['Countries'].isin(st.session_state['selected_countries'])]
        if st.session_state['selected_centers']:
            filtered_df = filtered_df[filtered_df['Primary center'].isin(st.session_state['selected_centers'])]
        if st.session_state['selected_impact_area']:
            filtered_df = filtered_df[filtered_df['Impact Area Target'].isin(st.session_state['selected_impact_area'])]
        if st.session_state['selected_sdg_targets']:
            filtered_df = filtered_df[filtered_df['SDG targets'].isin(st.session_state['selected_sdg_targets'])]

        st.session_state['filtered_df'] = filtered_df
        st.write(filtered_df.head())
    else:
        st.warning("Please ensure the default dataset exists in the 'input' directory.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = load_uploaded_dataset(uploaded_file)
        if df is not None:
            st.session_state['df'] = df.copy()
            st.session_state['using_default_dataset'] = False
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.write("Data preview:")
            st.write(df.head())
        else:
            st.warning("Failed to load the uploaded dataset.")
    else:
        st.warning("Please upload an Excel file to proceed.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Semantic Search", "Clustering", "Summarization"])

# Semantic Search Tab
with tab1:
    st.header("Semantic Search")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            df = st.session_state['df']  # full dataset
            filtered_df = st.session_state['filtered_df']

            # Ensure embeddings are loaded or computed once
            if 'embeddings' not in st.session_state:
                # Compute embeddings once dataset is loaded
                embeddings, embeddings_file = load_or_compute_embeddings(df, st.session_state.get('using_default_dataset', False), st.session_state.get('uploaded_file_name', None))
            else:
                embeddings = st.session_state['embeddings']

            # Input query
            query = st.text_input("Enter your search query:")
            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.35)

            # If user searches
            if st.button("Search"):
                if query:
                    model = get_embedding_model()
                    query_embedding = model.encode([query], device=device)
                    # Filtered_df is a subset of df, so we can get indices and slice embeddings
                    filtered_indices = filtered_df.index
                    filtered_embeddings = embeddings[filtered_indices]
                    similarities = cosine_similarity(query_embedding, filtered_embeddings)
                    above_threshold_indices = np.where(similarities[0] > similarity_threshold)[0]

                    if len(above_threshold_indices) == 0:
                        st.warning("No results found above the similarity threshold.")
                    else:
                        # Map back to filtered_df indices
                        selected_indices = filtered_indices[above_threshold_indices]
                        results = filtered_df.loc[selected_indices].copy()
                        results['similarity_score'] = similarities[0][above_threshold_indices]
                        results = results.sort_values(by='similarity_score', ascending=False)
                        st.session_state['search_results'] = results.copy()
                        st.write("Search Results:")
                        columns_to_display = [col for col in ['Title', 'Description'] if col in results.columns]
                        columns_to_display.append('similarity_score')
                        st.write(results[columns_to_display])
                else:
                    st.warning("Please enter a query to search.")
            else:
                # If we have previous search results, show them
                if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
                    st.write("Previous Search Results:")
                    columns_to_display = [col for col in ['Title', 'Description', 'similarity_score'] if col in st.session_state['search_results'].columns]
                    st.write(st.session_state['search_results'][columns_to_display])

        else:
            st.warning("The filtered dataset is empty. Please adjust your filters.")
    else:
        st.warning("Please select a dataset to proceed.")

# Clustering Tab
with tab2:
    st.header("Clustering")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            clustering_option = st.radio("Select data for clustering:", ('Full Dataset', 'Semantic Search Results'))

            if clustering_option == 'Semantic Search Results':
                if st.session_state.get('search_results') is not None and not st.session_state['search_results'].empty:
                    df_to_cluster = st.session_state['search_results']
                else:
                    st.warning("No search results found. Please perform a semantic search first.")
                    df_to_cluster = None
            else:
                df_to_cluster = st.session_state['filtered_df']

            if df_to_cluster is not None and not df_to_cluster.empty:
                # Use precomputed embeddings
                # df_to_cluster is a subset of df, so we can just index embeddings by df_to_cluster.index
                if 'embeddings' not in st.session_state:
                    # This should not happen if we followed the logic, but just in case:
                    st.error("Embeddings not found. Please go to 'Semantic Search' tab to trigger embedding computation.")
                else:
                    embeddings = st.session_state['embeddings']
                    selected_indices = df_to_cluster.index
                    embeddings_clustering = embeddings[selected_indices]

                df = df_to_cluster.copy()
                df['Title'] = df.get('Title', '').fillna('')
                df['Description'] = df.get('Description', '').fillna('')
                df['text'] = df['Title'] + ' ' + df['Description']

                if len(df['text']) == 0:
                    st.warning("No text data available for clustering.")
                else:
                    # Remove stop words
                    stop_words = set(stopwords.words('english'))
                    texts_cleaned = []
                    for text in df['text'].tolist():
                        word_tokens = word_tokenize(text)
                        filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
                        texts_cleaned.append(filtered_text)

                    if st.button("Run Clustering"):
                        st.write("Performing clustering...")
                        sentence_model = get_embedding_model()
                        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
                        topic_model = BERTopic(embedding_model=sentence_model, hdbscan_model=hdbscan_model)
                        try:
                            topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=embeddings_clustering)
                            df['Topic'] = topics
                            st.session_state['topic_model'] = topic_model
                            if clustering_option == 'Semantic Search Results':
                                st.session_state['clustered_data'] = df.copy()
                            else:
                                # Add Topics to the filtered_df as well (matches indexing)
                                st.session_state['filtered_df'].loc[df.index, 'Topic'] = df['Topic']

                            st.write("Clustering Results:")
                            columns_to_display = [col for col in ['Title', 'Description'] if col in df.columns]
                            columns_to_display.append('Topic')
                            st.write(df[columns_to_display])

                            st.write("Visualizing Topics...")

                            st.subheader("Intertopic Distance Map")
                            fig1 = topic_model.visualize_topics()
                            st.plotly_chart(fig1)

                            st.subheader("Topic Document Visualization")
                            fig2 = topic_model.visualize_documents(texts_cleaned, embeddings=embeddings_clustering)
                            st.plotly_chart(fig2)

                            st.subheader("Topic Hierarchy Visualization")
                            fig3 = topic_model.visualize_hierarchy()
                            st.plotly_chart(fig3)

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
                                    except Exception:
                                        st.warning("Could not generate Topics per Class visualization.")
                            else:
                                st.warning("No columns found in data to use for Topics per Class visualization.")

                        except Exception as e:
                            st.error(f"An error occurred during clustering: {e}")
                    else:
                        # If clustering is already done, show previous clustering results if available
                        if ('topic_model' in st.session_state) and (('clustered_data' in st.session_state and not st.session_state['clustered_data'].empty) or 'Topic' in st.session_state['filtered_df'].columns):
                            st.write("Clustering results are available from previous run.")
                            if clustering_option == 'Semantic Search Results' and 'clustered_data' in st.session_state:
                                df = st.session_state['clustered_data']
                            else:
                                df = st.session_state['filtered_df']

                            if 'Topic' in df.columns:
                                columns_to_display = [col for col in ['Title', 'Description', 'Topic'] if col in df.columns]
                                st.write(df[columns_to_display])
                            else:
                                st.write("No topics found. Please run clustering.")
                        # else do nothing
                # end of df_to_cluster checks
            else:
                st.warning("No data available for clustering.")
        else:
            st.warning("The filtered dataset is empty. Please adjust your filters.")
    else:
        st.warning("Please select a dataset to proceed.")

# Summarization Tab
with tab3:
    st.header("Summarization")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            # Choose the data to summarize
            # If we have clustered_data (from semantic search results), use it, otherwise filtered_df
            if 'clustered_data' in st.session_state and not st.session_state['clustered_data'].empty:
                df = st.session_state['clustered_data']
            else:
                df = st.session_state['filtered_df']

            if df is not None and not df.empty:
                if 'Topic' in df.columns:
                    impact_areas = ['Nutrition', 'Poverty Reduction', 'Gender', 'Climate Change', 'Environmental Biodiversity']
                    impact_area = st.selectbox("Select Impact Area", impact_areas)

                    impact_area_objectives = {
                        'Nutrition': """
                            Focusing on evidence and options for improving diets and human health ...
                            (omitted here for brevity, same text as original)
                        """,
                        'Poverty Reduction': """
                            Policy research and engagements in all segments of food systems ...
                            (omitted here for brevity)
                        """,
                        'Gender': """
                            Gender-transformative approaches, communication, and advocacy ...
                            (omitted here for brevity)
                        """,
                        'Climate Change': """
                            Scientific evidence, climate-smart solutions, and innovative finance ...
                            (omitted here for brevity)
                        """,
                        'Environmental Biodiversity': """
                            Use of modern digital tools to bring together state-of-the-art Earth system ...
                            (omitted here for brevity)
                        """
                    }
                    objectives = impact_area_objectives.get(impact_area, "")

                    if st.button("Generate Summaries"):
                        system_prompt = """
            You are an expert summarizer skilled in creating concise and relevant summaries ...
            """

                        openai_api_key = os.environ.get('OPENAI_API_KEY')
                        if not openai_api_key:
                            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                        else:
                            llm = ChatOpenAI(api_key=openai_api_key, model_name='gpt-4o')

                            all_texts = df['text'].tolist() if 'text' in df.columns else (df['Title'] + " " + df['Description']).tolist()
                            combined_text = " ".join(all_texts)
                            if combined_text.strip() == "":
                                st.warning("No text data available for summarization.")
                            else:
                                user_prompt = f"**Objectives**: {objectives}\n**Text to summarize**: {combined_text}"
                                system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                                human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                                chain = LLMChain(llm=llm, prompt=chat_prompt)
                                response = chain.run(user_prompt=user_prompt)
                                high_level_summary = response.strip()
                                st.write("### High-Level Summary:")
                                st.write(high_level_summary)

                                # Summaries per cluster
                                summaries = []
                                grouped_list = list(df.groupby('Topic'))
                                total_topics = len(grouped_list)
                                if total_topics == 0:
                                    st.warning("No topics found for summarization.")
                                else:
                                    progress_bar = st.progress(0)

                                    def generate_summary_per_topic(topic_group_tuple):
                                        topic, group = topic_group_tuple
                                        all_text = " ".join(group['text'].tolist())
                                        user_prompt = f"**Objectives**: {objectives}\n**Text to summarize**: {all_text}"
                                        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                                        human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                                        chain = LLMChain(llm=llm, prompt=chat_prompt)
                                        response = chain.run(user_prompt=user_prompt)
                                        summary = response.strip()
                                        return {'Topic': topic, 'Summary': summary}

                                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                                        futures = {executor.submit(generate_summary_per_topic, item): item[0] for item in grouped_list}
                                        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                                            result = future.result()
                                            summaries.append(result)
                                            progress_bar.progress((idx + 1) / total_topics)
                                    progress_bar.empty()
                                    summary_df = pd.DataFrame(summaries)
                                    st.write("### Summaries per Cluster:")
                                    st.write(summary_df)
                                    csv = summary_df.to_csv(index=False)
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="summaries.csv">Download Summaries CSV</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("Please perform clustering first to generate topics.")
            else:
                st.warning("No data available for summarization.")
        else:
            st.warning("The filtered dataset is empty. Please adjust your filters.")
    else:
        st.warning("Please select a dataset to proceed.")
