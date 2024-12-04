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

# Sidebar - Data Selection
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

if dataset_option == 'PRMS 2022+2023 QAed':
    # Load the default dataset
    default_dataset_path = os.path.join(os.path.dirname(__file__), 'input', 'export_data_table_results_20240312_160222CET.xlsx')
    df = load_default_dataset(default_dataset_path)
    if df is not None:
        st.session_state['df'] = df.copy()
        st.session_state['using_default_dataset'] = True
        st.write("Using default dataset:")
        st.write(df.head())
    else:
        st.warning("Please ensure the default dataset exists in the 'input' directory.")
else:
    # User uploads dataset
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        # Load the dataset
        df = load_uploaded_dataset(uploaded_file)
        st.session_state['df'] = df.copy()
        st.session_state['using_default_dataset'] = False
        st.write("Data preview:")
        st.write(df.head())
    else:
        st.warning("Please upload an Excel file to proceed.")

# Create tabs for each step
tab1, tab2, tab3 = st.tabs(["Semantic Search", "Clustering", "Summarization"])

# Semantic Search Tab
with tab1:
    st.header("Semantic Search")
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df'].copy()
        # Prepare data

        # Handle 'Title' and 'Description' columns optionally
        df['Title'] = df.get('Title', '').fillna('')
        df['Description'] = df.get('Description', '').fillna('')
        df['text_for_embedding'] = df['Title'] + " " + df['Description']

        @st.cache_resource
        def get_embedding_model():
            model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
            return model

        @st.cache_data
        def load_or_compute_embeddings(texts, embeddings_file):
            if os.path.exists(embeddings_file):
                st.write("Loading pre-calculated embeddings...")
                with open(embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                return embeddings
            else:
                st.write("Generating embeddings...")
                model = get_embedding_model()
                with st.spinner('Calculating embeddings...'):
                    embeddings = model.encode(texts, show_progress_bar=True, device=device)
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(embeddings, f)
                return embeddings

        # Load or generate embeddings
        if st.session_state.get('using_default_dataset'):
            embeddings_file = os.path.join(os.path.dirname(__file__), 'embeddings.pkl')
            embeddings = load_or_compute_embeddings(df['text_for_embedding'].tolist(), embeddings_file)
        else:
            embeddings_file = os.path.join(os.path.dirname(__file__), 'uploaded_embeddings.pkl')
            embeddings = load_or_compute_embeddings(df['text_for_embedding'].tolist(), embeddings_file)

        st.session_state['embeddings'] = embeddings

        # Input query
        query = st.text_input("Enter your search query:")
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.35)

        if st.button("Search"):
            if query:
                model = get_embedding_model()
                # Perform semantic search
                query_embedding = model.encode([query], device=device)
                similarities = cosine_similarity(query_embedding, embeddings)
                above_threshold_indices = np.where(similarities[0] > similarity_threshold)[0]
                results = df.iloc[above_threshold_indices].copy()
                results['similarity_score'] = similarities[0][above_threshold_indices]
                # Sort results by similarity_score in descending order
                results = results.sort_values(by='similarity_score', ascending=False)
                st.session_state['search_results'] = results.copy()
                st.write("Search Results:")
                # Dynamically select columns to display
                columns_to_display = [col for col in ['Title', 'Description'] if col in results.columns]
                columns_to_display.append('similarity_score')
                st.write(results[columns_to_display])
            else:
                st.warning("Please enter a query to search.")
    else:
        st.warning("Please select a dataset to proceed.")

# Clustering Tab
with tab2:
    st.header("Clustering")
    if 'df' in st.session_state and st.session_state['df'] is not None:
        # Option to cluster on full dataset or search results
        clustering_option = st.radio("Select data for clustering:", ('Full Dataset', 'Semantic Search Results'))

        if clustering_option == 'Semantic Search Results':
            if st.session_state.get('search_results') is not None:
                df_to_cluster = st.session_state['search_results']
            else:
                st.warning("No search results found. Please perform a semantic search first.")
                df_to_cluster = None
        else:
            df_to_cluster = st.session_state['df']

        if df_to_cluster is not None:
            df = df_to_cluster.copy()
            # Prepare text for clustering
            df['Title'] = df.get('Title', '').fillna('')
            df['Description'] = df.get('Description', '').fillna('')
            df['text'] = df['Title'] + ' ' + df['Description']
            texts = df['text'].tolist()

            # Remove stop words
            stop_words = set(stopwords.words('english'))
            texts_cleaned = []
            for text in texts:
                word_tokens = word_tokenize(text)
                filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
                texts_cleaned.append(filtered_text)

            @st.cache_resource
            def get_sentence_model():
                model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
                return model

            if clustering_option == 'Semantic Search Results':
                st.write("Generating embeddings for clustering...")
                sentence_model = get_sentence_model()
                with st.spinner('Calculating embeddings...'):
                    embeddings_clustering = sentence_model.encode(texts_cleaned, show_progress_bar=True, device=device)
            else:
                @st.cache_data
                def load_or_compute_clustering_embeddings(texts_cleaned, embeddings_file_clustering):
                    if os.path.exists(embeddings_file_clustering):
                        st.write("Loading pre-calculated embeddings for clustering...")
                        with open(embeddings_file_clustering, 'rb') as f:
                            embeddings_clustering = pickle.load(f)
                        return embeddings_clustering
                    else:
                        st.write("Generating embeddings for clustering...")
                        sentence_model = get_sentence_model()
                        with st.spinner('Calculating embeddings...'):
                            embeddings_clustering = sentence_model.encode(texts_cleaned, show_progress_bar=True, device=device)
                        with open(embeddings_file_clustering, 'wb') as f:
                            pickle.dump(embeddings_clustering, f)
                        return embeddings_clustering

                # Load or generate embeddings for clustering
                if st.session_state.get('using_default_dataset'):
                    embeddings_file_clustering = os.path.join(os.path.dirname(__file__), 'embeddings_clustering.pkl')
                else:
                    embeddings_file_clustering = os.path.join(os.path.dirname(__file__), 'uploaded_embeddings_clustering.pkl')
                embeddings_clustering = load_or_compute_clustering_embeddings(texts_cleaned, embeddings_file_clustering)

            st.session_state['embeddings_clustering'] = embeddings_clustering

            # Perform clustering
            if st.button("Run Clustering"):
                st.write("Performing clustering...")
                sentence_model = get_sentence_model()
                hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
                topic_model = BERTopic(embedding_model=sentence_model, hdbscan_model=hdbscan_model)
                topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=embeddings_clustering)
                df['Topic'] = topics
                st.session_state['topic_model'] = topic_model
                if clustering_option == 'Semantic Search Results':
                    st.session_state['clustered_data'] = df.copy()
                else:
                    st.session_state['df']['Topic'] = df['Topic']
                st.write("Clustering Results:")
                # Dynamically select columns to display
                columns_to_display = [col for col in ['Title', 'Description'] if col in df.columns]
                columns_to_display.append('Topic')
                st.write(df[columns_to_display])

                # Optionally display visualizations
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
        st.warning("Please select a dataset to proceed.")

# Summarization Tab
with tab3:
    st.header("Summarization")
    if 'df' in st.session_state and st.session_state['df'] is not None:
        if st.session_state.get('clustered_data') is not None:
            df = st.session_state['clustered_data']
        else:
            df = st.session_state['df']

        if 'Topic' in df.columns:
            # Select Impact Area and Objectives
            impact_areas = ['Nutrition', 'Poverty Reduction', 'Gender', 'Climate Change', 'Environmental Biodiversity']
            impact_area = st.selectbox("Select Impact Area", impact_areas)

            # Define objectives for each impact area
            impact_area_objectives = {
                'Nutrition': """
                    Focusing on evidence and options for improving diets and human health through food systems outcomes, policy research, and technical and institutional innovations for making healthy sustainable diets affordable, targeting consumer behavior, local urban and informal markets, and social protection. 
                    Accelerated innovation in agronomy, livestock, and fisheries management to increase and diversify food supply and to manage zoonotic diseases, food safety, and anti-microbial resistance. 
                    Research advanced on a wider range of foods and farming systems, including vegetables, insects, and urban farming, with a focus on affordable diets and perishable foods. 
                    Dietary diversity, quality, and resilience underpinned by custodianship and distribution of a wide variety of genetic materials of crops and their wild relatives, and livestock; breeding of nutrient-dense legumes, roots, tubers, bananas and cereals, including biofortification and market relevant traits and; breeding of more productive livestock and fish to increase the supply of nutrient-dense animal source foods.
                """,
                'Poverty Reduction': """
                    Policy research and engagements in all segments of food systems to help improve access to productive resources, knowledge, finance, and markets to foster inclusion and increase effectiveness of agricultural policies, social protection programs, and off-farm employment opportunities. 
                    Solutions for strengthening resilience, risk management, and competitiveness to improve income-generating opportunities and sustainability of small-scale agriculture and agrifood chain participation, with a focus on women and youth. 
                    Adoption of adapted and resilient varieties and breeds, with turnover of crop varieties, leading to higher or more resilient crop yields and livestock and fish productivity, in turn driving higher and more stable farmer incomes, access to new markets, and poverty reduction among farmers and value chain participants.
                """,
                'Gender': """
                    Gender-transformative approaches, communication, and advocacy that lead to empowerment of women and youth, encourage entrepreneurship, and address the socio-political barriers to social inclusion in food, land, and water systems. 
                    Interventions designed to enable equal access to innovations and capacity development, as well as financial, informational, and legal services for women and young people to enable them to shape agrifood systems. 
                    Supply of improved varieties and breeds that are affordable and accessible to women, youth, and disadvantaged social groups, meeting their specific market requirements and preferences.
                """,
                'Climate Change': """
                    Scientific evidence, climate-smart solutions, and innovative finance that feed into local, national, regional, and global processes governing land use, land restoration, forest conservation, and resilience to floods and droughts, contributing to climate action, peace, and security. 
                    Co-development of production systems and portfolios of practices, adapted to the local needs of small-scale producers to enhance their adaptive capacity while reducing emissions; provision of affordable and accessible climate-informed services, particularly using digital tools. 
                    Adaptation to a changing climate through adapted breeds and varieties, for example heat tolerant livestock breeds, strains and crosses, drought-tolerant maize, and heat-tolerant beans; inclusion of long-term accessions in genebanks to provide solutions for future climates.
                """,
                'Environmental Biodiversity': """
                    Use of modern digital tools to bring together state-of-the-art Earth system observation and big data analysis to inform codesign of global solutions and national policies for staying within planetary boundaries on water use, nutrient use, land use change, and biodiversity. 
                    Cost-effective improved management of water, soil, nutrients, and biodiversity in crop, livestock, and fisheries systems, coupled with higher-order landscape considerations as well as circular economy and agroecological approaches. 
                    Biodiversity function of genebanks; breeding to reduce environmental footprint, e.g. less water or pesticides, to help stay within planetary boundaries and to reduce local water stress, pollution, biodiversity loss, and undesirable land use change.
                """
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
                    llm = ChatOpenAI(api_key=openai_api_key, model_name='gpt-4o')

                    # First, generate the high-level general summary
                    all_texts = df['text'].tolist()
                    combined_text = " ".join(all_texts)
                    # Prepare prompts
                    user_prompt = f"**Objectives**: {objectives}\n**Text to summarize**: {combined_text}"
                    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                    human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                    chain = LLMChain(llm=llm, prompt=chat_prompt)
                    response = chain.run(user_prompt=user_prompt)
                    high_level_summary = response.strip()
                    # Display the high-level summary
                    st.write("### High-Level Summary:")
                    st.write(high_level_summary)

                    # Now generate summaries per cluster/topic in parallel
                    summaries = []
                    grouped_list = list(df.groupby('Topic'))
                    total_topics = len(grouped_list)
                    progress_bar = st.progress(0)

                    def generate_summary_per_topic(topic_group_tuple):
                        topic, group = topic_group_tuple
                        all_text = " ".join(group['text'].tolist())
                        # Prepare prompts
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
                    # Optionally save summaries
                    csv = summary_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="summaries.csv">Download Summaries CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Please perform clustering first to generate topics.")
    else:
        st.warning("Please select a dataset to proceed.")