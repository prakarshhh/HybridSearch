import streamlit as st
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Initialize the Pinecone client
api_key = "5467e6f2-b135-4a07-b237-b92fb467feb3"
index_name = "hybrid-search-langchain-pinecone"
pc = Pinecone(api_key=api_key)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Set up embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Define sample sentences
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited New Orleans",
]

# Fit the BM25 encoder
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")

# Load the BM25 encoder from the saved file
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Set up the retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Add texts to the index
retriever.add_texts(sentences)

# Streamlit app layout
st.set_page_config(page_title="Enhanced Search App", layout="wide")

st.title("Enhanced Hybrid Search App")
st.write("Explore our advanced search tool. Type a query below to find relevant results.")

# Add custom CSS for button animation
st.markdown("""
    <style>
    .animated-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .animated-button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Add custom HTML and JavaScript for balloon animation
st.markdown("""
    <style>
    .balloon {
        position: absolute;
        width: 30px;
        height: 30px;
        background-color: #f00;
        border-radius: 50%;
        animation: float 5s infinite;
    }
    @keyframes float {
        0% {
            transform: translateY(0);
        }
        100% {
            transform: translateY(-1000px);
        }
    }
    </style>
    <script>
    function createBalloon() {
        var balloon = document.createElement('div');
        balloon.className = 'balloon';
        balloon.style.left = Math.random() * window.innerWidth + 'px';
        document.body.appendChild(balloon);
        setTimeout(() => balloon.remove(), 5000); // Remove balloon after 5 seconds
    }
    setInterval(createBalloon, 500); // Create a balloon every 500ms
    </script>
    """, unsafe_allow_html=True)

# Sidebar for input
st.sidebar.header("Search Configuration")
query = st.sidebar.text_input("Type your search query:")
search_button = st.sidebar.button("Perform Search", key="search_button", css_class="animated-button")

# Display search results
if search_button:
    if query:
        results = retriever.invoke(query)
        st.subheader("Search Results:")
        if results:
            for idx, result in enumerate(results):
                st.write(f"**Result {idx + 1}:** {result.page_content}")
        else:
            st.write("No matches found for your query.")
    else:
        st.write("Please provide a query to search.")
