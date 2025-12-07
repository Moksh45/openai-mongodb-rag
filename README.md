# RAG with MongoDB Atlas Vector Search

A Retrieval-Augmented Generation (RAG) application using MongoDB Atlas Vector Search and OpenAI embeddings to answer questions based on document context.

## Overview

This project demonstrates how to build a RAG system that:
- Generates embeddings using OpenAI's `text-embedding-3-large` model
- Stores and indexes embeddings in MongoDB Atlas using Vector Search
- Retrieves relevant documents based on semantic similarity
- Uses GPT-4 to generate contextual answers

## Prerequisites

- Python 3.9+
- MongoDB Atlas account (free tier available)
- OpenAI API key

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd RAG-With-Mongodb
```

### 2. Create Virtual Environment & Install Dependencies

#### Option A: Using UV (Recommended - Faster) ⚡

[UV](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver, written in Rust.

```bash
# Install UV if you haven't already
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Initialize and set up the project
uv init
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install pymongo openai python-dotenv

# Install Jupyter kernel support
uv pip install ipykernel
```

#### Option B: Using Traditional pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pymongo openai python-dotenv ipykernel
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_URI=your_mongodb_atlas_connection_string_here
```

### 4. Set Up MongoDB Atlas

1. **Create a MongoDB Atlas Account**
   - Sign up at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
   - Create a free M0 cluster

2. **Configure Network Access**
   - Add your IP address to the IP Access List
   - Or allow access from anywhere (0.0.0.0/0) for development

3. **Create Database User**
   - Create a database user with read/write permissions
   - Save the username and password

4. **Get Connection String**
   - Click "Connect" on your cluster
   - Choose "Connect your application"
   - Copy the connection string and update `.env`

5. **Create Vector Search Index**
   - Navigate to your cluster in Atlas
   - Go to "Search" tab
   - Create a new Search Index with the following configuration:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 3072,
      "similarity": "cosine"
    }
  ]
}
```

## Project Structure

```
RAG-With-Mongodb/
├── .env                 # Environment variables (not in git)
├── .gitignore          # Git ignore file
├── rag.ipynb           # Main Jupyter notebook
├── README.md           # This file
└── .venv/              # Virtual environment
```

## Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open `rag.ipynb`**

3. **Run the cells sequentially** to:
   - Connect to MongoDB Atlas
   - Generate embeddings for your documents
   - Store embeddings in MongoDB
   - Create vector search index
   - Query the system with natural language questions
   - Get AI-generated answers based on retrieved context

### Key Components

#### 1. Embedding Generation

```python
from openai import OpenAI

client = OpenAI()
model = "text-embedding-3-large"

def get_embedding(text):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding
```

#### 2. Vector Search Query

```python
def get_query_results(query):
    query_embedding = get_embedding(query)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": 5
            }
        }
    ]
    
    results = collection.aggregate(pipeline)
    return list(results)
```

#### 3. RAG with GPT-4

```python
from openai import OpenAI

query = "What are MongoDB's latest AI announcements?"
context_docs = get_query_results(query)
context_string = " ".join([doc["text"] for doc in context_docs])

prompt = f"""Use the following pieces of context to answer the question at the end.
    {context_string}
    Question: {query}
"""

openai_client = OpenAI()
completion = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

print(completion.choices[0].message.content)
```

## Important Notes

### Atlas Search vs Local MongoDB

⚠️ **Atlas Vector Search requires MongoDB Atlas or AtlasCLI**

This project uses MongoDB Atlas Vector Search, which is **not available on standard local MongoDB installations**. You must use either:

1. **MongoDB Atlas (Cloud)** - Recommended for production and development
2. **AtlasCLI Local Deployment** - For local development with Atlas features

If you encounter the error:
```
OperationFailure: Using Atlas Search Database Commands requires additional configuration
```

This means you're trying to use Atlas Search features on a local MongoDB instance. Please follow the setup instructions above to use MongoDB Atlas.

## Features

- ✅ Semantic search using vector embeddings
- ✅ OpenAI GPT-4 integration for answer generation
- ✅ MongoDB Atlas Vector Search for efficient similarity search
- ✅ Jupyter notebook for interactive development
- ✅ Environment variable configuration for security

## Troubleshooting

### Connection Issues

If you can't connect to MongoDB Atlas:
- Check your IP is whitelisted in Atlas Network Access
- Verify your connection string in `.env`
- Ensure your database user credentials are correct

### Vector Search Not Working

- Verify your Search Index is created in Atlas
- Check the index name matches your code (`vector_index`)
- Ensure embedding dimensions match (3072 for `text-embedding-3-large`)

### OpenAI API Errors

- Verify your API key is valid
- Check you have sufficient credits
- Ensure you're using the correct model names

## Resources

- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/local-rag/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
