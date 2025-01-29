# Abstract

In recent years, single-cell RNA sequencing (scRNA-seq) technologies have revolutionised our understanding of cellular heterogeneity. Various methods have been developed to interpret this complexity, with Seurat and scVI emerging as prominent tools for scRNA-seq data clustering. This analysis gives insight into the differences in functionality and results of both methods. This is done by exploring the workflow of primarily scVI with relevant mentions of the Seurat workflow. The result of both workflows is a UMAP, which comes from the clustering of the data. 

# Project Dependencies

This project requires several Python packages which are listed in the `requirements.txt` file. 
You can install these dependencies using the following command:

```
pip install -r requirements.txt
```

## PyTorch Installation

Note that the `requirements.txt` does not include a version of PyTorch, you must install a different version depending on your system and whether you are utilising CPUs or GPUs.

Please visit the official PyTorch installation guide to find the command that is appropriate for your environment:

[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

Follow the instructions on the website to install the correct version of PyTorch for your system.

## Note for NVIDIA GPU Users

If you plan to use an NVIDIA GPU, ensure that you have the correct version of CUDA installed that is compatible with the PyTorch version you are installing. The PyTorch installation guide linked above provides information about which CUDA version is required for each PyTorch version.


# Workflow Prerequisites

To conduct the analysis presented in this project, you will need a Seurat `.h5ad` file. This file format is specifically designed for single-cell genomics data and is used to store the output of the Seurat workflow.

You can download the necessary `.h5ad` file and the original Seurat workflow from the following Google Drive location:

[Seurat `.h5ad` file and Workflow](https://drive.google.com/drive/folders/1_qHpi0s9k8x54v2LVw6mtNl2ylpWUx2j?usp=sharing)

Please ensure you have the correct permissions to access this data and download the file to your local environment before proceeding with the analysis.

# How to Run

## Prerequisites

Before proceeding, ensure you have Jupyter Notebook installed on your system. Jupyter Notebook is essential for running and interacting with the analysis scripts. If you do not have Jupyter Notebook installed, please visit [the official Jupyter installation guide](https://jupyter.org/install) for instructions.

## Setting Up the Data

After downloading the `.h5ad` file, place it in the designated `/h5ad/seurat/` directory of this project.

## Running the Analysis

Follow the instructions provided in the subsequent sections to run the analysis with the Seurat and scVI tools, and to compare their respective UMAP results.

## Customising Hyperparameters

To train the model with bespoke hyperparameters, alter the `train_params` variable in the script. Change `train_params = False` to `train_params = True` and modify the hyperparameter grid to suit your requirements.

For example, to adjust the settings, you would amend the grid parameters as follows:

```python
train_params = True
param_grid = {'n_hidden': [your_params], 'n_latent': range(your_params), 'n_layers': range(your_params)}
```
Please be aware that the hyperparameters: 
```python
param_grid = {'n_hidden': [128], 'n_latent': range(2, 51, 1), 'n_layers': range(1, 6, 1)}
```
have already been executed, and the outcomes are stored in the hp_models directory.

## Creating Additional UMAP Visualisations

If you wish to create other UMAP visualisations based on different model files, you need to modify the model variable in the script. For instance, to analyse different sets of hyperparameters, change the models list to include the corresponding model names, like so:
```python
model_dir = os.path.join(save_dir, "hp_model/")
models = ["1_128_2", "1_128_10", "1_128_20", "Your model names here"]
```
This will allow the script to generate UMAP visualisations for each specified model configuration.

This addition guides users on how to generate UMAP visualizations for different model configurations, making the markdown document more comprehensive for diverse use cases.


# Original Seurat Workflow

For those interested in the original Seurat workflow used to generate the `.h5ad` file, you can also find this in the Google Drive folder linked above. It may provide additional context and understanding of how the initial data was processed and analysed.

# Contact

If you encounter any issues accessing the data or have questions regarding the workflow, please feel free to reach out using the contact information provided below.

mghuibregtse@gmail.com

jaimymohammadi@gmail.com


# Biomedical Literature Query Processor

![Project Logo](path/to/logo.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Input and Output](#input-and-output)
- [Running the Application](#running-the-application)
- [Customization](#customization)
- [Code Structure](#code-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The **Biomedical Literature Query Processor** is a comprehensive tool designed to facilitate advanced querying and information retrieval from biomedical literature. Leveraging natural language processing (NLP) techniques, embedding models, and efficient search algorithms, this tool provides accurate and relevant responses to user queries. Whether you're a researcher, data scientist, or bioinformatics specialist, this tool streamlines the process of extracting meaningful insights from vast datasets and documents.

## Features

- **Query Expansion:** Enhances user queries by generating related terms and synonyms using OpenAI's GPT-4 model.
- **Document Embedding:** Utilizes transformer-based models to create embeddings for efficient similarity searches.
- **Search Algorithms:** Implements both FAISS for vector-based similarity search and BM25 for keyword-based ranking.
- **PDF and Data Processing:** Extracts and processes information from PDFs and structured data files.
- **Caching Mechanism:** Efficiently handles gene symbol resolution with caching to minimize redundant API calls.
- **Performance Monitoring:** Tracks and logs execution times of various functions for optimization insights.
- **Customizable Configurations:** Easily adjust settings and parameters through configuration files.

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager

### Clone the Repository

```bash
git clone https://github.com/yourusername/biomedical-query-processor.git
cd biomedical-query-processor
Create a Virtual Environment (Recommended)
bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies
bash
Copy
pip install -r requirements.txt
Install Additional Dependencies
Some libraries like torch may require specific installation steps based on your system and hardware. Refer to the PyTorch Installation Guide for detailed instructions.

Configuration
All configurations are managed through JSON files located in the ./configs_system_instruction/ directory.

Configuration Files
Default Configuration: config_GPT_generated.json
Role-Based Configuration: config_role_based.json
Key Configuration Parameters
query: The user's search query.
number_of_expansions: Number of query expansions to generate.
batch_size: Number of documents to process in each batch.
model: Name of the transformer model to use for embeddings.
amount_docs: Number of top documents to retrieve.
weight_faiss: Weight assigned to FAISS scores during ranking.
weight_bm25: Weight assigned to BM25 scores during ranking.
Role-Based Parameters (only if using config_role_based.json):
persona
instruction
context
context_with_example_pathways
user_input
examples
output_indicator
Environment Variables
Create a .env file in the root directory to manage sensitive information like API keys.

bash
Copy
OPENAI_API_KEY=your_openai_api_key_here
Input and Output
Input Directories
Configuration Files: ./configs_system_instruction/
Data Files: ./Data/biomart/ (contains .gmt.gz and .txt.gz files)
PDF Documents: ./Data/PDF/
Gene ID to Symbol Cache: ./Data/JSON/ncbi_id_to_symbol.json
Output Directories
Database: chunks_embeddings.db (SQLite database storing document chunks)
FAISS Index: faiss_index.bin
Logs:
./file_log/file_log.json
time.txt
Responses:
answer.txt
documents.txt
scores.xlsx
unknown_genes.txt
Data Flow
Input Data: Data files and PDFs are ingested from the specified input directories.
Processing:
Gene IDs are resolved and converted to symbols.
Documents are chunked and embedded.
Embeddings are stored in the FAISS index and SQLite database.
Querying:
User queries are expanded.
Relevant documents are retrieved using FAISS and BM25.
Responses are generated using the GPT-4 model.
Output: Responses and relevant scores are saved to output files.
Running the Application
Prepare Your Data
Ensure that your data files are placed in the appropriate directories:

.gmt.gz and .txt.gz files in ./Data/biomart/
PDF documents in ./Data/PDF/
Execute the Script
bash
Copy
python your_script_name.py
Replace your_script_name.py with the actual name of your Python script containing the provided code.

Monitoring Execution
Execution times for various functions are logged in time.txt. Check this file to monitor performance and identify potential bottlenecks.

Customization
Configuration Parameters
Adjust the following parameters in your configuration JSON files to customize the application's behavior:

Query Settings:
query: Modify the default search query.
number_of_expansions: Change the number of query expansions generated.
Model Settings:
model: Switch between different transformer models (e.g., bert-base-uncased, roberta-large).
Search Settings:
amount_docs: Determine how many top documents to retrieve.
weight_faiss & weight_bm25: Adjust the weighting between FAISS and BM25 scores for ranking.
Batch Processing:
batch_size: Increase or decrease the number of documents processed per batch based on your system's capabilities.
Environment Variables
Update the .env file with your OpenAI API key and any other necessary environment variables.

Extending Functionality
Adding New Data Sources: Modify the load_gz_files and load_pdf_files functions to include additional data formats or sources.
Custom Embedding Models: Change the transformer model by updating the model parameter in the configuration and ensuring compatibility.
Adjusting Tokenization: Customize the tokenize function to include or exclude specific tokens or to change tokenization rules.
Code Structure
Main Components
Data Processing:
process_excel_data()
convert_gene_id_to_symbols()
chunk_documents()
chunk_pdfs()
Embedding and Indexing:
load_model_and_tokenizer()
embed_documents()
initialize_faiss_index()
build_bm25_index()
Search and Retrieval:
query_faiss_index()
query_bm25_index()
weighted_rrf()
rank_and_retrieve_documents()
Query Expansion and Response:
query_expansion()
generate_gpt4_turbo_response_with_instructions()
generate_response_and_save()
Utilities:
Timer decorators for performance monitoring.
File handling and caching mechanisms.
Database
Utilizes SQLite (chunks_embeddings.db) to store and retrieve document chunks efficiently.

Indexing
FAISS: Handles vector-based similarity search for embeddings.
BM25: Manages keyword-based ranking of documents.
Troubleshooting
Common Issues
Missing Dependencies:
Ensure all required packages are installed via pip install -r requirements.txt.
Some packages like torch may need specific installation commands.
API Key Errors:
Verify that the OpenAI API key is correctly set in the .env file.
Ensure that the key has the necessary permissions and is not expired.
Data File Issues:
Confirm that data files are placed in the correct directories.
Check for file corruption, especially for .gz and .pdf files.
Memory Errors:
Adjust batch_size in the configuration to a lower number if encountering memory issues.
Ensure your system has sufficient RAM and, if available, GPU resources.
FAISS Index Loading Errors:
Verify that faiss_index.bin exists and is not corrupted.
Ensure that the embedding dimensions match between the model and the FAISS index.
Logs and Monitoring
Execution Times: Review time.txt for function execution times.
Error Messages: Check console outputs for any error messages during execution.
Output Files: Inspect unknown_genes.txt and other output files for unexpected content.
Contributing
Contributions are welcome! Please follow these steps:

Fork the Repository
Create a Feature Branch
bash
Copy
git checkout -b feature/YourFeature
Commit Your Changes
bash
Copy
git commit -m "Add some feature"
Push to the Branch
bash
Copy
git push origin feature/YourFeature
Open a Pull Request
Please ensure that your code adheres to the project's coding standards and includes appropriate documentation and tests.

License
This project is licensed under the MIT License.

Contact
For any questions or support, please contact:

Name: Your Name
Email: your.email@example.com
LinkedIn: Your LinkedIn Profile
GitHub: yourusername
