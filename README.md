<p align="center">
  <img src="https://img.shields.io/badge/ProRAG-RAG%20Pipeline-blueviolet?style=for-the-badge&logo=python&logoColor=white" alt="ProRAG Banner"/>
</p>

<h1 align="center">ProRAG</h1>

<p align="center">
  <em>A Fully Configurable Retrieval-Augmented Generation Pipeline for Document Q&A Applications</em>
</p>

<p align="center">
  <a href="https://github.com/ProRAG/ProRAG/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License: MIT"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace"/></a>
  <a href="https://python.langchain.com/"><img src="https://img.shields.io/badge/LangChain-0.2+-1C3C3C.svg?style=flat-square" alt="LangChain"/></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-0.5+-F7931A.svg?style=flat-square" alt="ChromaDB"/></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Programmatic API](#programmatic-api)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Supported Models](#supported-models)
- [Roadmap & Future Enhancements](#roadmap--future-enhancements)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)

---

## Overview

**ProRAG** is an open-source, fully configurable **Retrieval-Augmented Generation (RAG)** pipeline for document Q&A applications. It bridges the gap between large language models and domain-specific knowledge by combining semantic document retrieval with instruction-tuned text generation.

Built on top of **LangChain**, **ChromaDB**, and **Hugging Face Transformers**, ProRAG allows researchers and developers to perform context-aware question answering over text corpora with minimal setup. Whether you're building a chatbot, an academic research tool, or a document Q&A system — PoRAG provides the modular, extensible foundation to get started.

---

## Key Features

| Feature | Description |
|---|---|
| **Language-Aware Design** | End-to-end pipeline optimized for text — from chunking with configurable sentence delimiters (`!`, `?`, and custom) to instruction-tuned LLM generation. |
| **Plug-and-Play Models** | Seamlessly swap chat and embedding models via Hugging Face Hub IDs. Use any compatible model without code changes. |
| **4-Bit Quantization** | Built-in support for **BitsAndBytes NF4 quantization**, enabling inference of 8B+ parameter models on consumer GPUs with as little as ~6 GB VRAM. |
| **ChromaDB Vector Store** | Persistent vector storage with similarity-based retrieval for fast, scalable document search. |
| **Configurable Chunking** | Fine-grained control over `chunk_size` and `chunk_overlap` parameters for optimal retrieval granularity. |
| **LangChain LCEL Chains** | Modern LangChain Expression Language (LCEL) pipeline using `RunnableParallel` and `RunnablePassthrough` for composable, debuggable chains. |
| **Interactive CLI** | Rich terminal UI with colored panels, progress bars, and an interactive Q&A loop. |
| **Context Transparency** | Optional `--show_context` flag to inspect retrieved source passages alongside generated answers. |
| **GPU Auto-Detection** | Automatic CUDA device detection with graceful CPU fallback. |
| **Hugging Face Auth** | Native `--hf_token` support for gated or private model access. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ProRAG Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌───────────────────┐    ┌────────────────┐  │
│   │  Text File     │───▶│ Text Splitter      │───▶│ Text Chunks    │  │
│   │  (.txt file)  │    │ (Recursive, ।!?)  │    │                │  │
│   └──────────────┘    └───────────────────┘    └───────┬────────┘  │
│                                                         │           │
│                        ┌────────────────────────────────▼────────┐  │
│                        │  Embedding Model (Sentence Transformer)│  │
│                        │  l3cube-pune/bengali-sentence-           │  │
│                        │  similarity-sbert                        │  │
│                        └────────────────────────────────┬────────┘  │
│                                                         │           │
│                        ┌────────────────────────────────▼────────┐  │
│                        │  ChromaDB Vector Store                  │  │
│                        │  (Similarity Search, Top-K Retrieval)   │  │
│                        └────────────────────────────────┬────────┘  │
│                                                         │           │
│   ┌──────────────┐                                      │           │
│   │  User Query   │──────────────────────┐              │           │
│   │  (User Query) │                      │              │           │
│   └──────────────┘                      ▼              ▼           │
│                        ┌─────────────────────────────────────────┐  │
│                        │  LangChain RAG Chain (LCEL)             │  │
│                        │  ┌─────────────┐  ┌──────────────────┐ │  │
│                        │  │  Retriever   │  │ Prompt Template  │ │  │
│                        │  │  (Top-K)     │──│ (Instruction)    │ │  │
│                        │  └─────────────┘  └────────┬─────────┘ │  │
│                        │                             │           │  │
│                        │              ┌──────────────▼─────────┐ │  │
│                        │              │  LLM Generation        │ │  │
│                        │              │  (LLM Generation)      │ │  │
│                        │              └──────────────┬─────────┘ │  │
│                        └─────────────────────────────┼───────────┘  │
│                                                      │              │
│                        ┌─────────────────────────────▼───────────┐  │
│                        │  Response (Answer + Context)             │  │
│                        └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| **Orchestration** | [LangChain](https://python.langchain.com/) `>=0.2.3` | RAG chain composition & LCEL pipelines |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) `>=0.5.0` | Document embedding storage & similarity retrieval |
| **LLM Framework** | [Hugging Face Transformers](https://huggingface.co/docs/transformers) `>=4.40.1` | Model loading, tokenization & text generation |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) `>=3.0.1` | Sentence embedding generation |
| **Quantization** | [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) `0.41.3` | 4-bit NF4 quantization for memory-efficient inference |
| **Fine-Tuning** | [PEFT](https://huggingface.co/docs/peft) `>=0.11.1` | Parameter-efficient fine-tuning (LoRA/QLoRA) support |
| **Acceleration** | [Accelerate](https://huggingface.co/docs/accelerate) `0.31.0` | Multi-GPU & mixed-precision training utilities |
| **Deep Learning** | [PyTorch](https://pytorch.org/) | Tensor computation & CUDA acceleration |
| **Terminal UI** | [Rich](https://rich.readthedocs.io/) `>=13.7.1` | Beautiful terminal output with panels & progress bars |

---

## Getting Started

### Prerequisites

- **Python** 3.10 or higher
- **CUDA-compatible GPU** (recommended; CPU fallback available but significantly slower)
- **Git** for cloning the repository
- ~16 GB GPU VRAM for full-precision inference (~6 GB with 4-bit quantization)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/healer-125/pro-rag.git
cd pro-rag
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install PyTorch with CUDA** (if not already installed)

```bash
# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### Command-Line Interface

Run ProRAG with a text file:

```bash
python main.py --text_path ./test.txt
```

**With all options:**

```bash
python main.py \
  --text_path ./test.txt \
  --chat_model hassanaliemon/bn_rag_llama3-8b \
  --embed_model l3cube-pune/bengali-sentence-similarity-sbert \
  --k 4 \
  --top_k 2 \
  --top_p 0.6 \
  --temperature 0.6 \
  --chunk_size 500 \
  --chunk_overlap 150 \
  --max_new_tokens 256 \
  --quantization \
  --show_context \
  --hf_token YOUR_HF_TOKEN
```

**Interactive session example:**

```
Your question: When was the author of the document born?
Answer: The author was born on May 7, 1861.

Your question: exit
Goodbye, thank you!
```

### Programmatic API

Use ProRAG as a Python library in your own applications:

```python
from prorag import RAGChain

# Initialize the pipeline
rag = RAGChain()

# Load models and data
rag.load(
    chat_model_id="hassanaliemon/bn_rag_llama3-8b",
    embed_model_id="l3cube-pune/bengali-sentence-similarity-sbert",
    text_path="./test.txt",
    quantization=True,       # Enable 4-bit quantization
    k=4,                     # Retrieve top 4 chunks
    top_k=2,
    top_p=0.6,
    temperature=0.6,
    chunk_size=500,
    chunk_overlap=150,
    max_new_tokens=256,
    hf_token=None,           # Optional: for gated models
)

# Ask questions
answer, context = rag.get_response("Tell me about the main subject of the document.")
print(f"Answer: {answer}")
print(f"Context: {context}")
```

---

## Configuration

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| Chat Model | `--chat_model` | `hassanaliemon/bn_rag_llama3-8b` | Hugging Face model ID for the instruction-tuned LLM |
| Embedding Model | `--embed_model` | `l3cube-pune/bengali-sentence-similarity-sbert` | Hugging Face model ID for sentence embeddings |
| Text Path | `--text_path` | *required* | Path to the `.txt` file to index |
| Top-K Retrieval | `--k` | `4` | Number of document chunks to retrieve |
| Top-K Sampling | `--top_k` | `2` | Top-k sampling parameter for generation |
| Top-P (Nucleus) | `--top_p` | `0.6` | Nucleus sampling probability threshold |
| Temperature | `--temperature` | `0.6` | Controls randomness in generation (lower = more deterministic) |
| Max New Tokens | `--max_new_tokens` | `256` | Maximum number of tokens to generate |
| Chunk Size | `--chunk_size` | `500` | Character-level chunk size for text splitting |
| Chunk Overlap | `--chunk_overlap` | `150` | Overlap between consecutive chunks |
| Show Context | `--show_context` | `False` | Display retrieved context alongside answers |
| Quantization | `--quantization` | `False` | Enable 4-bit NF4 quantization |
| HF Token | `--hf_token` | `None` | Hugging Face API token for private/gated models |

---

## Project Structure

```
ProRAG/
├── main.py                        # CLI entry point & interactive Q&A loop
├── prorag/                         # Core package
│   ├── __init__.py                # Package exports (RAGChain)
│   └── rag_pipeline.py            # RAG pipeline implementation
├── test.txt                       # Sample text file for testing
├── requirements.txt               # Python dependencies
├── CITATION.cff                   # Academic citation metadata
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## How It Works

ProRAG follows a standard RAG workflow:

1. **Document Ingestion** — Text is read from a `.txt` file and split into overlapping chunks using `RecursiveCharacterTextSplitter` with configurable delimiters (e.g. `!`, `?`).

2. **Embedding & Indexing** — Each chunk is embedded using a sentence transformer model and stored in a ChromaDB vector database.

3. **Query & Retrieval** — When a user submits a query, the retriever performs similarity search against the vector store and returns the top-K most relevant chunks.

4. **Augmented Generation** — Retrieved chunks are formatted as context and injected into an instruction prompt template. The instruction-tuned LLM generates a grounded response.

5. **Response Extraction** — The raw model output is parsed to extract the clean response from the `### Response:` section of the template.

---

## Supported Models

### Default Models

| Role | Model | Source |
|---|---|---|
| **Chat / Generation** | `hassanaliemon/bn_rag_llama3-8b` | Instruction-tuned Llama 3 8B |
| **Embeddings** | `l3cube-pune/bengali-sentence-similarity-sbert` | Sentence-BERT for embeddings |

### Compatible Model Families

You can replace the default models with any compatible Hugging Face model:

- **Chat Models**: Llama 3.x, Mistral, Gemma 2, Qwen 2.5, Phi-3/4, Command R+, or any compatible causal LM
- **Embedding Models**: Any `sentence-transformers` compatible model for your language or domain

---

## Roadmap & Future Enhancements

ProRAG is actively evolving. Below are planned and aspirational features aligned with the latest advancements in the Python and AI ecosystem:

### Near-Term

- [ ] **Multi-Document Support** — Ingest multiple files, PDFs, and web-scraped content
- [ ] **Persistent Vector Store** — Persist ChromaDB collections to disk for reuse across sessions
- [ ] **Streaming Generation** — Token-by-token streaming responses for real-time UX
- [ ] **LangSmith Integration** — Observability, tracing, and evaluation of RAG chains via [LangSmith](https://smith.langchain.com/)

### Model & Inference Enhancements

- [ ] **vLLM / TGI Backend** — High-throughput inference with [vLLM](https://github.com/vllm-project/vllm) or [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [ ] **GGUF / llama.cpp Support** — CPU-optimized inference with quantized GGUF models via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [ ] **GPTQ & AWQ Quantization** — Post-training quantization methods beyond NF4 for deployment flexibility
- [ ] **Speculative Decoding** — Accelerated generation using draft models for faster inference
- [ ] **Multi-Modal RAG** — Support for image+text documents using vision-language models (e.g., LLaVA, Qwen-VL)

### Retrieval Enhancements

- [ ] **Hybrid Search** — Combine dense vector similarity with BM25 sparse retrieval for improved recall
- [ ] **Re-Ranking** — Cross-encoder re-ranking of retrieved passages using models like `ms-marco` or domain-specific re-rankers
- [ ] **Parent Document Retriever** — Retrieve small chunks but return full parent documents for richer context
- [ ] **Multi-Vector Retriever** — Generate multiple embeddings per document (summary + content) for semantic diversity
- [ ] **Knowledge Graph Integration** — Structured knowledge extraction and graph-based retrieval (GraphRAG)
- [ ] **Contextual Compression** — LLM-based compression of retrieved passages to reduce noise

### Advanced RAG Patterns

- [ ] **Agentic RAG** — Tool-using agents with LangGraph that can dynamically decide when and how to retrieve
- [ ] **Corrective RAG (CRAG)** — Self-reflective retrieval with hallucination detection and query rewriting
- [ ] **Self-RAG** — Adaptive retrieval where the model decides whether retrieval is needed
- [ ] **RAG Fusion** — Multiple query reformulations with reciprocal rank fusion for robust retrieval
- [ ] **RAPTOR** — Recursive abstractive processing for tree-organized retrieval across document hierarchies

### Python & Developer Experience

- [ ] **Python 3.12+ Features** — Leverage `typing` improvements (PEP 695 type aliases), `asyncio` task groups, and improved error messages
- [ ] **Async Pipeline** — Fully async chain execution using `asyncio` and LangChain's async APIs
- [ ] **Pydantic v2 Schemas** — Structured input/output validation with Pydantic v2 for type-safe pipelines
- [ ] **FastAPI / Gradio Server** — REST API and web UI for production deployment
- [ ] **Docker & Docker Compose** — Containerized deployment with GPU passthrough
- [ ] **Poetry / uv Package Management** — Modern dependency management with `pyproject.toml`
- [ ] **Comprehensive Test Suite** — Unit and integration tests with `pytest` and `pytest-asyncio`
- [ ] **CI/CD Pipeline** — GitHub Actions for linting, testing, and automated releases

### Evaluation & Observability

- [ ] **RAGAS Evaluation** — Automated RAG evaluation metrics (faithfulness, answer relevancy, context precision)
- [ ] **Custom Benchmarks** — Domain-specific evaluation datasets for Q&A
- [ ] **OpenTelemetry Tracing** — Distributed tracing for production monitoring
- [ ] **LangFuse Integration** — Open-source LLM observability and analytics

---

## Contributing

Contributions are welcome! Whether it's bug fixes, new features, or documentation improvements — every contribution helps grow the RAG and NLP ecosystem.

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

---

## Citation

If you use ProRAG in your research, please cite it:

```bibtex
@software{prorag2024,
  title     = {ProRAG: A Fully Configurable RAG Pipeline for Document Q&A Applications},
  author    = {Abdullah, Al Asif and Al Emon, Hasan},
  year      = {2024},
  url       = {https://github.com/healer-125/pro-rag},
  license   = {MIT}
}
```

Or use the `CITATION.cff` file included in this repository for automatic citation generation on GitHub.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

