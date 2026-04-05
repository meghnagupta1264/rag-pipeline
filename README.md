# RAG Pipeline

A full Retrieval Augmented Generation pipeline built from scratch. Index any number of PDFs into a local vector database, then query across all of them with semantic search. Built with Groq, ChromaDB, and local sentence embeddings — no OpenAI needed for the retrieval step.

---

## Why RAG exists

The naive approach to document Q&A (Project 2) stuffs the entire PDF into the context window on every message. This breaks at scale:

```
Problem:
  500 page document = ~400,000 tokens
  LLM context window = 128,000 tokens max
  400,000 > 128,000 → doesn't fit → completely breaks

Even if it fit:
  Sending 400k tokens every message = very slow + expensive
```

RAG solves this by only sending the relevant parts per query:

```
INDEXING (done once):
  PDF → split into chunks → embed each chunk → store vectors in DB

QUERYING (done per question):
  Question → embed → find similar vectors → retrieve top 5 chunks only
  → [system prompt + 5 chunks + question] → LLM → answer
```

Instead of sending the whole document, you send ~2,000 tokens of the most relevant content. Faster, cheaper, and scales to unlimited document size.

---

## How embeddings work

Meaning is represented as a list of numbers (a vector). Semantically similar text produces similar vectors — even with completely different words:

```
"I love pizza"     → [0.2, 0.8, 0.1, 0.9, ...]
"I enjoy pizza"    → [0.2, 0.7, 0.1, 0.8, ...]  ← close → high similarity
"The stock fell"   → [0.9, 0.1, 0.7, 0.2, ...]  ← far    → low similarity
```

Similarity is measured with cosine distance. Lower distance = more relevant chunk. This is semantic search — it finds meaning, not just matching keywords.

---

## Full pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  INDEXING (runs once per document)                          │
│                                                             │
│  PDF → extract text → split into 500-char chunks            │
│      → embed with all-MiniLM-L6-v2 (local, free)           │
│      → store vectors + metadata in ChromaDB                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  QUERYING (runs per question)                               │
│                                                             │
│  Question → embed → cosine search in ChromaDB              │
│           → retrieve top 5 chunks with page + source       │
│           → build context string                           │
│           → send to Groq LLM with grounding instruction    │
│           → answer with citations                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM Provider | [Groq](https://groq.com) (free tier) |
| LLM Model | LLaMA 3.3 70B Versatile |
| Embedding Model | all-MiniLM-L6-v2 (runs locally, free) |
| Vector Database | ChromaDB (local, persistent) |
| PDF Extraction | PyMuPDF (fitz) |
| Terminal UI | Rich |

---

## Setup

1. Clone the repo
2. Create and activate a virtual environment
3. Install dependencies: pip install groq python-dotenv pymupdf chromadb sentence-transformers rich
4. Get a Groq API key
5. Create your `.env` file
6. Add PDFs

---

## Usage

```bash
python main.py
```

The pipeline automatically indexes all PDFs in `docs/` on first run. On subsequent runs it detects existing chunks and skips re-indexing. Add new PDFs to `docs/` and rerun — only the new files get indexed.

```
Indexing: attention.pdf
  Chunking: 15 pages → 96 chunks
  [████████████] 100%
  Indexed 96 chunks

Ready. 1 file(s), 96 chunks total in vector DB.

You: What optimizer was used to train the model?

  Retrieved chunks:
  1. attention.pdf p.7 (distance: 0.4806)
  2. attention.pdf p.8 (distance: 0.5566)
  ...

╭──────────────────────────────────────────────────────╮
│ The Adam optimizer was used, with β1 = 0.9,          │
│ β2 = 0.98, and ε = 10-9.                             │
╰──────────────────────────────────────────────────────╯
```

---

## Test results on the Attention Is All You Need paper

| Query | Top distance | Result |
|---|---|---|
| BLEU scores on WMT 2014 | 0.35 | Correct — full comparison table from p.8 |
| How many encoder layers | 0.37 | Correct — N=6 from p.3 |
| How does attention work | 0.38 | Correct — formal definition from p.3 |
| What problem does Transformer solve | 0.48 | Correct — from p.2 |
| What optimizer was used | 0.48 | Correct — Adam with exact hyperparameters |
| Dimensionality of the model | 0.61 | Correct — dmodel=512 despite weak retrieval |
| Why was recurrence removed | 0.59 | Correctly refused — reason implicit across sections |

The pipeline correctly refused to hallucinate on the recurrence question — it said the exact reason wasn't in the retrieved context rather than making something up. This is the grounding instruction working as intended.

---

## Key concepts demonstrated

Chunking with overlap — documents are split into 500-character chunks with 50-character overlap between adjacent chunks. The overlap prevents information loss at chunk boundaries — a sentence split across two chunks still appears fully in at least one of them.

Local embeddings — the `all-MiniLM-L6-v2` model runs entirely on your machine. No API call, no cost, no rate limits for the retrieval step. Only the final answer generation hits the Groq API.

Persistent vector store — ChromaDB saves vectors to `.chroma/` on disk. Restart the program and your index is instantly available — no re-indexing. Delete `.chroma/` to force a fresh index.

Distance scores as quality signals — every retrieved chunk shows its cosine distance. Below `0.3` is a strong match. Above `0.5` is weak. Watching these scores teaches you when to trust the answer and when to be skeptical — a critical skill for production RAG systems.

Semantic search vs keyword search — the retriever finds relevant chunks even when the query uses different words than the document. "How does attention work" retrieves chunks about "query-key-value mapping" because the vectors are semantically close, not because the words match.

Grounding — the system prompt explicitly instructs the model to only answer from retrieved context and to cite sources. This prevents hallucination and makes answers auditable — you can verify every claim against the page number provided.

---

## Limitations and what comes next

Conceptual questions retrieve poorly — questions like "why did the authors do X" require reasoning across multiple sections. Single-chunk retrieval misses the full picture. Production systems address this with query rewriting and multi-hop retrieval.

Fixed chunk size is naive — splitting every 500 characters ignores document structure. A paragraph boundary mid-sentence loses context. Advanced chunking strategies (recursive, semantic, by heading) produce cleaner chunks.

No reranking — the top-5 results are returned by vector similarity alone. A reranker model (like Cohere Rerank) adds a second pass to reorder by true relevance, significantly improving answer quality.

Single-turn only — each query is independent. The pipeline doesn't maintain conversation history across questions. Adding memory so follow-up questions work ("what about the decoder?") is the next natural extension.

---

## Experiments worth running

1. Change chunk size — delete `.chroma/` and re-index with `chunk_size=200`. Ask the same questions. Smaller chunks are more precise but lose surrounding context. Larger chunks keep context but retrieve noisier results.
<img width="2788" height="1194" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/fe929fe8-fe97-46e0-ac25-0709a0380c11" />
<img width="2784" height="1176" alt="Pasted Graphic 1" src="https://github.com/user-attachments/assets/b7f7c3de-6036-490e-b971-c8569987462f" />

2. Multi-document retrieval — add two or three different PDFs to `docs/`. Ask questions that span both. Watch the retriever pull chunks from multiple sources and the LLM synthesize across them.
<img width="2788" height="1222" alt="image" src="https://github.com/user-attachments/assets/4c878734-1536-492d-a9c3-16fc01d62122" />
<img width="2786" height="1238" alt="image" src="https://github.com/user-attachments/assets/282be3f5-2124-47c5-acea-dc176b3deaf0" />
