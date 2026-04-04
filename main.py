import os
import fitz
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.panel import Panel
from rich import box

load_dotenv()

client  = Groq(api_key=os.environ.get("GROQ_API_KEY"))
console = Console()

# ── 1. Embedding model (runs locally, no API needed) ─────────────────

# This model runs 100% on your machine — free, fast, no API key.
# It converts text → 384-dimensional vectors.
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ── 2. Vector database ───────────────────────────────────────────────

# ChromaDB persists to disk in ./.chroma
# On restart it loads existing vectors — no re-indexing needed.
chroma_client = chromadb.PersistentClient(path=".chroma")


def get_or_create_collection(name: str):
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}   # use cosine similarity
    )


# ── 3. PDF extraction (same as Project 2) ────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page by page, return list of {page, text} dicts."""
    doc   = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


# ── 4. Chunking ───────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    page: int,
    chunk_size: int  = 500,
    overlap: int     = 50
) -> list[dict]:
    """
    Split text into overlapping chunks.

    chunk_size: characters per chunk (~125 tokens)
    overlap:    characters shared between adjacent chunks
                so context isn't lost at chunk boundaries

    Example with chunk_size=20, overlap=5:
      "The quick brown fox jumps over the lazy dog"
       chunk 1: "The quick brown fox "
       chunk 2: "fox jumps over the l"  ← starts 5 chars back
       chunk 3: "the lazy dog"
    """
    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append({
                "id":     f"{source}_p{page}_c{idx}",
                "text":   chunk,
                "source": source,
                "page":   page,
            })
            idx += 1

        start += chunk_size - overlap   # move forward, minus the overlap

    return chunks


# ── 5. Indexing pipeline ─────────────────────────────────────────────

def index_pdf(pdf_path: str, collection) -> int:
    """
    Full indexing pipeline for one PDF:
    extract → chunk → embed → store in ChromaDB
    Returns number of chunks indexed.
    """
    filename = os.path.basename(pdf_path)
    console.print(f"\n[cyan]Indexing:[/cyan] {filename}")

    # check if already indexed
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        console.print(f"  [dim]Already indexed ({len(existing['ids'])} chunks). Skipping.[/dim]")
        return len(existing["ids"])

    pages      = extract_text_from_pdf(pdf_path)
    all_chunks = []

    for page_data in pages:
        chunks = chunk_text(
            text   = page_data["text"],
            source = filename,
            page   = page_data["page"],
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        console.print("  [red]No text extracted.[/red]")
        return 0

    console.print(f"  [dim]Chunking: {len(pages)} pages → {len(all_chunks)} chunks[/dim]")

    # embed all chunks in one batch (much faster than one at a time)
    texts      = [c["text"]   for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    # store in ChromaDB
    collection.add(
        ids        = [c["id"]     for c in all_chunks],
        documents  = [c["text"]   for c in all_chunks],
        embeddings = embeddings,
        metadatas  = [{"source": c["source"], "page": c["page"]} for c in all_chunks],
    )

    console.print(f"  [green]Indexed {len(all_chunks)} chunks[/green]")
    return len(all_chunks)


# ── 6. Retrieval ─────────────────────────────────────────────────────

def retrieve(query: str, collection, top_k: int = 5) -> list[dict]:
    """
    Convert query to vector, find the top_k most similar chunks.
    Returns list of {text, source, page, distance} dicts.
    """
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta["source"],
            "page":     meta["page"],
            "distance": round(dist, 4),   # lower = more similar
        })

    return chunks


# ── 7. RAG query ─────────────────────────────────────────────────────

def rag_query(query: str, collection) -> str:
    """
    Full RAG query:
    1. Retrieve relevant chunks
    2. Build context string from chunks
    3. Send to LLM with source attribution instruction
    """
    chunks = retrieve(query, collection)

    if not chunks:
        return "No relevant content found in the indexed documents."

    # show what was retrieved (great for learning)
    console.print("\n[dim]Retrieved chunks:[/dim]")
    for i, chunk in enumerate(chunks, 1):
        console.print(
            f"  [dim]{i}. {chunk['source']} p.{chunk['page']} "
            f"(distance: {chunk['distance']})[/dim]"
        )

    # build context block from retrieved chunks
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""You are a document assistant. Answer questions using
only the context provided below. Always cite the source filename and page number.
If the answer is not in the context, say so clearly.

CONTEXT:
{context}"""

    response = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": query},
        ],
        temperature = 0.2,
        max_tokens  = 1024,
    )

    return response.choices[0].message.content


# ── 8. Main ───────────────────────────────────────────────────────────

def main():
    collection = get_or_create_collection("documents")

    console.print(Panel(
        "[bold]RAG Pipeline[/bold]\n"
        "Index PDFs once, query them forever.",
        border_style = "cyan",
        box          = box.ROUNDED,
    ))

    # index all PDFs in ./docs
    pdf_files = [
        f for f in os.listdir("docs") if f.endswith(".pdf")
    ]

    if not pdf_files:
        console.print("\n[red]No PDFs found in ./docs — add some and rerun.[/red]")
        return

    total_chunks = 0
    for pdf_file in pdf_files:
        total_chunks += index_pdf(f"docs/{pdf_file}", collection)

    console.print(f"\n[green]Ready.[/green] {len(pdf_files)} file(s), "
                  f"{total_chunks} chunks total in vector DB.\n")

    # query loop
    console.print("Ask questions about your documents. Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue
        if query.lower() == "quit":
            break

        answer = rag_query(query, collection)
        console.print(Panel(answer, border_style="green", box=box.ROUNDED))
        print()


if __name__ == "__main__":
    main()