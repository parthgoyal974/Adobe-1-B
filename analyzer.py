#!/usr/bin/env python3
"""
Plain-text PDF analyser + summariser (single-process, Flan-T5 version)
---------------------------------------------------------------------
1. Extract text from PDFs and concatenate it page-wise.
2. Break the big paragraph into overlapping word chunks.
3. Embed the chunks (all-MiniLM-L6-v2).
4. Rank them against a persona-and-task prompt.
5. Summarise the TOP_K chunks with google/flan-t5-small.
6. Emit JSON on stdout.
"""

import os, sys, json, datetime
import fitz                         # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# ───────── CONFIG ─────────
EMBED_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_NAME    = "google/flan-t5-small"        # ← switched here
SUMMARY_IN_TOKENS  = 512                           # Flan-T5 input limit
MAX_CHUNK_WORDS    = 512                        # for embedding only
OVERLAP_TOKENS     = 50
TOP_K              = 15

# ───────── UTILITIES ─────────
def pdf_to_paragraph(path: str) -> str:
    """Return all selectable text in the PDF as one long paragraph."""
    doc = fitz.open(path)
    page_texts = [page.get_text("text") for page in doc]
    joined = " ".join(t.replace("\n", " ") for t in page_texts)
    return " ".join(joined.split())                 # collapse whitespace


def chunk_text(text: str, doc_name: str):
    """Split a long paragraph into overlapping word chunks."""
    words, chunks, start, cid = text.split(), [], 0, 0
    while start < len(words):
        end   = min(len(words), start + MAX_CHUNK_WORDS)
        chunk = " ".join(words[start:end])
        chunks.append({"doc": doc_name, "chunk_id": cid, "text": chunk})
        cid  += 1
        start = end - OVERLAP_TOKENS
        if start >= len(words): break
    return chunks

# ───────── MAIN PIPELINE ─────────
def main(pdf_paths, persona, job):
    # 1. Read PDFs
    paragraphs = [
        (os.path.basename(p), pdf_to_paragraph(p)) for p in pdf_paths
    ]
    paragraphs = [(n, t) for n, t in paragraphs if t]
    if not paragraphs:
        print("No extractable text found in the PDFs. Exiting."); sys.exit(1)

    # 2. Chunk
    chunks = []
    for name, text in paragraphs:
        chunks.extend(chunk_text(text, name))
    texts = [c["text"] for c in chunks]
    print(f"✓ Created {len(chunks)} chunks from {len(pdf_paths)} PDFs")

    # 3. Embeddings
    embedder   = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    chunk_vecs = embedder.encode(texts, batch_size=32, show_progress_bar=True)

    # 4. Similarity ranking
    task_prompt = f"{persona} needs to {job}"
    task_vec    = embedder.encode([task_prompt], show_progress_bar=False)[0]
    sims = (chunk_vecs @ task_vec) / (
        np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(task_vec)
    )
    top_idx = sims.argsort()[-TOP_K:][::-1]
    extracted = [
        {
            "document": chunks[i]["doc"],
            "chunk_id": chunks[i]["chunk_id"],
            "importance_rank": r + 1,
            "text": chunks[i]["text"],
        }
        for r, i in enumerate(top_idx)
    ]

    # 5. Summarisation with Flan-T5-small
    tok  = AutoTokenizer.from_pretrained(SUMMARIZER_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_NAME).to("cpu")

    subsection_analysis = []
    for sec in extracted:
        prompt = f"Summarise for a {persona} who must {job}: {sec['text']}"
        ids = tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=SUMMARY_IN_TOKENS
        ).input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=150)
        refined = tok.decode(out[0], skip_special_tokens=True)
        subsection_analysis.append(
            {"document": sec["document"],
             "chunk_id": sec["chunk_id"],
             "refined_text": refined}
        )

    # 6. Output JSON
    result = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.datetime.utcnow().isoformat(),
        },
        "extracted_sections": [
            {k: v for k, v in sec.items() if k != "text"} for sec in extracted
        ],
        "subsection_analysis": subsection_analysis,
    }
    print(json.dumps(result, indent=2))

# ───────── CLI ─────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: analyzer.py PERSONA JOB_TO_BE_DONE file1.pdf [file2.pdf ...]")
        sys.exit(1)
    persona, job, pdfs = sys.argv[1], sys.argv[2], sys.argv[3:]
    main(pdfs, persona, job)
