#!/usr/bin/env python3
import os
import sys
import json
import datetime
import fitz                                # PyMuPDF
import numpy as np
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- CONFIGURATION ----------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_NAME = "google/long-t5-tglobal-base"
MAX_TOKENS = 16000
OVERLAP_TOKENS = 50
TOP_K = 15

# ---------- UTILITIES ----------
def parse_pdf(path):
    doc = fitz.open(path)
    sections = []
    for pno, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            text = block.get("text", "").strip()
            if not text:
                continue
            # estimate font size
            spans = block.get("lines", [])
            font_sizes = [span["spans"][0]["size"] for span in spans if span.get("spans")]
            font = max(font_sizes) if font_sizes else 0
            sections.append({
                "doc": os.path.basename(path),
                "page": pno,
                "font": font,
                "text": text
            })
    return sections

def chunk_sections(sections):
    chunks = []
    current = {"text": "", "meta": None, "tokens": 0}
    for sec in sections:
        words = sec["text"].split()
        if current["tokens"] + len(words) > MAX_TOKENS:
            chunks.append(current.copy())
            # start overlap
            overlap_words = current["text"].split()[-OVERLAP_TOKENS:]
            current = {"text": " ".join(overlap_words), "meta": sec, "tokens": len(overlap_words)}
        current["text"] += " " + sec["text"]
        current["tokens"] += len(words)
        if current["meta"] is None:
            current["meta"] = sec
    if current["text"]:
        chunks.append(current)
    # attach metadata
    return [
        {
            "doc": c["meta"]["doc"],
            "page": c["meta"]["page"],
            "text": c["text"].strip()
        }
        for c in chunks
    ]

# ---------- MAIN PIPELINE ----------
def main(pdf_paths, persona, job):
    # 1. PARSE PDFs in parallel
    with Pool(min(len(pdf_paths), cpu_count())) as pool:
        all_sections = sum(pool.map(parse_pdf, pdf_paths), [])
    # 2. CHUNK
    chunks = chunk_sections(sorted(all_sections, key=lambda x: (x["doc"], x["page"], -x["font"])))
    texts = [c["text"] for c in chunks]
    # 3. EMBEDDINGS
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    
    # after embedding:
    chunk_vecs = embedder.encode(texts, batch_size=32, show_progress_bar=True)

    # ensure chunk_vecs is 2D
    if chunk_vecs.ndim == 1:
        # only one chunk → make it a batch of 1
        chunk_vecs = chunk_vecs[np.newaxis, :]

    if chunk_vecs.shape[0] == 0:
        print("No text chunks found. Exiting.")
        sys.exit(1)

    
    task_prompt = f"{persona} needs to {job}"
    task_vec = embedder.encode([task_prompt], show_progress_bar=False)[0]
    # 4. RANK
    norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(task_vec)
    sims = (chunk_vecs @ task_vec) / norms
    top_idx = sims.argsort()[-TOP_K:][::-1]
    extracted = []
    for rank, idx in enumerate(top_idx, start=1):
        meta = chunks[idx]
        extracted.append({
            "document": meta["doc"],
            "page_number": meta["page"],
            "section_title": meta["text"].split("\n",1)[0][:100],
            "importance_rank": rank,
            "text": meta["text"]
        })
    # 5. SUMMARISE
    tok = AutoTokenizer.from_pretrained(SUMMARIZER_NAME, use_fast=True)
    summarizer = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_NAME).to("cpu")
    subsection_analysis = []
    for sec in extracted:
        inp = f"Summarise for a {persona} who must {job}: {sec['text']}"
        ids = tok(inp, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).input_ids
        out = summarizer.generate(ids, max_new_tokens=150)
        refined = tok.decode(out[0], skip_special_tokens=True)
        subsection_analysis.append({
            "document": sec["document"],
            "page_number": sec["page_number"],
            "refined_text": refined
        })
    # 6. ASSEMBLE
    result = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.datetime.utcnow().isoformat()
        },
        "extracted_sections": [
            {k:v for k,v in sec.items() if k!="text"} for sec in extracted
        ],
        "subsection_analysis": subsection_analysis
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: analyzer.py PERSONA JOB_TO_BE_DONE file1.pdf [file2.pdf ...]")
        sys.exit(1)
    persona = sys.argv[1]
    job = sys.argv[2]
    pdfs = sys.argv[3:]
    main(pdfs, persona, job)
