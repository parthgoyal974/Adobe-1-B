# pdf_to_t5.py
# pip install pymupdf transformers torch --quiet

import fitz                           # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. -------- PDF text extraction -------------------------------------------
def extract_text(pdf_path: str) -> str:
    """Return all text from a PDF as one big string."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages).strip()

# 2. -------- Prompt template -----------------------------------------------
def build_prompt(persona: str, job: str, resume_text: str) -> str:
    """
    You can change the wording here.  The more specific the instructions,
    the better t5-small will behave.
    """
    return (f"You are {persona}.\n"
            f"Job opening: {job}.\n\n"
            "Using the resume below, write a customised cover letter.\n\n"
            "Resume:\n"
            f"{resume_text}\n\n"
            "Cover letter:")

# 3. -------- Main driver ----------------------------------------------------
def main():
    # ---- simple I/O ----
    pdf_path = "South of France - Cities.pdf"
    persona  = input("Persona description : ").strip()
    job      = input("Target job          : ").strip()

    resume_text = extract_text(pdf_path)
    prompt      = build_prompt(persona, job, resume_text)

    # ---- load model & tokenizer ----
    model_name = "t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ---- encode / generate ----
    encoded   = tokenizer(prompt,
                          return_tensors="pt",
                          truncation=True,
                          max_length=512).to(device)

    generated_ids = model.generate(
        **encoded,
        max_length=256,
        num_beams=5,
        early_stopping=True
    )

    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n" + "="*60 + "\nGenerated output:\n")
    print(result)
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
