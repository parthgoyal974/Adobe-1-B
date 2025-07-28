T5-powered PDF Summarizer
------------------------------------------

**`analyzer.py` expects only two string arguments (persona + job) and automatically reads every *.pdf found in `app/input/`.**
The generated reply is saved as `app/output/<persona>.txt`, where spaces in *persona* are replaced by underscores.

## Quick start

```bash
# 1. Install once
pip install pymupdf transformers torch

# 2. Place your source PDFs in:
#    app/input/
#       ├─ cv.pdf
#       └─ portfolio.pdf

# 3. Run
python analyzer.py "<persona>" "<job description>"
```


### Required arguments

| Position | `sys.argv` index | Example value | Purpose |
| :-- | :-- | :-- | :-- |
| 1 | `sys.argv` | `"Hiring Manager"` | **Persona** – who the model should pretend to be (spaces allowed). |
| 2 | `sys.argv` | `"Write a cover-letter for Data Scientist role"` | **Job / task** the model must perform. |

*No* PDF paths follow on the command line—every PDF present in **`app/input/`** is processed.

## What the script does

1. Scans `app/input/` for every *.pdf*.
2. Extracts plain text from each file and merges the results.
3. Builds a prompt:

```
You are <persona>.
Job to do by you: <job>.
...
```

4. Sends the prompt to **T5-small** and generates a tailored response.
5. Writes the output to `app/output/<persona_with_underscores>.txt`.
Example: `Hiring_Manager.txt`

## Example

```bash
# PDFs: app/input/South of France - Cities.pdf
python analyzer.py "Travel Planner" "Plan a 4-day trip for 10 college friends"
```

After finishing, check:

```
app/output/<persona>.txt
```

It will contain the complete 4-day itinerary crafted from the PDF content.

# Build Docker Image using

```docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .```

# Run Docker Image using 

```docker run --rm \
  -v $(pwd)/app/input:/app/input \
  -v $(pwd)/app/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier \
  "<persona>" "<job description>"
```
