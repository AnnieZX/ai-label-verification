## Minimal Pipeline

Run the placeholder end-to-end pipeline:

```bash
python -m src.pipeline.run_pipeline
```

## Gemini Verifier

Install:

```bash
pip install google-genai pillow
```

Set API key:

```bash
export GEMINI_API_KEY=your_key
```

Optional model override (default: `gemini-2.5-pro`):

```bash
export GEMINI_MODEL=gemini-2.5-pro
```

Run:

```bash
python scripts/test_gemini_sample.py
```
