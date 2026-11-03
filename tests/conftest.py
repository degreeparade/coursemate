import os, sys
from pathlib import Path
os.environ["EMBED_BACKEND"] = "hash"        # offline embedder -> no model download in CI
os.environ["LLM_BACKEND"] = "extractive"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
