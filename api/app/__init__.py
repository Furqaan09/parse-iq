import os
import sys

# faiss, torch and scikit-learn each bundle their own libomp, and on macOS the
# process aborts when they collide. One OpenMP thread avoids it.
# Temporary: delete this file in PR2, when FAISS and torch are removed.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
