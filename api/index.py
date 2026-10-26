import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
