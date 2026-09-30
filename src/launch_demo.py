"""Open the existing local demo or launch it with the selected Python runtime."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser

URL = 'http://127.0.0.1:8765'
ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        with urllib.request.urlopen(URL+'/api/status',timeout=2) as response:
            status = json.load(response)
        if status.get('app') != 'ufm-rec-demo':
            raise RuntimeError('Port 8765 is used by another service. Choose a different --port manually.')
        webbrowser.open(URL)
        return 0
    except urllib.error.URLError:
        pass
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        print('Missing dependencies. Run: py -m pip install -r requirements.txt')
        return 1
    return subprocess.call([sys.executable,'-u',str(ROOT/'src/demo_recommender.py'),
                            '--backend','content','--open-browser'],cwd=ROOT)


if __name__=='__main__':
    sys.exit(main())
