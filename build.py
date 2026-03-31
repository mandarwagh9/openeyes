import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "mkdocs-material"])
subprocess.check_call([sys.executable, "-m", "mkdocs", "build"])