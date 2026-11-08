import sys
from pathlib import Path

print("Python 실행 경로:", sys.executable)
print("현재 작업 폴더:", Path.cwd())
print("Python 버전:", sys.version)