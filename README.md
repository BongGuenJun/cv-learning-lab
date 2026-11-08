# CV Learning Lab

CV 모델의 원리, 구현, 적용과 비교 실험을 기록하는 학습 저장소입니다.

## 개발 환경

- Ubuntu / WSL
- VS Code / WSL 연결
- Python 3.14.4
- 프로젝트별 가상환경: .venv

## 학습 기록

<details>
<summary>개발 환경 — Linux에서 Python 실행</summary>

### 확인한 내용

- Linux 파일시스템 안에 프로젝트 생성
- Python 가상환경 생성·활성화
- 가상환경의 Python으로 파일 실행

### 발생한 문제와 해결

- 가상환경 생성 중 ensurepip 관련 오류 발생
- python3.14-venv 설치 후 다시 생성하여 해결

### 실행

```bash
source .venv/bin/activate
python check_environment.py