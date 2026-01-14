PDF 텍스트 추출기
=================

간단한 Flask 웹앱입니다. PDF 파일을 업로드하면 내부의 텍스트를 추출해서 화면에 보여주고, 텍스트 파일(.txt)로 다운로드할 수 있습니다.

설치
---
1) Python 3.8+ 권장
2) 가상환경 생성 후 의존성 설치

Windows PowerShell 예시:

    py -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

실행
---

    python pdfExtract.py

웹 브라우저에서 http://127.0.0.1:5000 로 접속하면 웹 UI가 나타납니다.

ASGI (uvicorn)로 실행
--------------------
프로덕션 환경에서는 uvicorn(ASGI)으로 실행할 수 있습니다. 이미 `requirements.txt`에 `uvicorn`과 `asgiref`를 추가했습니다.

PowerShell 예시 (환경변수로 스위칭):

    $env:USE_UVICORN = '1'; python pdfExtract.py

또는 uvicorn을 직접 실행:

    uvicorn pdfExtract:asgi_app --host 0.0.0.0 --port 8000

`asgiref`가 없으면 스크립트가 자동으로 Flask 내장 서버로 폴백합니다.

API
---
POST /api/extract (multipart/form-data, key: file)
- 성공: {"success": true, "text": "..."}
- 실패: {"success": false, "error": "..."}

OCR (이미지 기반 PDF 처리)
-------------------------
이미지(스캔) 기반 PDF를 처리하려면 OCR이 필요합니다. Windows에서 설정하는 방법(간단):

1) 시스템에 Tesseract 설치
   - 다운로드: https://github.com/tesseract-ocr/tesseract/releases
   - 설치 후 설치 경로(예: C:\Program Files\Tesseract-OCR) 를 확인하고, 필요하다면 환경변수 PATH에 추가하세요.

2) Poppler 설치 (pdf2image용)
   - Windows용 poppler 설치 파일: https://github.com/oschwartz10612/poppler-windows/releases
   - 압축 해제 후 bin 폴더 경로를 기억해 두세요. 필요 시 환경변수 PATH에 추가하세요.

3) Python 패키지 설치 (가상환경 활성화 후)

    pip install pytesseract pdf2image Pillow

4) spaCy 영어 모델 설치

    python -m spacy download en_core_web_sm

5) 앱 실행 후 OCR 폴백 동작
   - PDF에서 텍스트가 추출되지 않으면 자동으로 OCR을 시도합니다. OCR은 정확도가 문서/언어별로 다릅니다.

추가: PyMuPDF(pymupdf)
--------------------
더 정교한 레이아웃 기반 텍스트 추출을 위해 PyMuPDF(패키명: `pymupdf`, import명: `fitz`)를 사용하도록 업데이트했습니다. 설치는 다음과 같이 합니다:

PowerShell:

    py -m pip install pymupdf

설치 후 앱은 PyMuPDF를 우선 사용해 단어 좌표 기반 리플로우와 블록 단위 추출을 시도합니다. `pymupdf`가 없으면 기존과 같이 `pdfplumber` → `PyPDF2` 순으로 폴백합니다.

주의: `pymupdf`는 바이너리 확장이 포함된 패키지로, 설치 시 인터넷 연결 및 컴파일된 휠을 다운로드할 수 있어야 합니다.

주의
- Windows에서 Tesseract 및 poppler 경로가 PATH에 없으면, `pdf2image.convert_from_bytes` 또는 `pytesseract`가 작동하지 않을 수 있습니다. 환경변수로 경로를 추가하거나, 코드에서 직접 경로를 지정해야 합니다.
- OCR은 시간이 걸리며 CPU/메모리를 많이 사용할 수 있습니다.

커맨드라인 추출 유틸리티 (extract_and_save.py)
---------------------------------------------
프로젝트에 간단한 커맨드라인 유틸리티 `extract_and_save.py`를 추가했습니다. 이 스크립트는 지정한 PDF 파일을 추출하여 같은 폴더에
- <원본명>.extracted.txt  (합쳐진 텍스트, 페이지 헤더 및 이미지 마커 포함)
- <원본명>.extracted.docx (python-docx가 설치되어 있으면 생성)
파일을 생성합니다.

사용 예시 (Windows PowerShell)

1) 가상환경 생성 및 활성화(권장)

    py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1

2) 의존성 설치

    py -3 -m pip install -r .\requirements.txt

3) PDF 추출 및 저장

    py -3 .\extract_and_save.py "C:\Users\ghdwn\Downloads\Africa in Irish Primary Geography Textbooks  developing and applying a Framework to investigate the potential of Irish Primary Geography textbooks in .pdf"

결과 파일은 동일한 폴더에 `.extracted.txt`와 (python-docx가 설치된 경우) `.extracted.docx`로 생성됩니다.

팁
- python-docx가 없으면 DOCX 출력은 생략됩니다. 설치하려면: `py -3 -m pip install python-docx`.
- PyMuPDF(pymupdf)가 필요합니다(더 나은 레이아웃 기반 리플로우). 설치: `py -3 -m pip install pymupdf`.
- OCR(스캔된 이미지 기반 PDF)이 필요하면 Tesseract와 poppler를 먼저 시스템에 설치하고, 관련 Python 패키지(pytesseract, pdf2image, Pillow)를 설치하세요.

주의사항
--------
- 이미지(스캔) 기반 PDF는 OCR이 필요합니다. OCR을 사용하려면 Tesseract 및 추가 패키지(pytesseract, pdf2image 등)를 설치하고 코드를 확장해야 합니다.
- 기본 최대 업로드 크기는 50MB로 설정되어 있습니다.

## 환경 변수 및 비밀 정보

- 민감한 키/설정(예: API 키, 비밀번호 등)은 절대 소스코드에 하드코딩하거나 커밋하지 마세요.
- 로컬에서는 `.env` 파일을 만들고 `.gitignore`에 추가해 관리하세요. 예:

```
# .env 예시
API_KEY=your_api_key_here
```

- CI에 필요한 시크릿은 GitHub 저장소의 `Settings → Secrets and variables → Actions`에 등록하십시오.

## 브랜치 및 협업 규칙

- 기본 브랜치는 `main`입니다. 직접 푸시 대신 Pull Request(PR)로 변경 사항을 병합하세요.
- PR 템플릿을 사용하고, 변경 의도와 테스트 방법을 설명하세요.
- PR은 코드 리뷰(한 명 이상)와 자동 테스트(있을 경우) 통과를 요구합니다.
- `main` 브랜치는 보호 설정(Branch protection)을 적용하여 강제 리뷰 및 CI 통과 없이 병합되지 않도록 권장합니다.

## 라이선스

본 프로젝트는 예시용으로 제공됩니다.
