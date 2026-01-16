PDF 텍스트 추출기
=================

PDF 텍스트 추출기 (Streamlit UI)
-------------------------------
이 리포지토리는 Streamlit 기반의 간단한 PDF 텍스트 추출 도구입니다. PDF 파일을 업로드하면 내부 텍스트를 추출하고 다운로드하거나 (선택적으로) 번역할 수 있습니다.
추가: PyMuPDF(pymupdf)
--------------------
이 프로젝트는 PyMuPDF(`pymupdf`, import명 `fitz`)를 주 추출 엔진으로 사용합니다. 설치:

```powershell
py -m pip install pymupdf
```

`pymupdf`가 설치되어 있으면 레이아웃 기반 리플로우와 단어 좌표를 이용한 더 나은 텍스트 추출을 수행합니다.
앱 실행
---

이 프로젝트는 Streamlit UI가 기본으로 사용됩니다.

```powershell
streamlit run app_streamlit.py
# 브라우저에서 Streamlit이 제공하는 로컬 주소를 열어 사용
```

`app_streamlit.py`는 내부적으로 `pdfExtract`의 `extract_pages_from_pdf_bytes`와 번역 헬퍼(`get_translator`, `safe_translate`)를 사용합니다.

API
---
POST /api/extract (multipart/form-data, key: file)
- 성공: {"success": true, "text": "..."}
- 실패: {"success": false, "error": "..."}

추가: PyMuPDF(pymupdf)
--------------------
이 프로젝트는 PyMuPDF(`pymupdf`, import명 `fitz`)를 주 추출 엔진으로 사용합니다. 설치:

```powershell
py -m pip install pymupdf
```

`pymupdf`가 설치되어 있으면 레이아웃 기반 리플로우와 단어 좌표를 이용한 더 나은 텍스트 추출을 수행합니다.

커맨드라인 유틸리티
------------------
이 리포지토리는 Streamlit 우선으로 단순화되었습니다. 이전에 포함되었던 CLI(`extract_and_save.py`)는 비활성화되어 있습니다. 필요하면 Streamlit UI 또는 `pdfExtract.py`의 함수를 직접 호출하는 스크립트를 작성하세요.

주의사항
--------
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
