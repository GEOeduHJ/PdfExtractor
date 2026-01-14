# Contributing

감사합니다! 기여는 언제나 환영합니다. 아래 가이드는 협업을 원활하게 하기 위한 기본 규칙입니다.

## 브랜치 규칙
- 기본 브랜치: `main`
- 모든 새로운 기능/수정은 기능 브랜치(feature/*, fix/* 등)로 작업 후 PR 생성
- 브랜치 이름 예: `feature/add-cli`, `fix/handle-empty-pdf`
- `main` 브랜치는 보호(protected)되어야 하며 다음을 권장합니다:
  - Require pull request reviews before merging
  - Require status checks to pass before merging (CI가 설정된 경우)
  - Include administrators 옵션 선택

## Pull Request(푸시 요청)
- PR 제목은 변경 목적을 간결하게 작성
- PR 설명에 변경 사항 요약, 테스트 방법(또는 재현 방법), 관련 이슈 번호 기입
- 코드 스타일, 가독성, 테스트를 중점으로 리뷰합니다.

## 커밋 메시지
- 간단명료하게 작성: `type(scope): short description`
  - 예: `fix(pdf): handle empty pages`
- 필요 시 본문에 상세 변경사항 기재

## 테스트
- 주요 변경사항은 수동/자동 테스트를 통해 동작 확인 권장
- 테스트 스크립트가 있으면 README에 실행 방법을 적어주세요.

## 비밀(Secrets) 및 자격증명
- 절대 민감 정보를 리포지토리에 커밋하지 마세요.
- 로컬: `.env` 파일을 사용하고 `.gitignore`에 추가
- CI: GitHub Secrets를 사용하여 민감 정보를 등록
- 공개 레포지토리인 경우 민감 정보가 노출되면 즉시 토큰/키를 폐기하세요.

## 코드 스타일
- 기존 코드 스타일을 따르세요.
- Python은 PEP8를 준수하는 것이 권장됩니다.

## 기타
- 큰 변경(아키텍처 변경, 외부 API 추가 등)은 이슈로 먼저 논의해주세요.
- 기여해 주셔서 감사합니다!
