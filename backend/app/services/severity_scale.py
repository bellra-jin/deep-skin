"""등급 -> severity 매핑의 정본.

왜 별도 모듈인가
----------------
이 매핑은 두 곳에서 필요하다.
  backend/app/services/multivalue_parser.py   백엔드 파서 (dev JSON 경로)
  backend/scripts/inference_engine.py         AI 서버의 추론 엔진

테이블을 양쪽에 두면 갈라진다. 실제로 그렇게 갈라진 적이 있다 - 눈가 주름의
상한을 파서에서만 고치고 엔진은 그대로 둔 탓에 같은 등급이 경로에 따라 다른
severity 로 나갔다.

파서에서 직접 가져다 쓸 수도 있지만, 파서는 app.models.* 를 통해 SQLAlchemy 를
끌고 온다. AI 서버가 severity 매핑 하나 때문에 DB 스택 전체를 로드해야 한다.
그래서 표준 라이브러리만 쓰는 이 모듈에 테이블을 두고 양쪽이 import 한다.
파서는 이 값을 그대로 재노출하므로 앱 코드 입장에서는 여전히 파서가 정본이다.

어휘를 넓히지 말 것
-------------------
severity 는 normal / mild / moderate / severe 네 개로 고정이다
(backend/docs/ai_inference_contract.md). 소비처가 전부
_SEVERITY_ORDER.get(severity, 0) 패턴이라, 어휘 밖 값은 예외 없이 가장 낮은
등급으로 조용히 강등된다. 최상위 등급이 정렬에서 최하위로 취급된다는 뜻이다.

상한을 낮게 잡지 말 것
----------------------
매핑에 없는 등급은 grade_to_severity() 가 None 을 돌려주고 파서가 그 행을
통째로 버린다("unknown annotation grade skipped"). 사용자에게는 항목이 조용히
사라진다. 문서의 상한을 믿고 만든 매핑이 실제 범위를 못 덮어, 원본 라벨
137,995건 중 4,511건(3.27%)이 그렇게 새고 있었다. 미간 주름은 23.94% 였다.

OBSERVED_MAX_GRADE 가 실측 상한의 정본이고,
backend/tests/test_severity_coverage.py 가 매핑이 그 범위를 전부 덮는지,
그리고 엔진이 이 모듈에 위임하고 있는지를 검사한다.
"""

from __future__ import annotations

# severity 어휘. 넓히면 소비처에서 최저 등급으로 강등된다.
SEVERITY_VOCAB = ("normal", "mild", "moderate", "severe")

# 6등급(0~5) 라벨용. 볼 모공·볼 색소·이마 색소.
ZERO_12_34_5 = {
    0: "normal",
    1: "mild",
    2: "mild",
    3: "moderate",
    4: "moderate",
    5: "severe",
}

# 7등급(0~6) 라벨용. 눈가 주름·미간 주름·이마 주름·턱 처짐.
ZERO_12_345_6 = {
    0: "normal",
    1: "mild",
    2: "mild",
    3: "moderate",
    4: "moderate",
    5: "moderate",
    6: "severe",
}

# 라벨별 실제 등급 범위는 원본 라벨 JSON 112,905건 전수 집계로 확인한 값이다.
# 등급 수가 같은 라벨은 같은 테이블을 재사용한다. 새 어휘를 만들지 않는다.
SEVERITY_BY_ANNOTATION = {
    "forehead_pigmentation": ZERO_12_34_5,       # 0~5
    "forehead_wrinkle": ZERO_12_345_6,           # 0~6
    "glabellus_wrinkle": ZERO_12_345_6,          # 0~6
    "l_perocular_wrinkle": ZERO_12_345_6,        # 0~6
    "r_perocular_wrinkle": ZERO_12_345_6,        # 0~6
    "l_cheek_pore": ZERO_12_34_5,                # 0~5
    "l_cheek_pigmentation": ZERO_12_34_5,        # 0~5
    "r_cheek_pore": ZERO_12_34_5,                # 0~5
    "r_cheek_pigmentation": ZERO_12_34_5,        # 0~5
    "lip_dryness": {0: "normal", 1: "mild", 2: "mild", 3: "moderate", 4: "severe"},
    "chin_sagging": ZERO_12_345_6,               # 0~6
}

# 라벨별 실제 최대 등급 (원본 전수 집계). 테스트가 이 값을 기준으로
# 매핑 누락을 잡는다. 라벨이 늘어나면 여기부터 갱신한다.
OBSERVED_MAX_GRADE = {
    "forehead_pigmentation": 5,
    "forehead_wrinkle": 6,
    "glabellus_wrinkle": 6,
    "l_perocular_wrinkle": 6,
    "r_perocular_wrinkle": 6,
    "l_cheek_pore": 5,
    "l_cheek_pigmentation": 5,
    "r_cheek_pore": 5,
    "r_cheek_pigmentation": 5,
    "lip_dryness": 4,
    "chin_sagging": 6,
}


def grade_to_severity(annotation_key: str, grade: int) -> str | None:
    """등급을 severity 로 옮긴다. 모르는 키나 범위 밖 등급이면 None.

    None 을 "normal" 같은 값으로 대체하지 말 것. 호출부가 그 사실을 알고
    처리해야 한다 - 조용히 뭉개면 잘못된 값이 사용자에게 나간다.
    """
    return SEVERITY_BY_ANNOTATION.get(annotation_key, {}).get(grade)


def num_classes_for(annotation_key: str) -> int | None:
    """그 라벨의 등급 수(= 분류 헤드의 출력 차원). 모르는 키면 None."""
    top = OBSERVED_MAX_GRADE.get(annotation_key)
    return None if top is None else top + 1
