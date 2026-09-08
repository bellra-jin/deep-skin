# Deep Skin — AI 피부 분석 · 맞춤 스킨케어 추천

## 데이터 전처리 · 부위별 등급 분류 모델 · AI 추론 연동

> 얼굴 이미지 한 장에서 부위별 피부 상태를 등급으로 정량화하고, 그 결과를 성분 추천까지 연결하는 서비스입니다.
> **"모델이 안 나오는 이유를 데이터에서 먼저 찾았다"** — 라벨 경계, 클래스 불균형, 크롭 비율을 차례로 좁혀 가며 ResNet-50의 한계를 진단하고 다음 백본으로 넘긴 기록입니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | Deep Skin — 20-40 여성을 위한 맞춤 뷰티 AI Model |
| 진행 기간 | 2026.05.07 ~ 2026.05.15 |
| 인원 | 4인 팀 프로젝트 |
| 담당 역할 | **부위별 크롭 전처리**, **등급 분류 모델 학습·한계 분석**, **AI 추론 연동(계약·파싱·E2E)**, 서비스 백엔드/프론트 |
| 데이터 | AI-Hub 한국인 피부상태 측정 데이터 (원천 이미지 + 라벨 JSON) |

AI-Hub 데이터셋의 부위별 bbox와 전문가 진단 등급을 학습해 이마·미간·눈가·볼·입술·턱의 피부 상태를 판정하고, 판정 결과로 성분 추천을 생성합니다. 이 README는 **제가 맡은 데이터 전처리와 모델 학습, 그리고 모델 출력을 서비스로 옮기는 구간**을 중심으로 정리했습니다.

---

## 2. 시연 영상

[![Deep Skin 시연](https://img.youtube.com/vi/FYQIVGGnxXY/maxresdefault.jpg)](https://youtu.be/FYQIVGGnxXY)

> 이미지를 클릭하면 YouTube에서 재생됩니다. 이미지 업로드 → AI 추론 → 부위별 등급·측정값 리포트 → 성분 추천까지의 흐름을 담았습니다.

---

## 3. 문제 정의

### 3.1 피부 등급 라벨은 경계가 모호하다

AI-Hub 라벨은 전문가가 매긴 0~5등급입니다. 문제는 **인접 등급의 경계가 사람이 봐도 모호하다**는 것입니다. 2등급과 3등급의 차이를 모델에게 구분하라고 요구하면, 실제로는 존재하지 않는 경계를 학습하게 됩니다.

여기에 **클래스 불균형**이 겹칩니다. 특정 등급에 데이터가 거의 없어서, 정확도만 보면 잘 맞히는 것처럼 보이지만 희소 등급은 아예 예측하지 못하는 모델이 나옵니다.

### 3.2 부위마다 크롭 크기와 비율이 다르다

같은 얼굴 이미지에서 잘라내도 부위별로 원본 픽셀 크기와 가로세로 비율이 크게 다릅니다.

- **미간** — 영역이 작아 확대하면 화질이 깨져 학습을 방해
- **이마** — 가로로 긴 영역이라 정사각형으로 resize하면 찌그러짐 발생

부위 특성을 무시하고 같은 전처리를 적용하면, 모델이 피부 상태가 아니라 **왜곡 패턴을 학습**하게 됩니다.

### 3.3 설계 원칙

- **모델을 바꾸기 전에 데이터를 의심한다** — 성능이 안 나오면 하이퍼파라미터부터 만지지 않고 라벨 정의·분포·전처리를 먼저 본다.
- **accuracy 하나로 판단하지 않는다** — 불균형 데이터에서는 macro-F1을 기준 지표로 둔다.
- **분류 라벨과 측정 수치를 섞지 않는다** — `annotations`(등급)와 `equipment`(장비 측정값)는 끝까지 다른 축으로 다룬다.
- **개선 시도는 실패해도 기록한다** — 되돌린 실험도 근거와 함께 남긴다.

---

## 4. 담당 범위와 전체 흐름

```text
AI-Hub 원천 이미지 + 라벨 JSON
        │
        ▼
 ① 부위별 크롭 전처리          ← 담당
    bbox · 각도 필터 · 품질 필터 · 메타데이터 CSV
        │
        ▼
 ② ID 그룹 분할                ← 담당
    같은 사람이 train/val에 동시에 들어가지 않게
        │
        ▼
 ③ 등급 분류 모델 학습          ← 담당
    ResNet-50 · 불균형 대응 · 3등급 전환 실험 · 한계 진단
        │
        ▼
 ④ AI 추론 서버 (팀 공동 산출물)
    부위 검출 + 멀티태스크 추론
        │
        ▼
 ⑤ 추론 계약 · 응답 파싱 · E2E 검증   ← 담당
    등급 / 측정값 / 검출 결과를 분리해 DB에 적재
        │
        ▼
 ⑥ 리포트 · 성분 추천 (FastAPI + Streamlit)  ← 담당
```

```text
notebooks/jh/          크롭 · 분할 · 학습 · 진행상황 조회 · 파이프라인 테스트
docs/cheek/            학습 전제 · 크롭 규격 · 검증 기준 설계 문서
docs/labeling_codes_guide.md   AI-Hub 라벨 코드값 해설
configs/               학습 실험 설정 JSON
backend/               FastAPI — 추론 호출 · 파싱 · 저장 · 추천 · 리포트
frontend/              Streamlit — 업로드 · 리포트 · 추천 화면
```

---

## 5. 데이터 전처리 파이프라인

`notebooks/jh/crop.py` · `docs/cheek/cheek_crop_pipeline_guide.md`

### 5.1 부위별 bbox 크롭과 각도 필터

AI-Hub 이미지는 7개 촬영 각도로 찍혀 있습니다. 문제는 **부위마다 가려지는 각도가 다르다**는 것입니다. 얼굴이 오른쪽으로 돌면 왼쪽 볼이 가려지고, 그 이미지를 학습에 넣으면 모델은 볼이 아닌 것을 볼이라고 배웁니다.

| 부위 | 사용 각도 | 제외 각도 | 제외 이유 |
|---|---|---|---|
| 왼쪽 볼 | 정면 · 위 · 아래 · L15° · L30° | R15° · R30° | 얼굴이 오른쪽으로 돌면 왼볼이 가려짐 |
| 오른쪽 볼 | 정면 · 위 · 아래 · R15° · R30° | L15° · L30° | 얼굴이 왼쪽으로 돌면 오른볼이 가려짐 |

위·아래 각도는 제외하지 않았습니다. **조명과 각도 변화는 노이즈가 아니라 일반화에 필요한 다양성**이라고 봤기 때문입니다.

크롭 자체에도 세 가지 규칙을 넣었습니다.

```text
1. 파일명(_05, _06)을 믿지 않고 JSON 내부 images.facepart 값으로 부위를 확정
2. bbox에 각도별 margin(ANGLE_MARGIN)을 적용해 주변 맥락을 남긴 채 확장
3. 각도별 최소 해상도 규칙(ANGLE_QUALITY_RULES)으로 저품질 크롭을 걸러냄
```

margin을 각도별로 다르게 준 이유는 3.2에서 확인한 **부위·각도별 크롭 크기 편차** 때문입니다. 고정 비율로 자르면 어떤 각도에서는 경계가 잘리고, 어떤 각도에서는 배경이 과하게 들어옵니다.

### 5.2 ID 그룹 분할 — 데이터 누수 차단

`notebooks/jh/rebalance_split.py`

같은 사람의 사진이 train과 val에 동시에 들어가면 검증 점수가 부풀려집니다. 한 사람당 여러 각도·장비로 촬영된 데이터셋이라 **랜덤 분할은 곧 데이터 누수**입니다.

```text
1. train/val CSV를 합쳐 사람(id) 단위로 그룹화
2. 그룹별 라벨 분포를 집계
3. 희소 클래스를 먼저 덮도록 그룹을 선택하며 val 세트 구성
4. 사람 단위를 유지한 채 라벨 분포를 맞춘 CSV 재생성
```

무작위 분할이 아니라 **희소 클래스 커버리지를 우선하는 그리디 선택**입니다. 그냥 사람 단위로만 나누면 희소 등급이 한쪽에 몰려 검증이 불가능해지기 때문입니다.

### 5.3 라벨 체계 정리 — 분류와 회귀를 섞지 않는다

`docs/labeling_codes_guide.md` · `docs/cheek/cheek_training_prerequisites.md`

라벨 JSON에는 두 종류의 값이 같은 이름으로 들어 있습니다.

| JSON 영역 | 의미 | 학습 유형 | 예 |
|---|---|---|---|
| `annotations` | 전문가 진단 등급 | Classification | `l_cheek_pore` = 0~5 등급 |
| `equipment` | 장비 측정 수치 | Regression | `l_cheek_pore` = 측정값 |

**이름은 같은데 의미가 다릅니다.** 이걸 혼동한 채 학습을 돌리면 결과 전체가 무의미해지므로, 코드를 쓰기 전에 문서로 먼저 고정했습니다. 1차 학습은 `annotations`만 사용하고 `equipment`는 제외했습니다.

이 구분은 나중에 백엔드 DB 설계까지 그대로 이어집니다(§7.2).

---

## 6. 등급 분류 모델 학습

`notebooks/jh/train.py` · `model.py` · `configs/`

담당 부위는 **양쪽 볼(facepart 5·6)**, 예측 대상은 모공·색소침착 등급입니다. 아래 학습 코드와 설정은 볼 부위 파이프라인 기준입니다.

> **수치 출처** — 6.2~6.4의 성능 수치는 팀 발표 자료에 기록된 **3등급 전환 실험(검증 세트 기준)** 결과입니다. 부위별로 분해된 기록이 남아 있지 않아, 특정 부위의 성능으로 표기하지 않았습니다.

### 6.1 ResNet-50 학습 설계

```text
backbone    ResNet-50 (ImageNet 사전학습 가중치)
head        2048 → 512 (BatchNorm + ReLU + Dropout) → num_classes
optimizer   AdamW (weight_decay 1e-3)
scheduler   ReduceLROnPlateau
early stop  macro-F1 기준
```

백본은 `freeze` / `unfreeze`를 전환할 수 있게 분리했습니다. 데이터가 적을 때는 head만 학습시키고, 충분해지면 전체를 미세조정하는 두 전략을 같은 코드로 비교하기 위해서입니다.

실험 설정은 코드가 아니라 JSON으로 분리해, 조건만 바꿔 가며 재현할 수 있게 했습니다.

```jsonc
// configs/pore_run11.json
{ "target": "pore", "loss": "focal", "focal_gamma": 3.0, "oversample": true,
  "epochs": 80, "early_stopping_patience": 20, "batch_size": 8,
  "learning_rate": 1e-4, "weight_decay": 1e-3, "dropout_p": 0.3,
  "label_smoothing": 0.05 }
```

### 6.2 클래스 불균형 대응 — accuracy에 속지 않기

불균형 데이터에서 accuracy는 **다수 클래스만 맞혀도 높게 나옵니다.** 실제로 초기 실험에서 이 격차가 그대로 드러났습니다.

| 실험 조건 | Mean Macro F1 | Mean Accuracy |
|---|---:|---:|
| 3등급 · quick check | **0.303** | 0.673 |

정확도 0.673은 그럴듯해 보이지만 macro-F1 0.303은 **희소 등급을 거의 못 맞히고 있다**는 뜻입니다. 그래서 기준 지표를 macro-F1으로 고정하고, 불균형 대응을 두 겹으로 넣었습니다.

- **손실 함수** — `FocalLoss`(γ 조정)로 쉬운 다수 클래스의 기여도를 낮춤. 또는 label smoothing + class weight
- **샘플링** — `WeightedRandomSampler`로 배치 단위 클래스 균형
- **조기 종료** — accuracy가 아니라 macro-F1이 개선되지 않을 때 중단

### 6.3 5등급 → 3등급 전환 실험

모델이 아니라 **라벨 정의를 바꾸는 실험**을 했습니다. 0~5의 6개 등급을 **양호 / 보통 / 주의** 3등급으로 묶은 것입니다.

**근거**

- 인접 등급 간 경계가 모호해 모델이 존재하지 않는 차이를 학습해야 함
- 특정 등급의 데이터 부족으로 클래스 불균형이 심화됨
- 5등급 예측은 결과가 흔들려 신뢰도와 해석 용이성이 떨어짐
- 서비스에서 사용자에게 보여줄 단위도 결국 "양호/보통/주의"

**결과 (검증 세트 기준)**

| 실험 조건 | Mean Macro F1 | Mean Accuracy |
|---|---:|---:|
| 3등급 · ordinal optimized | **0.665** | **0.728** |
| 3등급 · balanced 보정 | 0.644 | 0.703 |
| 3등급 · quick check | 0.303 | 0.673 |

등급 병합으로 macro-F1이 0.303 → 0.665까지 올라갔습니다. **모델이 아니라 문제 정의를 바꿔서 얻은 개선**입니다.

반대로 **추가 균형 보정은 오히려 성능을 떨어뜨렸습니다**(0.665 → 0.644). 등급을 이미 3개로 합친 뒤에는 분포가 어느 정도 평탄해졌기 때문에, 그 위에 균형 보정을 또 얹으면 다수 클래스의 정상 신호까지 깎이는 것으로 보입니다. 이 실험은 되돌렸지만 **"해봤는데 안 됐다"는 근거와 함께 기록**했습니다.

### 6.4 한계 진단과 다음 백본으로의 전환

3등급 전환 이후에도 학습 곡선은 더 올라가지 않았습니다.

```text
3-grade ordinal ResNet-50 training curve (16 epochs, 검증 세트)

EPOCH 4 이후   val loss 정체 시작
EPOCH 11       best — macro-F1 0.665
EPOCH 11 이후  train / val macro-F1 간격이 계속 벌어짐 → 과적합
```

**학습 데이터는 잘 맞지만 새로운 데이터로의 일반화가 제한되는 패턴**입니다. 여기서 하이퍼파라미터를 더 만지는 대신 백본 자체의 한계로 판단했습니다.

| ResNet-50의 한계 | DINOv3 확장 필요성 |
|---|---|
| CNN 기반이라 미세한 피부 질감·색 변화 표현에 한계 (조명·그림자 영향) | 얼굴 여러 부위를 동일한 구조로 확장 가능 |
| Epoch 11 이후 검증 성능 개선 정체 | 소량·불균형 데이터에서 일반화 개선 가능성 |
| 부위별로 모델을 따로 학습해야 함 | 자기지도학습 — 라벨이 부족해도 학습 가능 |

이 진단을 근거로 팀은 백본을 **DINOv3 멀티태스크 구조로 전환**했고, 최종 서비스 추론은 그 모델이 담당합니다. 제 기여는 **전환 결정의 근거가 된 한계 진단과 실험 기록**까지입니다.

| | ResNet-50 | DINOv3 (팀 전환 결과) |
|---|---|---|
| 사전학습 데이터 | 140만 장 | 17억 장 |
| 학습 방식 | 지도학습 (라벨 필요) | 자기지도학습 |
| 백본 처리 | Fine-tuning | Frozen + Caching |
| 처리 부위 | 부위별 개별 모델 | 8개 부위 통합 |
| 작업 유형 | 분류 | 분류 + 개수 + 회귀 |

---

## 7. AI 추론 연동

모델이 아무리 잘 나와도 **출력을 서비스가 받아 쓸 수 없으면 의미가 없습니다.** 학습과 서비스 사이를 잇는 구간을 맡았습니다.

### 7.1 추론 계약

`backend/docs/ai_inference_contract.md` · `backend/app/schemas/image_upload.py`

AI 추론 서버와 백엔드 사이의 응답 구조를 **스키마로 고정**했습니다. 모델이 바뀌어도 이 계약만 지키면 서비스 코드는 그대로입니다.

```text
POST {AI_INFERENCE_URL}   multipart/form-data
  file · session_id · user_id · image_id

응답 → InferenceResult 로 Pydantic 검증
  model_name · model_version · parts[]

parts[] 필수      raw_part_name · display_part_name · metric_name
                  issue_type · grade_value(int) · severity · confidence_score
parts[] 선택      predicted_value · measured_value      ← Optional, 하위 호환
```

`AI_INFERENCE_MODE` 하나로 추론 소스를 전환합니다.

| 모드 | 동작 |
|---|---|
| `mock` | 외부 호출 없이 고정 결과 반환 — 서버 없이 서비스 흐름 개발/테스트 |
| `remote` | 단일값 응답 서버 호출 후 스키마 검증 |
| `multivalue` | 부위 검출 + 멀티태스크 추론 서버 호출 — **현재 운영 모드** |

`scripts/dummy_ai_server.py`는 이 계약을 그대로 반환하는 검증용 목 서버입니다. 모델이 학습 중일 때도 서비스 계층을 계속 검증할 수 있었고, 실제 서버로 전환할 때 바뀐 것은 **URL과 파서뿐**이었습니다.

### 7.2 응답 파싱 — 등급 · 측정값 · 검출을 분리 저장

`backend/app/services/multivalue_parser.py`

추론 응답 한 건을 성격에 따라 네 테이블로 쪼갭니다. §5.3에서 정리한 **분류/회귀 구분이 그대로 DB 스키마가 된 것**입니다.

```text
AI 응답 1건
  ├─ annotations  → skin_part_results     등급값 (분류)
  ├─ equipment    → skin_metric_values    연속 측정값 (회귀)
  ├─ detections   → skin_part_detections  부위 bbox + bbox_source
  └─ 원본 JSON    → ai_raw_responses      전체 보존
```

같은 실수라도 출처가 다르면 다른 컬럼에 넣습니다.

| 컬럼 | 출처 | 저장 경로 |
|---|---|---|
| `grade_value` | 등급 판정 (0~3) | 공통 |
| `predicted_value` | 모델이 예측한 회귀값 | 이미지 업로드 추론 |
| `measured_value` | 장비·AI-Hub JSON 원본 측정값 | 개발용 JSON 입력 |

**부위 검출에 실패한 경우도 버리지 않고 기록합니다.** 검출된 부위는 `bbox_source = yolo`, 미검출로 전체 이미지를 사용한 부위는 `full_image_fallback`으로 구분 저장합니다. 값이 같아도 신뢰도가 다르기 때문에, "이 수치를 믿어도 되는가"를 나중에 판단할 근거를 남겨 둔 것입니다.

### 7.3 E2E 검증

`backend/docs/remote_ai_e2e_test_result.md`

업로드부터 리포트 렌더링까지 전 구간을 실제 이미지로 검증하고, DB에 무엇이 몇 건 들어갔는지 표로 남겼습니다.

```text
이미지 업로드 → 백엔드 이미지 검증 → AI 서버 추론 요청
  → 부위 검출 (미검출 부위는 전체 이미지 fallback)
  → 멀티태스크 추론
→ 응답 파싱 → 4개 테이블 저장 → 추천 생성 → 리포트 API → 화면 표시
```

검증 과정에서 **학습 라벨에 없어 항상 0으로 채워지던 지표(`chin_moisture`)를 발견해 응답과 저장 대상에서 제거**했습니다. 값이 있는 것처럼 보이지만 실제로는 추론되지 않은 값이라, 그대로 두면 사용자에게 거짓 정보를 주게 됩니다.

같은 이유로 등급 `0`이 **"미검출"인지 "이상 없음"인지**를 `bbox_source`와 함께 구분해 판정했습니다.

---

## 8. Backend · Frontend 연동

### Backend (FastAPI, `:8000`)

- 인증(JWT) · 프로필 · 분석 세션 · 이미지 업로드 · 추론 호출 · 추천 · 리포트
- 이미지 검증 파이프라인 — 확장자 / MIME / 크기 / Pillow 유효성 확인 후 저장
- 실패 시 `uploaded_images.failure_reason`과 `analysis_sessions.error_message`에 사유를 남겨 **어디서 끊겼는지 DB만 봐도 알 수 있게** 처리
- MySQL + SQLAlchemy + Alembic (마이그레이션 `0001` ~ `0008`)
- 추천은 `(부위, 이슈, severity)` 규칙 매칭 후 프로필의 민감성·알러지 성분으로 필터링

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/analysis/sessions/{id}/images` | 업로드 → 검증 → 추론 → 결과·추천 저장 |
| `GET` | `/analysis/sessions/{id}/report` | 전체 요약 + 부위별 리포트 |
| `GET` | `/analysis/reports/latest` | 로그인 사용자 기준 최신 완료 리포트 |
| `GET` | `/analysis/sessions/{id}/metrics` | 세션 상세 측정값 |
| `GET` | `/analysis/metrics/trends` | 지표별 시계열 추이 |
| `GET` | `/recommendations/sessions/{id}` | 저장된 추천 결과 |

### Frontend (Streamlit, `:8501`)

- 촬영/업로드 → 분석 → 리포트 → 추천 흐름, `app.py`에서 직접 라우팅
- 수치 표시 정책 — `measured_value`가 있으면 "측정값", 없으면 `predicted_value`를 "예측값"으로 표시하고, 둘 다 없으면 등급만 표시
- `session_id` · `image_id` · `raw_part_name` · `model_name` 같은 내부 값은 화면에 노출하지 않음

---

## 9. 기술 스택

| 분류 | 기술 |
|---|---|
| ML | Python 3.12, PyTorch, torchvision (ResNet-50), scikit-learn, pandas, NumPy, Pillow |
| Backend | FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2, python-jose(JWT), bcrypt, httpx |
| Database | MySQL 8 (utf8mb4) |
| Frontend | Streamlit, requests, CSS |
| 데이터 | AI-Hub 한국인 피부상태 측정 데이터 |
| 도구 | uv, Jupyter, pytest, Git/GitHub |

---

## 10. 프로젝트 구조 (담당 범위)

```text
notebooks/jh/                      # 학습 파이프라인
├── crop.py                        #   부위별 bbox 크롭 · 각도 필터 · 품질 필터
├── rebalance_split.py             #   ID 그룹 · 희소 클래스 우선 분할
├── dataset.py                     #   CSV 기반 Dataset · 증강
├── model.py                       #   ResNet-50 + 분류 head (freeze 전환 지원)
├── train.py                       #   focal loss · 샘플러 · macro-F1 조기 종료
├── status.py                      #   체크포인트/히스토리에서 진행 상황 조회
└── test/                          #   크롭 · 파이프라인 스모크 테스트

docs/
├── labeling_codes_guide.md        # AI-Hub 라벨 코드값 해설
└── cheek/                         # 크롭 규격 · 학습 전제 · 검증 기준 · 현재 설정
configs/                           # 학습 실험 설정 JSON

backend/
├── app/
│   ├── routers/                   # auth · users · analysis · images · recommendations · dev
│   ├── services/
│   │   ├── inference_service.py   #   mock / remote / multivalue 추론 호출
│   │   ├── multivalue_parser.py   #   응답 → 4개 테이블 분해
│   │   ├── image_service.py       #   업로드 검증 · 모드 분기
│   │   ├── recommendation_service.py
│   │   └── report_service.py
│   ├── schemas/                   # 요청/응답 + AI 응답 계약
│   ├── models/                    # ORM
│   └── core/                      # config · security · exceptions
├── alembic/versions/              # 0001 ~ 0008
├── docs/                          # 추론 계약 · DB 설계 · E2E 검증 결과
├── scripts/dummy_ai_server.py     # 계약 검증용 목 AI 서버
└── tests/

frontend/                          # Streamlit — views · services · components · styles · docs
```

---

## 11. 실행 방법

### 학습 파이프라인

```bash
# 1) 크롭 — AI-Hub 원본에서 부위 영역을 잘라내고 메타데이터 CSV 생성
python notebooks/jh/crop.py --facepart 5 6

# 2) 분할 — 사람(id) 단위로 train/val 재구성
python notebooks/jh/rebalance_split.py

# 3) 학습 — 실험 조건은 JSON으로 분리
python notebooks/jh/train.py --config configs/pore_run11.json

# 4) 진행 상황 확인
python notebooks/jh/status.py --target pore

# 파이프라인 스모크 테스트
python notebooks/jh/test/test_cheek_pipeline.py
```

### 서비스

**Python 3.12+**, **MySQL 8**이 필요합니다. `.env`는 저장소에 포함되지 않습니다.

```bash
# Backend
cd backend
pip install -r requirements.txt
# .env — app/core/config.py 참고
#   DB_* / SECRET_KEY / AI_INFERENCE_MODE / AI_INFERENCE_URL / AI_MULTIVALUE_INFERENCE_URL
alembic upgrade head
python -m uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py

# AI 서버 없이 서비스만 개발할 때
AI_INFERENCE_MODE=mock
# 또는 계약 검증용 목 서버
python -m uvicorn scripts.dummy_ai_server:app --reload --port 9000
```

---

## 12. 트러블슈팅

### 12.1 정확도 0.673, macro-F1 0.303 — 불균형에 속을 뻔했다

**문제**
초기 학습에서 accuracy가 0.673으로 나쁘지 않아 보였습니다. 그런데 macro-F1은 0.303이었습니다. 다수 등급만 맞히고 희소 등급은 거의 예측하지 못하는 모델이었습니다.

**해결**

- 기준 지표를 accuracy에서 **macro-F1으로 교체**하고, 조기 종료도 macro-F1 기준으로 변경
- `FocalLoss`(γ 조정)로 쉬운 다수 클래스의 손실 기여를 낮춤
- `WeightedRandomSampler`로 배치 단위 클래스 균형 확보
- 클래스별 분포를 학습 시작 시 로그로 출력해 **매 실험마다 눈으로 확인**

**결과**
지표가 실제 성능을 반영하게 됐고, 이후 실험의 개선/악화를 신뢰할 수 있게 됐습니다.

---

### 12.2 균형 보정을 했더니 성능이 오히려 떨어졌다

**문제**
5등급을 3등급으로 병합해 macro-F1 0.665를 얻은 뒤, 여기에 데이터 균형 보정을 추가로 적용했습니다. 더 좋아질 것이라 예상했지만 결과는 **0.665 → 0.644, accuracy 0.728 → 0.703**으로 하락했습니다. (수치는 §6의 출처 기준)

**해결**

- 등급을 이미 3개로 합친 시점에서 분포가 상당히 평탄해졌고, 그 위에 균형 보정을 또 얹으면 **다수 클래스의 정상 신호까지 깎인다**고 판단
- 보정을 되돌리고 3등급 ordinal 설정을 기준선으로 확정
- 실패한 실험을 지우지 않고 조건·수치와 함께 기록

**결과**
"불균형 대응은 무조건 좋다"는 가정을 버리고, **문제 정의를 바꾼 뒤에는 대응 강도도 다시 재야 한다**는 기준을 얻었습니다.

---

### 12.3 같은 사람의 사진이 train과 val에 동시에 들어갈 뻔했다

**문제**
한 사람당 여러 각도·장비로 촬영된 데이터셋입니다. 이미지 단위로 무작위 분할하면 같은 얼굴이 학습과 검증에 모두 들어가 검증 점수가 부풀려집니다.

**해결**

- 사람(`id`) 단위로 그룹을 묶어 분할하도록 `rebalance_split.py` 작성
- 단순 그룹 분할만 하면 희소 등급이 한쪽에 몰려 검증이 불가능해지므로, **희소 클래스를 먼저 덮도록 그룹을 선택**하는 방식으로 구성
- 메타데이터 CSV에 `id`를 필수 컬럼으로 두어 이후에도 누수 여부를 검증 가능하게 함

**결과**
검증 점수가 실제 일반화 성능을 반영하게 됐고, 6.3의 실험 비교가 의미를 갖게 됐습니다.

---

### 12.4 모델 출력을 서비스로 옮길 때 등급과 측정값이 섞일 뻔했다

**문제**
모델은 회귀 예측값을, 장비/AI-Hub JSON은 측정 수치를 줍니다. 둘 다 실수라 한 컬럼에 담기 쉬운데, 그러면 나중에 **이 값이 예측인지 측정인지 알 수 없습니다.** 단위와 범위도 다릅니다.

**해결**

- `grade_value`(등급) · `predicted_value`(모델 예측) · `measured_value`(장비/JSON 측정)를 **별도 컬럼**으로 분리
- 두 회귀 필드는 `Optional`로 두어 이 필드를 모르는 이전 버전 AI 서버와도 호환
- 프론트는 `measured_value` → `predicted_value` → `grade_value` 순으로 **라벨을 다르게 붙여** 표시
- 부위 미검출 시 값도 버리지 않고 `bbox_source`로 신뢰도를 구분 저장

**결과**
학습 단계에서 정리한 분류/회귀 구분(§5.3)이 DB 스키마와 화면 표시까지 일관되게 이어졌습니다.

---

## 13. 회고

이 프로젝트에서 가장 크게 배운 것은 **성능이 안 나올 때 어디를 먼저 봐야 하는가**였습니다. 처음에는 하이퍼파라미터와 증강을 바꾸는 데 시간을 썼는데, 실제로 성능을 끌어올린 것은 **라벨 정의를 5등급에서 3등급으로 바꾼 것**이었습니다. macro-F1이 0.303에서 0.665로 올라간 개선은 모델이 아니라 문제 정의에서 나왔습니다. 모델을 바꾸기 전에 데이터와 라벨을 의심하는 순서를 몸으로 익혔습니다.

두 번째는 **지표를 잘못 고르면 실험 전체가 무의미해진다**는 것이었습니다. accuracy 0.673 / macro-F1 0.303은 같은 모델의 두 얼굴입니다. 불균형 데이터에서 accuracy만 보고 있었다면 "잘 되고 있다"고 착각한 채 계속 갔을 겁니다. 기준 지표를 macro-F1으로 바꾼 뒤에야 이후 실험들의 개선과 악화를 신뢰할 수 있었습니다.

세 번째는 **실패한 실험도 자산이라는 것**입니다. 균형 보정을 추가했다가 성능이 떨어져 되돌린 실험(0.665 → 0.644)은 결과만 보면 버릴 기록이지만, "등급 병합 이후에는 추가 균형 보정이 역효과"라는 판단 근거로 남았습니다. 그리고 이런 진단이 쌓여 **ResNet-50의 한계를 백본 문제로 규정하고 DINOv3 전환을 결정하는 근거**가 됐습니다.

마지막은 **모델과 서비스 사이가 생각보다 넓다는 것**이었습니다. 학습이 끝나도 출력을 어떤 구조로 넘길지, 등급과 측정값을 어떻게 구분해 저장할지, 검출 실패를 어떻게 표시할지를 정하지 않으면 서비스에 붙지 않습니다. `chin_moisture`처럼 **학습 라벨에 없어 항상 0이던 지표를 E2E 검증에서 잡아내 제거한 경험**이 특히 그랬습니다. 값이 비어 있는 것과 값이 0인 것은 다르고, 그 차이를 스키마로 표현하지 않으면 사용자에게 거짓 정보가 나갑니다.

아쉬운 점은 **제가 학습한 모델을 최종 서비스에 연결하지 못한 것**입니다. 운영 추론은 팀이 전환한 DINOv3 모델이 담당하고, 제 파이프라인은 실험 설정과 진단 기록으로 남았습니다. run별 성능 지표를 저장소에 남기지 못한 것도 아쉽습니다 — 설정은 `configs/`로 분리해 뒀지만 결과는 로컬 체크포인트에만 있습니다.

---

## 14. 고도화 방향

### Phase 1. 실험 기록 자동화

현재 run별 성능은 로컬 체크포인트에만 남습니다. `status.py`가 읽는 history 로그를 실험 결과 CSV로 누적하고 설정 해시와 함께 저장하면, 조건과 결과를 나중에 그대로 대조할 수 있습니다.

### Phase 2. 볼 부위 모델의 서비스 연결

학습한 모공·색소침착 분류 모델을 추론 서버 라벨 레지스트리에 등록합니다. 추론 계약(§7.1)이 고정돼 있으므로 **백엔드 수정 없이** 연결 가능합니다.

### Phase 3. 지표별 임계값 확정

`predicted_value` · `measured_value`의 값 범위와 방향성(값이 클수록 좋은지 나쁜지)이 확정되면, 현재 등급 기반인 severity를 연속값 기준으로 더 정교하게 계산할 수 있습니다. 현재는 분포를 산출할 세션 수가 부족해 후보값만 문서에 기록해 두었습니다.

### Phase 4. 부위 검출률 개선

눈가처럼 검출 실패가 잦은 부위는 전체 이미지 fallback으로 추론돼 신뢰도가 떨어집니다. 검출 실패 케이스를 `bbox_source` 기준으로 모아 원인을 분석하고, 촬영 가이드(정면·조명)와 검출 모델 양쪽에서 개선합니다.
