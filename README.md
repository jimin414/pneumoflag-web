# PneumoFlag: Reliable and Explainable Pediatric Pneumonia Diagnosis

> Grad-CAM 최적화 및 TTA 불확실성 기반의 Selective Prediction (Human-in-the-Loop) 흉부 X-ray 진단 보조 시스템

---

## 1. Project Overview

본 프로젝트는 ResNet18 기반의 소아 흉부 X-ray 폐렴 이진 분류 모델을 구축하고, 의료 AI의 실질적 도입을 위해 필수적인 설명 가능성(Explainability)과 진단 신뢰성(Reliability)을 공학적으로 검증하고 고도화하는 것을 목적으로 합니다.

모델이 모든 데이터를 강제로 분류하도록 강제하는 대신, TTA(Test-Time Augmentation) 기반 불확실성 지표($\sigma$)를 산출하여 판단이 모호한 고난도 샘플을 '판독 보류(Reject)' 처리하고 전문의에게 재검토를 요청하는 **Selective Prediction (Human-in-the-loop)** 파이프라인을 구축하였습니다.

---

## 2. Key Achievements

* **Selective Prediction 정확도 향상**: 불확실성 상위 10% 샘플 Reject 시 최종 진단 정확도 **99.79%** 달성 (상위 40% 배제 시 정확도 100.0% 달성).
* **정량적 불확실성($\sigma$) 지표 구축**: 동적 TTA 로직 개선을 통해 오답 샘플에서 정답 대비 약 **14배 높은 불확실성**($\sigma \approx 0.140$) 감지.
* **도메인 특화 Grad-CAM 노이즈 제거**: 가우시안 마스킹 및 적응형 임계값을 도입하여 시각화 외곽 노이즈를 **90% 이상 제거**하고 병변 집중도 개선.
* **학습 파이프라인 및 시스템 최적화**: AMP 및 DataLoader 병렬화 설정을 통해 학습 속도 **4~5배 가속** (0.40 it/s -> 1.8~2.0 it/s) 및 GPU 활용률 85~95% 확보.
* **실험 재현성 및 무결성 보장**: Patient-wise Split 기준 고정 및 Multi-seed(42, 84) 검증을 통해 난수 의존성을 배제한 일반화 성능 입증.

---

## 3. Directory Structure

```text
PneumoFlag_Project_2026/
├── 00_admin/           # 실험 기준 파일 (split.json 등)
├── 01_data/            # 원본 및 전처리 데이터 보관
├── 02_notebooks/       # 기능별 Notebook (EDA, Train, Eval)
├── 03_runs/            # 실험 결과 (Weight, TensorBoard Log) 자동 저장
└── 04_reports/         # 시각화 결과 및 최종 리포트
```

---
## 4. Pipeline & Methodology

### 4.1 Data Integrity & Custom Pipeline
* **Patient-wise Split**: 동일 환자 데이터 중복으로 인한 데이터 누수를 방지하기 위해 `GroupShuffleSplit`을 적용하여 환자 ID 기준으로 Train(70%), Val(15%), Test(15%)를 분리.
* **DataLoader 최적화**: `num_workers=4`, `pin_memory=True`, `prefetch_factor=2`, `persistent_workers=True`를 적용해 CPU-GPU 간 데이터 병목(Data Starvation) 해소.

### 4.2 Class Imbalance Correction & Mixed Precision
* **Loss Correction**: 소수 클래스(Pneumonia)의 재현율(Recall)을 확보하기 위해 양성 가중치 손실 함수 적용:
  $$\text{pos\_weight} = \frac{n_{\text{negative}}}{n_{\text{positive}}}$$
* **AMP (Automatic Mixed Precision)**: FP16/FP32 혼합 연산 및 `GradScaler`를 적용하여 GPU 메모리 버퍼를 최적화하고 속도 가속.

---

## 5. Experimental Results

### 5.1 Ablation Study (Data Augmentation & Weighting)
현실적인 임상 환경(기울기 $\pm7^\circ$, 노출 변화 Brightness $\pm10\%$)을 모사한 모델 견고성(Robustness) 분석 결과입니다.

| Model | F1-Score | PR-AUC | Recall | Best Epoch |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 0.9976 | 0.9999 | 0.9953 | 3 |
| Aug + No Weighted | 0.9936 | 0.9999 | 0.9874 | 3 |
| **Aug + Weighted (Final)** | **0.9936** | **0.9999** | **0.9874** | **8 (Stabilized)** |

### 5.2 System Optimization Benchmark

| Metric | Baseline | Optimized (AMP + Pipeline) | Improvement |
| :--- | :---: | :---: | :---: |
| 학습 속도 (Training Speed) | 약 0.40 it/s (2.5s/it) | **약 1.8 ~ 2.0 it/s** | 약 4~5배 가속 |
| GPU 활용률 (Utilization) | 30~40% (불안정) | **85~95% (안정)** | 연산 효율 극대화 |
| 추론 속도 (Inference Latency) | - | **0.807 ms / image** | ResNet18 (Batch 32, GPU) |

---

## 6. Explainability & Uncertainty Analysis

### 6.1 Domain-Tailored Grad-CAM
기본 Grad-CAM의 외곽 활성화 문제를 해결하기 위해 흉부 X-ray 특성을 반영한 3단계 최적화를 적용했습니다.
1. **Gaussian Lung Focus**: 폐 영역 중심부 집중 마스크($\sigma = h/4$)를 적용하여 외곽(어깨, 표식) 노이즈 90% 이상 제거.
2. **Adaptive Thresholding**: 상위 40% 미만($0.4 \times \text{max}$) 에너지를 제거하여 핵심 병변 구역만 선명하게 추출.
3. **High-Resolution Blend**: OpenCV JET 컬러맵 및 가중 합성($\alpha=0.4$)을 적용해 갈비뼈 및 폐 실질의 미세 텍스처 보존.

### 6.2 Uncertainty-based Selective Prediction (TTA)
추론 시 8회 실시간 랜덤 변형(Dynamic TTA)을 수행하여 예측의 변동성(표준편차 $\sigma$)을 산출하였습니다.

| Classification Target | Mean Uncertainty ($\sigma$) | Analysis |
| :--- | :---: | :--- |
| **정답 샘플 (Correct)** | `0.010` | 높은 확신도 및 일관성 유지 |
| **오답 샘플 (Incorrect)** | **`0.140`** | **정답 대비 약 14배 높은 변동성 포착** |

* **Coverage-Accuracy Trade-off**:
  * **Coverage 1.00 (전체 샘플)**: Accuracy 98.9%
  * **Coverage 0.95 (상위 5% Reject)**: Accuracy **99.86%**
  * **Coverage 0.60 (상위 40% Reject)**: Accuracy **100.0%** (완벽 신뢰 구간 달성)


---

## 7. Future Work

* **Lung Segmentation 연계**: 소아 외 성인 데이터셋 확장을 위한 폐 영역 자동 분할 전처리 파이프라인 도입.
* **Loss Function 고도화**: 클래스 불균형 해결을 위한 Focal Loss 적용 및 가중치 파라미터 최적화.
* **MLLM 기반 소견서 생성**: Reject 판독 대상 샘플에 대해 임상 텍스트 리포트를 생성하는 멀티모달 보조 시스템 확장.
