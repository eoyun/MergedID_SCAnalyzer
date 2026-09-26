# gunheavy 파이프라인 — 코드 라인바이라인 리뷰

**목적**: `particle_gun_heavy`를 signal, W/Z를 background로 하는 cross-attention
**분류 + 연속 질량 회귀(mass regression)** 학습을 위한 데이터 준비 파이프라인의 전체
구조를 코드 수준에서 설명한다. 논문의 *Dataset / Sample preparation / Event selection
and weighting* 절을 쓸 때 이 문서를 근거로 삼을 수 있도록 물리적 근거와 구현을 함께 적었다.

작성일 2026-09-26. 대상 파일:
- `build_category_weights_gun.py` (248줄) — 이벤트 선택 + 가중치 계산 (stage 1)
- `build_compact_dataset_gun.py` (187줄) — 모델 입력용 compact 생성 (stage 2)
- `condor/run_gunheavy_weights.sh`, `condor/run_gunheavy_compact.sh` (+ `.sub`/`.txt`) — 배치 실행

---

## 0. 큰 그림 (파이프라인 데이터 흐름)

```
원본 h5 (EOS: data/MiniAOD/{W,Z,particle_gun_heavy}/*.h5)
        │   각 파일 = 여러 이벤트, 이벤트당 여러 supercluster(SC=병합전자 후보)
        │
        ▼  [stage 1] build_category_weights_gun.py   (pt50/100/200 각각)
category_weights.h5   ── per-object 가중치 + per-object pT(SC ET) + object→event map
        │              (EOS: weights/categories_gunheavy_pt{50,100,200}/gunheavy_all/)
        │
        ▼  [stage 2] build_compact_dataset_gun.py    (× eb/ee)
compact h5            ── train/val/test 분할된 이미지·트랙·label·weight·pt·a_mass
        │              (EOS: compact_gunheavy_pt{50,100,200}/gunheavy_{eb,ee}.h5)
        │
        ▼  [stage 3, 예정] train_multitask_cross (연속 a_mass)
학습된 모델 → 검증 (기존 signal/ H→AA + W/Z)
```

**설계 핵심 3가지** (논문 방법론에서 강조할 부분):
1. **Signal = particle gun** (A→ee), 연속 질량 0.1–50 GeV. 이벤트당 A가 2개지만 **질량이 정확히
   동일**하므로 전자 하나의 회귀 타깃 = 그 이벤트의 A 질량 (전자↔A 매칭 불필요).
2. **pT 변수 = per-object reco supercluster 횡에너지** `SC_energyT_sum`(EB)/`EE_seed_energyT_sum`(EE).
   기존 파이프라인의 gen A_pT와 달리 **실데이터에도 존재**하고, 크롭 중심 전자의 물리량이다.
3. **pT-bin 균등 가중치**: 각 클래스(signal/background)의 총 가중치를 1로 맞추고, 비어있지 않은
   pT bin마다 동일한 가중치 합을 배분 → 모델이 "고 pT = signal" 같은 자명한 지름길을 못 쓰게 함.

---

## 1. 물리 배경 (코드를 읽기 전 알아야 할 것)

원본 h5의 한 "오브젝트"는 **seed 전자를 중심으로 crop한 32×32 칼로리미터 이미지 1개**
(= 1개의 supercluster, 병합전자 후보)이다. 검출기별로 분리:
- **EB**(barrel): `SC_energy`(32×32), 트랙 `EB_track_pt_{PF,GSF,Lost}`, 이벤트맵 `EB_ele_event_idx`.
- **EE**(endcap): `EE_seed_energy`(32×32), preshower `ES_seed_plane1/2`, `EE_ele_event_idx`.

**`A_*` 배열의 정체 (중요)**: 시료마다 "부모 공명입자" 컨테이너로 채워진다.
- W 파일: `A_mass`≈80 (=W 보손), 이벤트당 1개 → `A_pT`=W 보손 pT
- Z 파일: `A_mass`≈91 (=Z 보손), 이벤트당 1개
- particle_gun_heavy: `A_mass`=0.1–50 (연속), 이벤트당 **2개, 질량 동일**, `A_pT`≈1300 GeV(gun 발사 운동량, 비물리적)

→ 그래서 gen `A_pT`를 pT로 쓰면 gun signal은 어떤 컷도 100% 통과하고, 무엇보다 **실데이터엔 gen
A_pT가 없다**. 이 때문에 **per-object reco SC ET**로 전환한 것 (섹션 3.1 참조).

---

## 2. `build_category_weights_gun.py` — 이벤트 선택 + 가중치 (stage 1)

### 2.1 상수 (L40–47)
```python
SIGNAL_DIR = "particle_gun_heavy"
SPLITS     = ("W", "Z", SIGNAL_DIR)          # 스캔할 하위 디렉토리
SC_ET_KEYS = {"eb": "SC_energyT_sum", "ee": "EE_seed_energyT_sum"}   # ← pT 변수
WEIGHT_KEYS= {"eb": "EB_weight_split", "ee": "EE_weight_split"}      # 출력 가중치 키
OBJ_PT_KEYS= {"eb": "EB_obj_pt", "ee": "EE_obj_pt"}                  # 출력 per-object pT
N_BINS     = len(PT_BIN_EDGES_WITH_OVERFLOW) - 1                     # pT bin 개수(22)
```
- `WEIGHT_KEYS`/이벤트맵 키 이름을 **기존 `build_category_weights.py`와 동일하게** 맞춰서,
  다운스트림(`build_object_manifest` 등)이 이 파일을 수정 없이 읽는다.
- `OBJ_PT_KEYS`는 **새로 추가한** per-object SC ET 데이터셋(stage 2가 진짜 per-object pT로 사용).

### 2.2 `_sample_id` (L50–52), `_read_meta` (L54–70)
```python
def _sample_id(split):
    return "background" if split in ("W", "Z") else "signal"
```
- 클래스 라벨의 원천. W/Z→`"background"`, gun→`"signal"`. (기존 코드는 signal을 이산 질량점으로
  세분했지만, gun은 단일 signal로 취급.)

`_read_meta`는 **worker 함수**(멀티프로세싱). 파일 하나에서 무거운 이미지는 건드리지 않고
필요한 작은 배열만 읽는다:
```python
ne = int(r["eventId"].shape[0])                      # 이벤트 수
for det in DETECTORS:
    ei = r[DET_EVENT_IDX_KEYS[det]][:].astype(np.int64)  # object→event
    pt = r[SC_ET_KEYS[det]][:].astype(np.float32)        # per-object SC ET (=pT)
```
- 반환 dict에 `split/path/stem/sample_id/n_events` + det별 `(event_idx, sc_et)`.
- 실패 시 `SkippableFileError`를 잡아 `ok=False`로 표시(EOS 불량 replica 대비, `robust_h5_open` 사용).

### 2.3 `scan_parallel` (L72–91)
- `SPLITS` 3개 디렉토리의 모든 `*.h5`를 모아 `_read_meta`를 **spawn Pool로 병렬** 실행
  (EOS FUSE 지연을 겹쳐 숨김).
- `metas.sort(key=(split, stem))`로 **결정론적 순서** 확보 (재현성 → 배경 서브샘플의 시드 일관성).

### 2.4 `select_background_masks` (L94–120) — 배경 서브샘플링
배경(W/Z)은 개수가 signal보다 훨씬 많으므로, 검출기별로 무작위 상한(`max_objects`,
기본 100만)을 둔다.
```python
elig = pt >= pt_min                # 컷: per-object SC ET가 pt_min 이상인 것만 후보
...
chosen = np.sort(rng.choice(n_elig, size=max_objects, replace=False))
```
- **후보 풀 = SC ET ≥ pt_min인 배경 오브젝트 전체** → 그 안에서 무작위로 `max_objects`개 선택.
- 파일 경계를 넘는 전역 인덱스를 `searchsorted`로 파일별 마스크로 되돌린다(L112–119).
- signal(gun)은 여기서 서브샘플 안 함 → 전부 유지(섹션 2.5).

### 2.5 `selected_mask` (L123–130) — 오브젝트 선택 규칙
```python
if m["sample_id"] == "background":
    mask = bg_masks[det].get(m["stem"], ...)   # 미리 뽑은 서브샘플 마스크
else:  # signal(gun): SC ET 컷을 통과한 전자 전부 유지
    mask = pt >= pt_min
```
- **한 줄 요약**: 배경=서브샘플된 것만, signal=컷 통과분 전부. 컷 변수는 둘 다 per-object SC ET.

### 2.6 `make_lookup` (L133–150) — pT-bin 균등 가중치 (**논문 핵심**)
```python
targets = {"background": 1.0}
for s in signal_ids: targets[s] = 1.0 / len(signal_ids)   # signal 총합도 1
...
nonempty = counts > 0; n = nonempty.sum()
per_bin = targets[s] / n                 # 클래스 목표(=1)를 비-빈 bin에 균등 배분
w[nonempty] = per_bin / counts[nonempty] # 그 bin 내 오브젝트끼리 다시 균등 분배
```
수식으로:
$$ w_{\text{obj}} = \frac{T_c}{N^{\text{nonempty}}_c \cdot n_{c,b}}, \quad T_c=1 $$
여기서 $c$=클래스, $b$=오브젝트가 속한 pT bin, $N^{\text{nonempty}}_c$=클래스의 비어있지
않은 bin 개수, $n_{c,b}$=(클래스 $c$, bin $b$)의 오브젝트 수.
- **효과 1**: 클래스별 총 가중치 = 1 (signal/background 균형).
- **효과 2**: bin마다 가중치 합이 동일 → 학습이 pT 분포 자체(고 pT=gun, 저 pT=W/Z)를
  지름길로 삼지 못하게 함. 검증에서 확인: 모든 비-빈 bin의 합 = 1/N_nonempty, 총합=1.000000.

### 2.7 `build` (L153–228) — 전체 오케스트레이션
1. `scan_parallel`로 메타 로드 (L167).
2. 검출기별 배경 마스크 계산 (L172–176).
3. **count pass** (L179–190): 선택된 오브젝트를 per-object SC ET로 binning하여
   `(sample, det, bin)`별 개수 집계 → `make_lookup`으로 가중치 룩업 생성.
4. **write pass** (L193–227): 파일별 그룹에 기록.
   - `event_pt`(L207–213): **per-event max SC ET**. 이는 다운스트림 `build_object_manifest`가
     반드시 읽는 필드라 호환용으로 채우는 **placeholder**이며, 실제 per-object pT는 stage 2가
     `EB/EE_obj_pt`로 덮어쓴다. (주석에 명시.)
   - det별로 `EB_weight_split`(가중치), `EB_ele_event_idx`(object→event),
     `EB_obj_pt`(per-object SC ET) 저장 (L214–225).
   - 그룹 attrs: `input_file/split/process_name/sample_id/n_events` — `parse_file_entries` 계약.
   - 파일 attrs: `pt_variable="reco_sc_energyT_sum"`, `pt_min`, `background_selected_objects_*` 등
     (재현·문서화용).

### 2.8 CLI (L230–245)
- `--pt-min`(필수), `--input-dir`(기본 `data/MiniAOD`), `--output-path`,
  `--background-max-objects`(기본 100만, -1=전체), `--seed`(1234), `--workers`(16).

---

## 3. `build_compact_dataset_gun.py` — 모델 입력 compact (stage 2)

기존 `build_compact_dataset.py`의 뼈대(분할·매니페스트·이미지/트랙 읽기)를 **재사용**하고,
회귀에 필요한 두 컬럼(`pt`=per-object SC ET, `a_mass`=연속 질량)만 새로 채운다.

### 3.1 임포트 (L28–37)
```python
from build_compact_dataset import (SPLIT_CODE, VLEN_*, calo_key, raw_source_keys,
                                    build_combined_manifest)
from build_pgun_compact import derive_event_mass        # per-event A 질량
```
- `build_combined_manifest`가 gun weight 파일을 읽어 train/val/test **분할 + object manifest**
  (label/weight/file_idx/object_idx/event_idx/sample_code/split)를 만들어 준다 → 재사용.
- `derive_event_mass(n_events, A_mass, A_event_idx)` = 이벤트별 A 질량(=평균, gun은 값이 같아 동일).

### 3.2 `_read_file_objects_gun` (L40–73) — worker
기존 worker와 동일하게 calo 이미지·ES·트랙을 읽되(L48–61), **추가로**:
```python
et = r[SC_ET_KEYS[det]][:].astype(np.float32)
out["pt"] = et[obj_idx]                       # per-object 진짜 pT (SC ET)

if is_signal:
    emass = derive_event_mass(ne, A_mass, A_event_idx)   # 이벤트별 A 질량
    dev = r[DET_EVENT_IDX_KEYS[det]][:]                  # object→event
    out["a_mass"] = emass[dev[obj_idx]]                  # 전자별 회귀 타깃
else:
    out["a_mass"] = np.zeros(...)                        # 배경은 0 (loss에서 마스킹)
```
- **왜 이렇게?** 섹션 1에서 gun 이벤트의 두 A 질량이 동일함을 확인 → 전자별 타깃 = 이벤트 A 질량.
  전자↔A 개별 매칭이 필요 없고, `A_recoIdx`(비어있음)에 의존하지 않는다.
- 배경 `a_mass=0`은 회귀 손실에서 `is_sig=(label==1)` 마스크로 제외되므로 값 자체는 무의미(안전).

### 3.3 `build` (L76–163)
1. `build_combined_manifest`로 매니페스트 확보 (L85). `man["pt"]`(gen broadcast)는 **일부러 안 씀**.
2. 출력 h5 attrs (L94–102): `detector/track_types/split fracs` + `pt_variable`,
   `sample_kind="gunheavy"`, pT bin edges.
3. 북키핑 컬럼(label/weight/file_idx/object_idx/event_idx/sample_code/split)은 매니페스트에서
   그대로 복사 (L106–110). `sample_id`/`raw_file`은 코드↔이름 매핑으로 기록 (L111–114).
4. **`pt`, `a_mass`는 빈 데이터셋으로 만들고 worker 출력으로 채움** (L116–117).
5. 이미지/ES/트랙 데이터셋 생성 (L119–129, 형상은 원본에서 추론).
6. 오브젝트를 파일별로 묶어 task 생성 → `is_signal` 플래그 포함 (L131–143).
7. `_write`(L145–155): worker 결과를 행 정렬해 각 데이터셋에 기록(`pt_ds`, `amass_ds` 포함).
8. spawn Pool로 병렬 읽기·직렬 쓰기 (L157–161).

### 3.4 CLI (L165–182)
- `--weight-h5`(stage1 산출물), `--detector eb|ee`, `--track-types "Lost,PF,GSF"`,
  `--output-path`, `--train-frac 0.8`, `--val-frac 0.1`, `--seed 42`, `--workers 16`.

### 3.5 검증 결과 (미니 데이터)
| 항목 | EB | EE |
|---|---|---|
| pt(SC ET) 컷≥50 | ✓ (sig 중앙 1267, bkg 58) | ✓ (sig 1305, bkg 58) |
| a_mass signal | 0.10–49.9 연속 | 0.11–49.7 연속 |
| a_mass background | 전부 0 | 전부 0 |
| 이미지·트랙 | 정상 | 정상(+ES) |

---

## 4. Condor 오케스트레이션

두 stage 모두 **로그아웃과 무관하게** 도는 배치 잡으로 실행한다(로그인 노드 백그라운드 금지).

### 4.1 `run_gunheavy_weights.sh` (인자: `<pt_min>`)
- LCG 뷰 source → scratch를 TMPDIR로 → `build_category_weights_gun.py` 실행(로컬에 기록)
  → `xrdcp`로 EOS 스테이지아웃. `.sub`가 `condor/gunheavy_weights.txt`(값 `50/100/200`)로 3잡 제출.
- 실행: cluster 18605831, 산출 `weights/categories_gunheavy_pt{50,100,200}/gunheavy_all/`.

### 4.2 `run_gunheavy_compact.sh` (인자: `<pt_min> <det>`)
- EOS의 weight 파일을 `xrdcp`로 로컬에 가져온 뒤 `build_compact_dataset_gun.py` 실행 →
  compact를 EOS로 스테이지아웃. `.txt`는 6줄(pt×det) → 6잡.
- 실행: cluster 18605838, 산출 `compact_gunheavy_pt{50,100,200}/gunheavy_{eb,ee}.h5`.
- `transfer_input_files`에 `build_compact_dataset_gun.py`가 임포트하는 모듈
  (`build_compact_dataset.py`, `build_pgun_compact.py`, `build_event_level_weights.py`,
  `train_resnet_image_classifier.py`, `pipeline/`)을 모두 포함.

---

## 5. 논문 작성용 요약 (Methods에 넣을 문장 재료)

- **Samples**: signal = A→ee particle gun (continuous $m_A\in[0.1,50]$ GeV); background =
  Drell–Yan/W+jets ($Z\to ee$, $W\to e\nu$). 각 오브젝트는 seed 전자를 중심으로 한 32×32
  ECAL supercluster 이미지 + 관련 트랙.
- **Object pT**: reconstruction-level supercluster transverse energy
  ($E_T^{\text{SC}}$; barrel `SC_energyT_sum`, endcap `EE_seed_energyT_sum`). 데이터에서
  재현 가능하도록 generator-level $A$ $p_T$ 대신 채택.
- **Preselection**: $E_T^{\text{SC}} \ge$ {50, 100, 200} GeV 세 시나리오.
- **Class/pT balancing**: 각 클래스 총 가중치를 1로, 비어있지 않은 $E_T$ bin마다 동일한 가중치
  합을 배분 → 태거가 $p_T$ 형태를 지름길로 학습하는 것을 방지.
- **Regression target**: 전자별 연속 $m_A$. gun 이벤트 내 두 $A$가 동일 질량임을 확인하여
  이벤트 질량을 전자 타깃으로 사용(개별 $A$–전자 매칭 불필요). 배경은 회귀 손실에서 마스킹.
- **Train/val/test**: 이벤트 단위 층화 분할 80/10/10.

## 6. 관련 파일
- `docs/dataset-structure.md` — 원본 h5 스키마와 기존(gen A_pT) pT 정의.
- `build_category_weights.py` / `build_compact_dataset.py` — 기존(18-way) 대응 스크립트(수정 안 함).
- 메모리: `project-gunheavy-pipeline` (상태·클러스터 ID), `project-18way-training-pipeline`.
