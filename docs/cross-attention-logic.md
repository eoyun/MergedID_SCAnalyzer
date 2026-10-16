# Cross-Attention 로직 (현재 사용 기준)

merged-electron ID에서 쓰는 **cross-attention fusion** 모델의 로직을 현재 코드 기준으로 정리한다.
(`train_fusion_cross_attention.py`의 `CrossAttentionFusion` + `multitask_common.py`의 `MultiTaskModel`.
학습은 `train_multitask_cross_gun.py`, 평가는 `eval_gunheavy.py`.)

핵심 아이디어: **칼로리미터 이미지**와 **트랙 point cloud**를 각각 토큰 시퀀스로 만들고,
**양방향 cross-attention**으로 두 모달리티가 서로 정보를 교환한 뒤, 풀링해서 분류(merged electron 여부)
와 연속 질량 회귀를 동시에 수행한다.

---

## 0. 데이터 흐름 한눈에

```
 calo image (raw crop 32x32 또는 48x48)        track point cloud  (N_t x F) + mask
        │                                                │
  [tile_to_square] 각 채널을 공통 정사각형으로 타일        trk_in: Linear→LN→GELU→Linear  (F→d)
   512(32-data) 또는 768(48-data)                         + null 토큰 선두 삽입
        │  (calo, ES 채널 스택 → [C, S, S])               TransformerEncoder (self-attn, track_depth)
  ResNet backbone (avgpool/fc 제거)                        │  + trk_type
   → feature map [B, C_f, S/32, S/32]                      │
  img_proj (1x1 conv → d)                                  │
   → flatten → [B, Ni, d]   (+ img_type)                   │
        │                                                  │
        └──────────────── 양방향 cross-attention × depth ──┘
             img2trk: 이미지 토큰이 트랙에 attend
             trk2img: 트랙 토큰이 이미지에 attend  (동시 갱신)
                        │
          모달리티별 masked-mean 풀링 → concat [B, 2d] → LayerNorm(out_norm)
                        │  feat [B, 2d]
          ┌─────────────┴─────────────┐
     cls_head (Linear)           reg_head (MLP)
     → merged-electron logit     → 예측 log(m_A)
```

핵심 하이퍼파라미터(기본): `d_model=256`, `feat=2·d=512`, `num_heads=4`, cross `depth=2`,
`track_depth=2`, `mlp_ratio=4`, `dropout=0.1`, backbone `resnet18`. fp32 + TF32.

---

## 1. 입력 토큰화

### 1-1. 이미지 토큰 (Ni개)
- raw crop(32×32 또는 48×48 calo, + EE는 ES planes)을 **`tile_to_square`로 공통 정사각형 S로 타일**
  (nearest-repeat 확대). `S = IMAGE_TILE_SIZE` (32-data는 512, 48-data는 768; S는 모든 채널 변의 공배수).
  → 채널 스택 `[C, S, S]`.
- **ResNet**(torchvision, avgpool+fc 제거)이 32배 다운샘플 → feature map `[B, C_f, S/32, S/32]`.
- `img_proj = Conv2d(C_f, d, 1)` (1×1) → `flatten(2).transpose` → **`[B, Ni, d]`**, `Ni = (S/32)²`.
- 학습 가능한 `img_type` 임베딩을 더해 모달리티 태그.

| 입력 타일 S | feature map | **Ni (이미지 토큰 수)** |
|---|---|---|
| 512 (32-data) | 16×16 | **256** |
| 768 (48-data, cropped) | 24×24 | **576** |

> 토큰 수가 동적이라 코드 변경 없이 S가 바뀌어도 동작한다 (positional embedding을 쓰지 않기 때문).

### 1-2. 트랙 토큰 (1 + N_t개)
- point cloud `[B, N_t, F]` (F = per-point feature dim) → `trk_in`(2층 MLP)로 `[B, N_t, d]`.
- **널(null) 토큰**을 맨 앞에 붙임: `[B, 1+N_t, d]`. 트랙이 0개인 오브젝트가 **all-masked key가 되어
  softmax가 NaN 나는 것을 방지** (merged electron은 트랙이 없을 수 있음). 널은 padding mask에서 항상 유효.
- `TransformerEncoder`(self-attention, `track_depth`층, pre-LN, padding mask) → 트랙끼리 문맥 교환.
- `trk_type` 임베딩 더함.

---

## 2. 양방향 cross-attention 블록

pre-LN 블록 (`CrossAttentionBlock`): query가 key/value에 attend → residual → FFN → residual.

```
q' = q + MHA( LN(q), LN(kv), LN(kv), key_padding_mask )      # cross attention
q'' = q' + FFN( LN(q') )                                      # FFN = Linear(d,4d)→GELU→Linear(4d,d)
```
- `nn.MultiheadAttention(d, num_heads, batch_first=True)`.
- `key_padding_mask`로 padding 트랙을 무시.

**attention 자체의 의미**: query 토큰 i가 모든 key 토큰 j에 대한 가중합을 받음.
가중치 `softmax(q_i·k_j / √d_k)` → 두 모달리티의 어떤 부분이 서로 연관되는지 학습.

---

## 3. 양방향 융합 루프 (depth회)

```python
for a, b in zip(img2trk, trk2img):
    new_img = a(img, trk, kv_key_padding_mask=kpm)   # 이미지(query) ← 트랙(key/value)
    new_trk = b(trk, img, kv_key_padding_mask=None)  # 트랙(query) ← 이미지(key/value)
    img, trk = new_img, new_trk                      # 동시 갱신(한쪽 먼저 갱신해 오염되는 것 방지)
```
- `img2trk`: 이미지 토큰이 트랙 정보를 흡수 (트랙 key엔 padding 있음 → kpm).
- `trk2img`: 트랙 토큰이 이미지 정보를 흡수 (이미지 key엔 padding 없음 → None).
- `depth`번 반복하며 두 모달리티가 점점 더 섞인다.

---

## 4. 풀링 + 융합

```python
img_pool = img.mean(dim=1)                                    # 이미지 토큰 평균 (Ni개)
m = valid.unsqueeze(-1).float()
trk_pool = (trk * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0) # 유효 트랙 토큰만 masked-mean
feat = out_norm( cat([img_pool, trk_pool], dim=1) )          # [B, 2d]=512, LayerNorm
```
- 각 모달리티를 토큰 평균으로 벡터화 → concat → LayerNorm → 공통 표현 `feat [B, 512]`.
- `clamp_min(1.0)`: 널 토큰 덕에 분모 ≥ 1 (0 division 방지).
- multitask에서는 백본의 최종 `head`가 `nn.Identity()`로 교체되어 **백본이 `feat`를 그대로 반환**.

---

## 5. 멀티태스크 헤드

```python
cls_logit = cls_head(feat)   # Linear(512, 1)          → merged-electron 분류 (BCE)
reg_mu    = reg_head(feat)   # Linear(512,512)→GELU→Dropout→Linear(512,1) → 예측 log(m_A)
```
- 하드 파라미터 공유: 백본(cross-attn)이 공통 `feat`를 만들고 두 헤드가 분기.
- 손실: `L = BCE(cls, label; weight) + λ · MAE(reg_mu, log m_A) · [signal mask]`
  (배경은 회귀에서 마스킹, λ=1.0 기본). 질량 참값 `m_A`는 **모델 입력이 아니라 손실에서만** 비교됨.

---

## 6. 중요한 설계 속성

| 속성 | 설명 |
|---|---|
| **동적 토큰 수** | positional embedding이 없어 Ni(256/576)가 바뀌어도 코드 수정 불필요. 타일 S만 바꾸면 됨. |
| **널 트랙 토큰** | 트랙 0개 오브젝트의 NaN softmax 방지. masked-mean 분모도 ≥1 보장. |
| **modality type 임베딩** | `img_type`/`trk_type`로 두 토큰 집합을 구분(세그먼트 임베딩 격). |
| **양방향·동시 갱신** | img↔trk 둘 다 서로를 참조, 같은 스텝에서 동시 갱신. |
| **fp32 + TF32** | AMP 미사용(NaN 안전), Ampere+에서 TF32로 속도. |
| **타일 업스케일 주의** | 이미지는 raw crop을 16배 nearest-repeat로 S에 타일 → 정보량 증가 아님, ES와 채널크기 맞추고 ResNet feature map을 비자명하게 만들기 위함. |

---

## 7. 어디서 쓰이나
- 학습: `train_multitask_cross_gun.py` (연속 `a_mass` 회귀 타깃).
- 평가: `eval_gunheavy.py` (gun/signal/background 3-class 스코어링).
- 48×48 cropped 데이터: `IMAGE_TILE_SIZE=768` 환경변수로 타일 크기만 바꿔 동일 로직 사용.

## 8. 관련 문서
- `docs/gunheavy-model-code-review.md` — 같은 모델의 라인바이라인 리뷰.
- `docs/gunheavy-pipeline-code-review.md` — 데이터 파이프라인(weight→compact→train).
- `docs/cross_attention_model.md` — 원본 단일태스크 분류기 설명.
