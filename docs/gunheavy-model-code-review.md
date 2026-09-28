# gunheavy 학습 모델 — 코드 라인바이라인 리뷰 (cross-attention multi-task)

**결론 먼저**: gunheavy_v1이 학습 중인 모델은 **기존 cross-attention multi-task 모델과 동일**하다.
백본 `CrossAttentionFusion`(`train_fusion_cross_attention.py`)에 분류 헤드 + 질량 회귀 헤드를 붙인
`MultiTaskModel`(`multitask_common.py`)이며, `train_multitask_cross_gun.py`는 기존
`train_multitask_cross.py`와 **모델·손실이 완전히 같고** 질량 타깃 소스만 (이산 sample_id →
연속 `a_mass`) 다르다. 이 문서는 논문의 *Model / Architecture / Loss* 절 근거용이다.

관련 파일:
- `train_fusion_cross_attention.py` (489줄) — `CrossAttentionFusion` 백본 + 데이터 결합
- `multitask_common.py` — `MultiTaskModel`, 헤드, 손실, 질량 변환
- `train_multitask_cross_gun.py` — 학습 루프 (gun 연속 질량)

---

## 0. 전체 구조 한눈에

```
 calo image [B, C, 32, 32]              track point cloud [B, Nt, F] (+ mask [B,Nt])
        │                                        │
   ResNet(-avgpool,-fc)                     Linear->LN->GELU->Linear   (trk_in)
   -> [B, 512, 1, 1]                        + null token 붙이고
   1x1 Conv (img_proj)                      TransformerEncoder(self-attn, track_depth)
   -> Ni=1 image token [B,1,d]             -> [B, 1+Nt, d]
        │  + img_type                            │ + trk_type
        └───────────────┬────────────────────────┘
              bidirectional cross-attention  × depth
              img <- track (img2trk),  track <- img (trk2img)
                        │
          masked-mean pool 각 모달리티 -> concat [B, 2d]
                        │  out_norm (LayerNorm)
                        ▼  feat [B, 512]      (backbone.head = Identity)
        ┌───────────────┴───────────────┐
   cls_head Linear(512,1)          reg_head MLP(512->512->1)
   -> cls_logit [B]                -> reg_mu [B] = 예측 log(m_A)
        │                               │
   BCE(weight=w)              λ · MAE(reg_mu, log m_A) · [signal mask]
```

핵심 수치(기본값): `d_model=256`, `feat_dim = 2·d_model = 512`, `num_heads=4`,
cross-depth `depth=2`, `track_depth=2`, `mlp_ratio=4.0`, `dropout=0.1`, backbone `resnet18`.

**주목할 점**: 입력 calo 크롭이 32×32이고 resnet18이 32× 다운샘플하므로 spatial feature map이
**1×1로 줄어 image 토큰이 Ni=1개**가 된다. 즉 이미지는 사실상 하나의 512-d 요약 벡터 토큰으로
cross-attention에 들어가고, track 토큰들이 그 이미지 요약에 attend 하는 구조다 (논문에 명시할 사항).

---

## 1. 데이터 결합 — `CombinedDataset` / `collate_combined`

### `CombinedDataset` (L63–87)
- 같은 manifest 행에서 나온 image 데이터셋과 track 데이터셋을 짝지음.
- L76 `assert int(img["object_idx"]) == int(trk["object_idx"])` — **같은 오브젝트인지 방어적 확인**
  (이미지와 트랙이 어긋나면 즉시 실패).
- `__getitem__`은 image/points/label/weight/pt/object_idx 등을 한 dict로 반환.

### `collate_combined` (L90–104)
- L91 이미지 스택 `[B, C, 32, 32]`.
- L92–100 **가변 길이 트랙 패딩**: 배치 내 최대 트랙 수 `max_n`으로 zero-pad, `mask[i,:n]=True`
  (유효 토큰 표시). 트랙 0개인 오브젝트도 `max_n>=1` 보장(L92).
- 결과: `points [B, max_n, F]`, `mask [B, max_n]`(bool, True=유효), 나머지 스칼라는 stack.

---

## 2. `CrossAttentionBlock` (L110–131) — pre-LN 교차 어텐션 1블록

```python
def forward(self, q, kv, kv_key_padding_mask=None):
    qn, kvn = self.norm_q(q), self.norm_kv(kv)                 # pre-LayerNorm
    attn_out, _ = self.attn(qn, kvn, kvn, key_padding_mask=..., need_weights=False)
    q = q + attn_out                                          # residual (attn)
    q = q + self.ff(self.norm_ff(q))                          # residual (FFN)
    return q
```
- **Pre-LN 트랜스포머 블록**: query 토큰이 key/value 토큰에 attend → residual → FFN → residual.
- `nn.MultiheadAttention(..., batch_first=True)` (L117). `key_padding_mask`로 패딩 트랙 무시.
- FFN = `Linear(d, 4d) -> GELU -> Dropout -> Linear(4d, d)` (L120–123).

---

## 3. `CrossAttentionFusion.__init__` (L134–175)

- **이미지 타워** (L139–143):
  - `build_resnet(backbone, in_channels)` — 입력 채널 수 = calo/ES 평면 스택 수(검출기·ES 여부에 따라).
  - `nn.Sequential(*list(base.children())[:-2])` — resnet에서 **avgpool·fc 제거** → spatial feature
    map `[B, feat_c, h, w]` 유지 (`feat_c`=512 for resnet18).
  - `img_proj = Conv2d(feat_c, d_model, 1)` — 1×1 conv로 채널을 `d_model`로 사영.
- **트랙 타워** (L146–153):
  - `trk_in`: `Linear(F, d) -> LN -> GELU -> Linear(d, d)` 점별 임베더.
  - `trk_encoder`: `TransformerEncoder`(self-attention, `track_depth`층, pre-LN, GELU) — 트랙들끼리
    문맥 교환.
- **특수 토큰** (L157–159):
  - `trk_null` — 항상 유효한 "널 트랙" 토큰. **트랙 0개인 오브젝트가 all-masked key set이 되어
    softmax가 NaN 나는 것을 방지** (병합전자에 트랙이 없을 수 있어 필수).
  - `img_type`, `trk_type` — 모달리티 구분용 학습 임베딩(BERT의 segment embedding 격).
- **교차 어텐션 스택** (L161–164): `img2trk`, `trk2img` 각각 `depth`개 블록 (양방향 융합).
- **출력부** (L166–170): `out_norm = LayerNorm(2d)`, `head = Linear(2d,d)->GELU->Dropout->Linear(d,1)`.
  - ⚠️ multi-task에서는 이 `head`가 `nn.Identity()`로 교체되어 **백본이 `out_norm(feat)` [B,2d]를 반환**
    (분류/회귀 헤드는 `MultiTaskModel`이 따로 붙임).
- L171 `reset_parameters()` — 3개 특수 토큰을 `trunc_normal_(std=0.02)`로 초기화.

---

## 4. `CrossAttentionFusion.forward` (L177–202) — 핵심 데이터 흐름

```python
B = image.size(0)
fmap = self.image_backbone(image)                    # [B, 512, 1, 1]  (32x32 -> 1x1)
img  = self.img_proj(fmap).flatten(2).transpose(1,2) # [B, Ni=1, d]
img  = img + self.img_type                            # 모달리티 태그
```
- L180–182: 이미지 → feature map → 1×1 conv → **토큰 시퀀스로 평탄화** (`flatten(2)`=h·w, 여기선 1).

```python
trk = self.trk_in(points)                            # [B, Nt, d]
null = self.trk_null.expand(B, -1, -1)               # [B, 1, d]
trk = torch.cat([null, trk], dim=1)                  # [B, 1+Nt, d]  널 토큰 선두 삽입
valid = cat([ones(B,1), mask], dim=1)                # [B, 1+Nt]  널은 항상 유효
kpm = ~valid                                         # True = 패딩(무시)
trk = self.trk_encoder(trk, src_key_padding_mask=kpm)
trk = trk + self.trk_type
```
- L184–191: 트랙 임베딩 → **널 토큰을 맨 앞에 붙이고** → 패딩 마스크 `kpm` 구성(널은 항상 유효) →
  self-attention 인코더 → 모달리티 태그.

```python
for a, b in zip(self.img2trk, self.trk2img):
    new_img = a(img, trk, kv_key_padding_mask=kpm)   # 이미지 토큰이 트랙에 attend
    new_trk = b(trk, img, kv_key_padding_mask=None)  # 트랙 토큰이 이미지에 attend
    img, trk = new_img, new_trk                      # 동시 갱신(순차 오염 방지)
```
- L193–196: **양방향 교차 어텐션**을 `depth`번. 이미지↔트랙이 서로 정보를 주고받음.
  트랙→이미지 방향은 이미지 key에 패딩이 없어 `kv_key_padding_mask=None`.

```python
img_pool = img.mean(dim=1)                                   # 이미지 토큰 평균 (Ni=1이면 그 토큰)
m = valid.unsqueeze(-1).float()
trk_pool = (trk * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0) # 유효 토큰만 masked-mean
feat = cat([img_pool, trk_pool], dim=1)                      # [B, 2d]
return self.head(self.out_norm(feat))                        # head=Identity -> [B, 2d]
```
- L198–202: 각 모달리티 **masked-mean 풀링** 후 concat → `out_norm` → (Identity) 반환.
  `clamp_min(1.0)` (L200) — 분모 0 방지(널 토큰 덕에 최소 1은 유효).

---

## 5. `MultiTaskModel` + 헤드 (`multitask_common.py` L102–122)

```python
def make_heads(feat_dim, dropout=0.1):                       # L102
    cls_head = nn.Linear(feat_dim, 1)                        # 분류: 선형 1개
    reg_head = nn.Sequential(nn.Linear(feat_dim, feat_dim),  # 회귀: 2층 MLP
                             nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(feat_dim, 1))
    return cls_head, reg_head

class MultiTaskModel(nn.Module):                            # L110
    def __init__(self, backbone, feat_dim, dropout=0.1):
        self.backbone = backbone                            # head=Identity인 백본
        self.cls_head, self.reg_head = make_heads(feat_dim, dropout)
    def forward(self, *inputs):                             # L120
        feat = self.backbone(*inputs)                       # [B, feat_dim=512]
        return self.cls_head(feat).squeeze(1), self.reg_head(feat).squeeze(1)
```
- **하드 파라미터 공유**: 백본(교차어텐션 융합)이 공통 표현 `feat[B,512]`를 만들고, 두 헤드가 분기.
- 반환 `(cls_logit[B], reg_mu[B])`. `reg_mu` = 예측 **log(m_A)** (선형 mass는 `inv_transform=exp`).
- gunheavy는 `feat_dim = 2·d_model = 512`로 생성 (`train_multitask_cross_gun.py`).

---

## 6. 손실 — `multitask_loss` (`multitask_common.py` L62–72)

```python
def multitask_loss(cls_logit, reg_mu, label, mass_target, sig_mask, weight=None, lam=1.0):
    label = label.float()
    if weight is None: weight = torch.ones_like(label)
    bce = F.binary_cross_entropy_with_logits(cls_logit, label, weight=weight, reduction="mean")
    denom = sig_mask.sum().clamp_min(1.0)
    mae = ((reg_mu - mass_target).abs() * sig_mask).sum() / denom
    return bce + lam*mae, bce.detach(), mae.detach()
```
$$ L = \underbrace{\mathrm{BCE}(\text{cls}, y; w)}_{\text{모든 오브젝트, 가중}} \;+\; \lambda\cdot
   \underbrace{\frac{\sum_i |\,\mu_i - \log m_{A,i}\,|\cdot s_i}{\max(\sum_i s_i,\,1)}}_{\text{signal만 (}s_i=1\text{), L1}} $$
- **분류(BCE)**: 전체 오브젝트, per-object `weight` 적용(학습 시 실제 weight 들어감 — 검증됨).
- **회귀(MAE on log-mass)**: `sig_mask`로 **signal만** 반영(배경은 0으로 마스킹). 분모 clamp로 배경만
  있는 배치에서도 안전. gunheavy에서 `sig_mask = (a_mass > 0)`, `mass_target = log(a_mass)`.
- 참고: 이 signal-masked 회귀 항은 Fast R-CNN의 `L_cls + λ[u≥1]L_loc` 형태(배경 제외)와 동형이고,
  다중태스크 가중 λ는 Kendall 2018 계열의 태스크 밸런싱과 같은 역할.

---

## 7. gunheavy 학습 루프에서의 차이 (`train_multitask_cross_gun.py`)

모델·손실은 위와 **동일**. 유일한 차이는 회귀 타깃 소스 (`run_epoch`):
```python
am = a_mass_all[batch["object_idx"].numpy()]   # compact의 연속 a_mass (배경=0)
sig_np = am > 0
logm_np[sig_np] = mc.transform_mass(am[sig_np]) # log(m_A), signal만
```
기존 `train_multitask_cross.py`는 `mass_targets_from_codes`(이산 sample_id→질량)를 썼다.
→ gun의 **연속 질량(0.1–50 GeV)** 회귀를 위해 compact에 실어온 per-object `a_mass`를 직접 사용.

---

## 8. 파라미터/차원 요약 (논문 표 재료)

| 구성요소 | 사양 |
|---|---|
| 이미지 백본 | ResNet-18 (avgpool·fc 제거), 입력 32×32×C → feature map 512×1×1 |
| 이미지 토큰 | 1×1 conv 사영 후 Ni=1개 (d=256) |
| 트랙 임베더 | Linear→LN→GELU→Linear (F→256) |
| 트랙 self-attn | TransformerEncoder, track_depth=2, heads=4, pre-LN |
| 특수 토큰 | trk_null, img_type, trk_type (학습, trunc_normal std=0.02) |
| 교차 어텐션 | 양방향(img↔trk), depth=2, heads=4, mlp_ratio=4 |
| 풀링 | 모달리티별 masked-mean → concat 2d=512 → LayerNorm |
| 분류 헤드 | Linear(512, 1) → BCE |
| 회귀 헤드 | Linear(512,512)→GELU→Dropout→Linear(512,1) → MAE(log m_A) |
| 손실 | BCE(weighted) + λ·MAE(log-mass, signal-only), λ=1.0 |
| 옵티마이저 | AdamW lr 1e-4, wd 1e-4, ReduceLROnPlateau(val AUC), fp32+TF32 |

## 9. 관련
- 데이터 파이프라인: `docs/gunheavy-pipeline-code-review.md`
- 메모리: `project-gunheavy-pipeline`.
