# ViT 손코딩 학습 프로젝트

GPT 없이 Vision Transformer를 **직접** 구현해 보는 연습용 프로젝트입니다.  
빈 화면에서 0→100을 만들지 않고, **뼈대 + TODO** 방식으로 한 단계씩 채워 나갑니다.

## 사용 방법

1. `vit_todo.py`만 열고 작업합니다. (`vit_reference.py`는 막혔을 때만 참고)
2. README의 **학습 순서**대로 TODO를 하나씩 완료합니다.
3. 각 TODO 아래 `GOAL`을 읽고, 스스로 구현한 뒤 테스트를 실행합니다.

```bash
pip install -r requirements.txt
python test_shapes.py
```

모든 테스트가 통과하면 ViT forward가 end-to-end로 동작하는 것입니다.

---

## 학습 순서 (권장)

| 단계 | TODO | 배우는 것 |
|------|------|-----------|
| 1 | `pair` | 이미지/패치 크기를 (H, W) 튜플로 통일 |
| 2 | `FeedForward` | Transformer의 MLP 블록 (Pre-LN 구조) |
| 3 | `Attention` | Q·Kᵀ·V self-attention + multi-head |
| 4 | `Transformer` | residual connection으로 블록 쌓기 |
| 5 | `ViT.__init__` | 패치 임베딩, CLS 토큰, positional embedding |
| 6 | `ViT.forward` | 패치화 → transformer → pooling → 분류 |

---

## 각 TODO 상세 목표

### TODO 1 — `pair(t)`
**GOAL:** 정수 하나 `(32, 32)`, 튜플 `(32, 32)` 모두 `(height, width)` 형태로 받을 수 있게 한다.

**체크:** `pair(32) == (32, 32)`, `pair((224, 384)) == (224, 384)`

---

### TODO 2 — `FeedForward`
**GOAL:** 입력 `x` (B, N, dim)에 대해 아래 순서의 MLP를 통과시킨다.

```
LayerNorm(dim) → Linear(dim, hidden_dim) → GELU → Dropout
→ Linear(hidden_dim, dim) → Dropout
```

**힌트:** `nn.Sequential`로 `__init__`에 정의하고 `forward`는 한 줄이면 충분하다.

**체크:** 출력 shape == 입력 shape `(B, N, dim)`

---

### TODO 3 — `Attention` (핵심)
**GOAL:** Pre-LN self-attention을 구현한다.

1. `x`에 `LayerNorm` 적용
2. `to_qkv` Linear로 Q, K, V 생성 후 head 차원으로 reshape  
   - `(B, N, inner_dim)` → `(B, heads, N, dim_head)`
3. `dots = Q @ Kᵀ * scale`  (`scale = dim_head ** -0.5`)
4. `softmax(dots)` → dropout → `attn @ V`
5. head를 다시 합쳐 `(B, N, inner_dim)` → `to_out` Linear

**einops 힌트:**
```python
rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
rearrange(out, 'b h n d -> b n (h d)')
```

**체크:** 출력 shape == `(B, N, dim)`

---

### TODO 4 — `Transformer`
**GOAL:** `depth`개의 (Attention, FeedForward) 쌍을 residual로 연결한다.

```python
for attn, ff in self.layers:
    x = attn(x) + x   # residual
    x = ff(x) + x     # residual
return self.norm(x)
```

**체크:** 출력 shape == 입력 shape `(B, N, dim)`

---

### TODO 5 — `ViT.__init__`
**GOAL:** ViT에 필요한 모듈과 파라미터를 선언한다.

1. `image_size`, `patch_size`를 `pair()`로 (H, W) 변환
2. `num_patches = (H // pH) * (W // pW)` 계산
3. `to_patch_embedding`: Rearrange → LayerNorm → Linear → LayerNorm
4. `cls_token`: `(1, dim)` learnable parameter (`pool='cls'`일 때만)
5. `pos_embedding`: `(num_patches + num_cls_tokens, dim)` learnable parameter
6. `transformer`, `mlp_head` (num_classes > 0일 때) 생성

**Rearrange 패턴:**
```
'b c (h p1) (w p2) -> b (h w) (p1 p2 c)'
```

**체크:** `model(img)` 호출 시 에러 없이 forward 가능

---

### TODO 6 — `ViT.forward`
**GOAL:** 이미지 텐서를 클래스 로짓으로 변환한다.

1. `to_patch_embedding(img)` → `(B, num_patches, dim)`
2. CLS 토큰을 배치 차원으로 복제해 앞에 concat → `(B, 1 + num_patches, dim)`
3. positional embedding 더하기 (`pos_embedding[:seq]`)
4. dropout → transformer
5. pooling: `cls`면 `x[:, 0]`, `mean`이면 `x.mean(dim=1)`
6. `mlp_head`로 `(B, num_classes)` 반환

**체크:** `python test_shapes.py` 전체 통과

---

## 막혔을 때

1. 해당 TODO의 **GOAL**만 다시 읽기
2. `test_shapes.py`에서 실패한 테스트의 shape 메시지 확인
3. 그래도 안 되면 `vit_reference.py`에서 **해당 함수만** 열어보기 (전체 복붙은 지양)

## 다음 단계 (ViT 완성 후)

- CIFAR-10 / MNIST로 학습 루프 작성
- `pool='cls'` vs `pool='mean'` 비교 실험
- `depth`, `heads`, `patch_size` ablation
