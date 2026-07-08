"""
ViT 손코딩 연습용 스켈레톤
========================
각 TODO의 GOAL을 읽고 직접 구현하세요.
"""

import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# TODO 1 — pair
# GOAL: int 하나 또는 (h, w) 튜플을 항상 (h, w) 튜플로 반환
# ---------------------------------------------------------------------------
# 실제로 사용되는 흐름 : image_h, image_w = pari(image_size)
# input : int or tuple -> always return tuple!
def pair(t):
    if (isinstance(t, tuple)) and len(t) == 2:
        return t
    elif (isinstance(t, int)):
        return (t,t)
    raise ValueError("Input must be an int or a tuple of length 2")

# ---------------------------------------------------------------------------
# TODO 2 — FeedForward
# GOAL: Pre-LN MLP — LayerNorm → Linear → GELU → Dropout → Linear → Dropout
#       입력/출력 shape: (B, N, dim)
# ---------------------------------------------------------------------------
# N : number of tokens, dim : embedding dim
class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # TODO 2: self.net = nn.Sequential(...) 정의
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            # Normalization : Internal Covariate Shift
            # Token vector = [x1, x2, ..., xd]
            # hat(x) = (x - mean(x)) / sqrt(var(x) + eps) -> eps : divide by zero 방지용
            # Z-Score normalization
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Regularization : Overfitting 방지용
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# TODO 3 — Attention
# GOAL: Multi-head self-attention (Pre-LN)
#       Q,K,V 분리 → (B, heads, N, dim_head) → softmax(QKᵀ/√d)V → concat → Linear
# ---------------------------------------------------------------------------
class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        # dim = d, head = h, dim_head = d_k (d / h)
        # Q: 내부에서 dim/heads를 dim_head로 정의하면 되는것을 왜 dim_head를 인자로 받는가?
        super().__init__()
        inner_dim = dim_head * heads # d_k * h = d
        project_out = not (heads == 1 and dim_head == dim)
        # W_0 Linear transformation 필요여부 flag
        self.innerdim = inner_dim

        self.heads = heads
        self.scale = dim_head ** 0.5
        # Q * K^T / sqrt(d_k) 계산시 사용
        # Q : Q와 K의 유사도를 측정하기위해서 |Q||K|로 나누어 정규화가 아닌 sqrt(d_k)로 나누는 이유?

        # TODO 3: self.norm, self.to_qkv, self.attend, self.dropout, self.to_out 정의
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        # 하나의 patch (size d) -> Q,K,V의 (d * 3) 변환
        self.attend = nn.Softmax(dim=-1) # 하나의 row에 대해 softmax 적용 -> V와 곱연산 후 Result row가 됨
        self.dropout = nn.Dropout(dropout)
        # self.to_out : project_out flag 여부에 따라 W_0 Linear transformation Net 적용
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # TODO 3: 아래 단계를 직접 구현
        #   1) x = self.norm(x)
        #   2) q, k, v = to_qkv(x).chunk(3, dim=-1) 후 rearrange로 head 분리
        #   3) dots = matmul(q, k.T) * scale → softmax → dropout
        #   4) out = matmul(attn, v) → rearrange로 head 합치기 → to_out
        x = self.norm(x)
        # self.qkv(x) : (N, 3*d) 1개의 patch에 대한 Q,K,V weight가중시킨 vector -> N장
        # 어떻게 Q, K, V를 qkv(x)에서 분리할까?
        # qkv(x) = (N, 3*d) 실제 생김새
        # B = 1 -> [[Q1, K1, V1], [Q2, K2, V2], ..., [QN, KN, VN]]
        # Q = [[Q1], [Q2], ..., [QN]]
        # K = [[K1], [K2], ..., [KN]]
        # V = [[V1], [V2], ..., [VN]] 형태로 나누고 나서 Attention 계산을 진행해야함
        # 즉, 마지막 축 [Q d개 | K d개 | V d개]를 3개의 축으로 나누어야함
        # [0...d-1 | d...2d-1 | 2d...3d-1]
        qkv = self.to_qkv(x)
        q = qkv[..., :self.innerdim]
        k = qkv[..., self.innerdim:2*self.innerdim]
        v = qkv[..., 2*self.innerdim:]
        # 이제 q, k, v를 head수만큼 분리해야함. d -> d/h 사용
        B, N, d = q.shape
        h = self.heads
        dh = d // h # dim_head
        q = q.reshape(B, N, h, dh).transpose(1, 2) # (B, N, d) -> (B, h, N, d/h)
        k = k.reshape(B, N, h, dh).transpose(1, 2)
        v = v.reshape(B, N, h, dh).transpose(1, 2)
        dots = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attn = self.attend(dots) # softmax(QK^T / sqrt(d_k))
        out = torch.matmul(attn, v)
        concat_out = out.transpose(1, 2).reshape(B, N, d) # (B, h, N, d/h) -> (B, N, h, d/h) -> (B, N, d)
        return concat_out

# ---------------------------------------------------------------------------
# TODO 4 — Transformer
# GOAL: depth개의 (Attention, FeedForward) 블록을 residual로 쌓기
# ---------------------------------------------------------------------------
class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        # TODO 4: depth번 반복하며 [Attention, FeedForward] 쌍을 layers에 append
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        # TODO 4: for attn, ff in self.layers: residual 두 번 적용 후 norm 반환
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            # Residual Connection : Gradient Vanishing 문제 해결, 학습 안정화
        return self.norm(x)

# ---------------------------------------------------------------------------
# TODO 5 & 6 — ViT
# GOAL 5 (__init__): 패치 임베딩, CLS 토큰, pos embedding, transformer, mlp_head
# GOAL 6 (forward):  이미지 → 패치 → +pos → transformer → pool → 분류
# ---------------------------------------------------------------------------
class ViT(Module):
    def __init__(
        self,
        *,
        image_size : tuple, # ex: 32 x 32 image
        patch_size : tuple, # ex: 4 x 4 patch -> 64 patches
        num_classes, # ex: 10 classes -> 정답 action수?
        dim, # patch -> flatten 후 Linear embedding 하고난 크기?
        depth, # transformer loop 횟수
        heads, # head 수
        mlp_dim, # FFN hidden dim
        pool='cls', # cls
        channels=3, # RGB
        dim_head=64, # head당 size
        dropout=0., # dropout rate
        emb_dropout=0., # embedding dropout rate
    ):
        super().__init__()

        # TODO 5: image_height/width, patch_height/width = pair(...)
        # TODO 5: assert로 image가 patch로 나누어떨어지는지 확인
        # TODO 5: num_patches, patch_dim 계산
        # TODO 5: pool 타입에 따라 num_cls_tokens 결정
        # =======================================================================
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        if (image_height % patch_height != 0) or (image_width % patch_width != 0):
            raise ValueError("Image dimensions must be divisible by patch dimensions")
        num_patches = (image_height // patch_height) * (image_width // patch_width) # N
        patch_dim = channels * patch_height * patch_width # (C * P * P)
        num_cls_tokens = 1 if pool == 'cls' else 0
        
        # 아래는 TODO 5에서 채울 멤버들 (참고용 시그니처)
        # self.to_patch_embedding = nn.Sequential(...)
        # self.cls_token = nn.Parameter(...)
        # self.pos_embedding = nn.Parameter(...)
        # self.dropout = nn.Dropout(emb_dropout)
        # self.transformer = Transformer(...)
        # self.pool = pool
        # self.to_latent = nn.Identity() -> 왜 써야하는가??
        # self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

        # input : (B, C, H, W) -> (B, N, P*P*C)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c i(h p1) (w p2) -> b (h w) (p1 p2 c)'),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        # nn.LayerNorm() 남발해도 괜찮나?? 의미가있는것?
        self.cls_token = nn.Parameter(torch.randn(1, num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_cls_tokens, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        # pool : cls하나 뽑아서 FFN에 보내기
        # pool 굳이 정의해서 구별한 이유? 다른 pooling?
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        # TODO 6: 전체 forward 파이프라인 구현
        #   batch = img.shape[0]
        #   x = self.to_patch_embedding(img)
        #   cls_tokens = repeat(self.cls_token, '... d -> b ... d', b=batch)
        #   x = cat(cls, patches), + pos_embedding, dropout, transformer
        #   pool → mlp_head
        batch = img.shape[0]
        x = self.to_patch_embedding(img) # (B, N, dim) patch화 완료
        # 현재 CLS token은 1개만 존재함. Batch마다 동일한 CLS token생성후 합치기
        cls_tokens = self.cls_token # (1, 1, dim)
        cls_tokens = cls_tokens.repeat(batch, 1, 1) # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 1+N, dim)
        x += self.pos_embedding # (B, 1+N, dim)
        x = self.dropout(x)
        x = self.transformer(x)
        # (B, 1+N, dim) -> (B, N+1, dim*3) -> (B, h, N+1, dim/h) -> (B, h, N+1, dim/h) -> (B, N+1, dim)
        if self.mlp_head is None:
            return x
        x = x[:, 0] # Batch마다 0번 index의 patch 가져오기 = cls token
        x = self.to_latent(x) # (B, dim)
        x = self.mlp_head(x) # (B, num_classes)
        return x