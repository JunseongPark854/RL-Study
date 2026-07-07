"""
vit_todo.py 구현이 맞는지 shape 단위로 검증합니다.
TODO를 하나 완료할 때마다 실행해 보세요.
"""

import torch

# 구현 중인 파일 — 완성되면 vit_todo에서 import
try:
    from vit_todo import pair, FeedForward, Attention, Transformer, ViT
except NotImplementedError as e:
    print(f"아직 구현 중: {e}")
    raise SystemExit(1)


def test_pair():
    assert pair(32) == (32, 32)
    assert pair((224, 384)) == (224, 384)
    print("✓ TODO 1 pair")


def test_feedforward():
    ff = FeedForward(dim=64, hidden_dim=128)
    x = torch.randn(2, 16, 64)
    out = ff(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print("✓ TODO 2 FeedForward")


def test_attention():
    attn = Attention(dim=64, heads=4, dim_head=16)
    x = torch.randn(2, 16, 64)
    out = attn(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print("✓ TODO 3 Attention")


def test_transformer():
    tr = Transformer(dim=64, depth=2, heads=4, dim_head=16, mlp_dim=128)
    x = torch.randn(2, 16, 64)
    out = tr(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print("✓ TODO 4 Transformer")


def test_vit_cls():
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=128,
        pool='cls',
    )
    img = torch.randn(2, 3, 32, 32)
    logits = model(img)
    assert logits.shape == (2, 10), f"expected (2, 10), got {logits.shape}"
    print("✓ TODO 5-6 ViT (cls pooling)")


def test_vit_mean():
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=64,
        depth=1,
        heads=4,
        mlp_dim=128,
        pool='mean',
    )
    img = torch.randn(2, 3, 32, 32)
    logits = model(img)
    assert logits.shape == (2, 10)
    print("✓ TODO 5-6 ViT (mean pooling)")


if __name__ == "__main__":
    test_pair()
    test_feedforward()
    test_attention()
    test_transformer()
    test_vit_cls()
    test_vit_mean()
    print("\n모든 테스트 통과! ViT end-to-end 구현 완료.")
