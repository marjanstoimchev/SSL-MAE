"""Shared model registry for pretrained timm backbones."""

MODEL_REGISTRY = {
    "deit": {
        "tiny": "deit_tiny_distilled_patch16_224",
        "small": "deit_small_distilled_patch16_224",
        "base": "deit_base_distilled_patch16_224",
    },
    "vit": {
        "tiny": "vit_tiny_patch16_224",
        "small": "vit_small_patch16_224",
        "base": "vit_base_patch16_224",
        "large": "vit_large_patch16_224",
    },
    "deit3": {
        "small": "deit3_small_patch16_224",
        "medium": "deit3_medium_patch16_224",
        "base": "deit3_base_patch16_224",
        "large": "deit3_large_patch16_224",
    },
}


def get_model_name(architecture, model_size):
    """Resolve architecture + size to a timm model name."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    if model_size not in MODEL_REGISTRY[architecture]:
        raise ValueError(f"Unknown model_size '{model_size}' for '{architecture}'. "
                         f"Choose from: {list(MODEL_REGISTRY[architecture].keys())}")
    return MODEL_REGISTRY[architecture][model_size]


def load_pretrained_vit(architecture, model_size):
    """Load a pretrained timm ViT and return (model, state_dict, kwargs)."""
    import timm
    import torch

    model_name = get_model_name(architecture, model_size)
    pretrained_model = timm.create_model(model_name, pretrained=True)
    pretrained_state = pretrained_model.state_dict()

    kwargs = dict(
        img_size=pretrained_model.patch_embed.img_size[0],
        patch_size=pretrained_model.patch_embed.patch_size[0],
        embed_dim=pretrained_model.embed_dim,
        depth=len(pretrained_model.blocks),
        num_heads=pretrained_model.blocks[0].attn.num_heads,
        mlp_ratio=pretrained_model.blocks[0].mlp.fc1.out_features / pretrained_model.embed_dim,
        num_classes=0,
        class_token=pretrained_model.cls_token is not None,
        no_embed_class=pretrained_model.no_embed_class,
    )

    return model_name, pretrained_state, kwargs
