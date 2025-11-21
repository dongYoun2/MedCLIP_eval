import torch
from sklearn.metrics import confusion_matrix


def tokenize_all_prompts(tokenizer, cls_prompts, class_names, device):
    all_prompts = []
    for class_name in class_names:
        all_prompts.extend(cls_prompts[class_name])

    tokenized = tokenizer(
        all_prompts,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in tokenized.items()}


@torch.no_grad()
def forward_and_fuse(model, pixel_values, all_tokenized, concat: bool = False):
    """
    Runs the model, computes fused embeddings using weights over all prompts.
    Returns: img_embeds, fused_embeds, logits
    """
    outputs = model(
        pixel_values=pixel_values,
        input_ids=all_tokenized["input_ids"],
        attention_mask=all_tokenized["attention_mask"],
    )

    img_embeds = outputs["img_embeds"]  # (B, 512)
    text_embeds = outputs["text_embeds"]  # (P, 512)
    logits = outputs["logits"]  # (B, P)

    # weights: (B, P)
    weights = torch.softmax(logits, dim=1)
    # weighted_text_embeds: (B, 512)
    weighted_text_embeds = weights @ text_embeds

    # fuse the embeddings
    if concat:
        fused_embeds = torch.cat([img_embeds, weighted_text_embeds], dim=1)  # (B, 1024)
    else:
        fused_embeds = img_embeds + weighted_text_embeds  # (B, 512)

    return img_embeds, fused_embeds, logits


@torch.no_grad()
def compute_embeddings_over_loader(
    model,
    dataloader,
    all_tokenized,
    num_classes,
    prompts_per_class,
    device,
    collect_for_calibration: bool = False,
    concat: bool = False,
):
    """
    For general use on calib/test loaders.
    If collect_for_calibration=True, also returns:
        - class_logits, preds, correct_mask
        - raw img_embeds, fused_embeds, labels
    Otherwise, only fused_embeds and labels.
    """

    image_embeddings_list = []
    fused_embeddings_list = []
    labels_list = []

    calib_class_logits_list = []
    calib_preds_list = []
    calib_correct_mask_list = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        img_embeds, fused_embeds, logits = forward_and_fuse(model, pixel_values, all_tokenized, concat=concat)

        # Move embeddings to CPU for storage
        image_embeddings_list.append(img_embeds.cpu())
        fused_embeddings_list.append(fused_embeds.cpu())
        labels_list.append(labels.cpu())

        if collect_for_calibration:
            # Aggregate prompt logits into class logits
            # logits: (B, num_classes * prompts_per_class)
            class_logits = logits.view(logits.size(0), num_classes, prompts_per_class).mean(dim=2)
            preds = class_logits.argmax(dim=1)
            correct = preds.cpu() == labels.cpu()

            calib_class_logits_list.append(class_logits.cpu())
            calib_preds_list.append(preds.cpu())
            calib_correct_mask_list.append(correct)

    all_fused_embeddings = torch.cat(fused_embeddings_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    if not collect_for_calibration:
        return {
            "fused_embeddings": all_fused_embeddings,
            "labels": all_labels,
        }

    # For calibration: concatenate everything
    all_image_embeddings = torch.cat(image_embeddings_list, dim=0)
    calib_class_logits = torch.cat(calib_class_logits_list, dim=0)
    calib_preds = torch.cat(calib_preds_list, dim=0)
    calib_correct_mask = torch.cat(calib_correct_mask_list, dim=0).bool()

    return {
        "image_embeddings": all_image_embeddings,
        "fused_embeddings": all_fused_embeddings,
        "labels": all_labels,
        "class_logits": calib_class_logits,
        "preds": calib_preds,
        "correct_mask": calib_correct_mask,
    }


def construct_prototypes(
    calib_results: dict,
    disease_classes: list[str],
    top_k: int = 30,
) -> dict[str, torch.Tensor]:
    """
    Construct confidence-weighted prototypes from calibration results.

    Uses top-K correctly classified examples per class (falling back to
    all GT examples if none are correct).
    """
    class_logits = calib_results["class_logits"]      # (N, C)
    labels = calib_results["labels"]                  # (N,)
    fused_embeds = calib_results["fused_embeddings"]  # (N, D)
    correct_mask = calib_results["correct_mask"]      # (N,)

    num_classes = len(disease_classes)
    class_probs = torch.softmax(class_logits, dim=1)  # (N, C)

    prototypes = {}

    for class_idx, class_name in enumerate(disease_classes):
        # Prefer correctly classified samples for this class
        cls_mask = (labels == class_idx) & correct_mask

        if cls_mask.sum() == 0:
            # Fallback: use all ground-truth samples of this class
            print(f"[WARN] {class_name}: no correctly classified samples, using all GT samples.")
            cls_mask = labels == class_idx

        cls_embeds = fused_embeds[cls_mask]           # (M, D)
        cls_conf = class_probs[cls_mask, class_idx]   # (M,)

        if cls_embeds.numel() == 0:
            # Extreme fallback: zero vector
            prototypes[class_name] = torch.zeros(fused_embeds.shape[1])
            print(f"[WARN] {class_name}: no samples at all, using zero prototype.")
            continue

        k = min(top_k, cls_embeds.size(0))
        topk_vals, topk_idx = torch.topk(cls_conf, k=k)  # (k,)

        weights = topk_vals / topk_vals.sum()
        proto = (cls_embeds[topk_idx] * weights.unsqueeze(1)).sum(dim=0)  # (D,)

        prototypes[class_name] = proto

    return prototypes



def classify_with_prototypes(
    test_fused_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    prototypes: dict[str, torch.Tensor],
    disease_classes: list[str],
):
    """
    Classify test samples via cosine similarity to class prototypes.
    Returns: predictions, overall accuracy, per-class accuracy dict, confusion matrix.
    """
    # (C, D)
    prototype_matrix = torch.stack([prototypes[c] for c in disease_classes])

    # Normalize for cosine similarity
    test_norm = test_fused_embeddings / test_fused_embeddings.norm(dim=-1, keepdim=True)
    proto_norm = prototype_matrix / prototype_matrix.norm(dim=-1, keepdim=True)

    # (N, C)
    similarities = test_norm @ proto_norm.T
    preds = similarities.argmax(dim=1)

    # Overall accuracy
    accuracy = (preds == test_labels).float().mean().item()

    # Per-class accuracy
    per_class_acc = {}
    for class_idx, class_name in enumerate(disease_classes):
        mask = test_labels == class_idx
        if mask.any():
            per_class_acc[class_name] = (
                (preds[mask] == test_labels[mask]).float().mean().item()
            )
        else:
            per_class_acc[class_name] = float("nan")

    # Confusion matrix (optional use)
    cm = confusion_matrix(test_labels.numpy(), preds.numpy())

    return preds, accuracy, per_class_acc, cm
