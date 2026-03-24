from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "embedding_clustering"


@dataclass
class ImageRecord:
    split: str
    sample_id: str
    label: str
    label_int: int
    view: str
    image_path: Path


class SingleViewDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        image = self.transform(image)
        return image, rec.sample_id, rec.label, rec.label_int, rec.split


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_records() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    label_map = {"stable": 0, "unstable": 1}
    for split in ["train", "dev"]:
        df = pd.read_csv(DATA_DIR / f"{split}.csv", encoding="utf-8-sig")
        for row in df.itertuples(index=False):
            sample_id = row.id
            label = row.label
            for view in ["front", "top"]:
                image_path = DATA_DIR / split / sample_id / f"{view}.png"
                records.append(
                    ImageRecord(
                        split=split,
                        sample_id=sample_id,
                        label=label,
                        label_int=label_map[label],
                        view=view,
                        image_path=image_path,
                    )
                )
    return records


def extract_embeddings(records: list[ImageRecord], model_name: str, img_size: int, batch_size: int) -> pd.DataFrame:
    device = torch.device("cpu")
    transform = build_transform(img_size)
    dataset = SingleViewDataset(records, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
    model.eval()
    model.to(device)

    feature_rows = []
    with torch.no_grad():
        for images, sample_ids, labels, label_ints, splits in tqdm(loader, desc="extract"):
            feats = model(images.to(device)).cpu().numpy()
            for feat, sample_id, label, label_int, split, rec in zip(
                feats,
                sample_ids,
                labels,
                label_ints.numpy(),
                splits,
                records[len(feature_rows): len(feature_rows) + len(feats)],
            ):
                row = {
                    "split": split,
                    "sample_id": sample_id,
                    "label": label,
                    "label_int": int(label_int),
                    "view": rec.view,
                }
                for i, value in enumerate(feat):
                    row[f"emb_{i:04d}"] = float(value)
                feature_rows.append(row)
    return pd.DataFrame(feature_rows)


def cluster_purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "c": clusters})
    correct = 0
    for _, part in df.groupby("c"):
        correct += int(part["y"].value_counts().max())
    return float(correct / len(df))


def analyze_view(view_df: pd.DataFrame, out_dir: Path, random_state: int = 42) -> pd.DataFrame:
    emb_cols = [c for c in view_df.columns if c.startswith("emb_")]
    X = view_df[emb_cols].to_numpy()
    y = view_df["label_int"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_2d = PCA(n_components=2, random_state=random_state)
    pca_coords = pca_2d.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=random_state)
    tsne_coords = tsne.fit_transform(X_scaled)

    metric_rows = []
    clustered_df = view_df[["split", "sample_id", "label", "label_int", "view"]].copy()
    clustered_df["pca_x"] = pca_coords[:, 0]
    clustered_df["pca_y"] = pca_coords[:, 1]
    clustered_df["tsne_x"] = tsne_coords[:, 0]
    clustered_df["tsne_y"] = tsne_coords[:, 1]

    for n_clusters in [2, 4]:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        cluster_ids = km.fit_predict(X_scaled)
        clustered_df[f"kmeans_{n_clusters}"] = cluster_ids
        metric_rows.append(
            {
                "view": view_df["view"].iloc[0],
                "n_samples": len(view_df),
                "n_clusters": n_clusters,
                "silhouette": float(silhouette_score(X_scaled, cluster_ids)),
                "ari_vs_label": float(adjusted_rand_score(y, cluster_ids)),
                "nmi_vs_label": float(normalized_mutual_info_score(y, cluster_ids)),
                "purity_vs_label": float(cluster_purity(y, cluster_ids)),
                "pca_var_1": float(pca_2d.explained_variance_ratio_[0]),
                "pca_var_2": float(pca_2d.explained_variance_ratio_[1]),
            }
        )

    clustered_df.to_csv(out_dir / f"{view_df['view'].iloc[0]}_embedding_projection.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    label_palette = {"stable": "#2b8cbe", "unstable": "#de2d26"}

    sns.scatterplot(data=clustered_df, x="pca_x", y="pca_y", hue="label", palette=label_palette, s=35, alpha=0.8, ax=axes[0, 0])
    axes[0, 0].set_title(f"{view_df['view'].iloc[0]} PCA by label")

    sns.scatterplot(data=clustered_df, x="tsne_x", y="tsne_y", hue="label", palette=label_palette, s=35, alpha=0.8, ax=axes[0, 1])
    axes[0, 1].set_title(f"{view_df['view'].iloc[0]} t-SNE by label")

    sns.scatterplot(data=clustered_df, x="pca_x", y="pca_y", hue="kmeans_2", palette="Set2", s=35, alpha=0.8, ax=axes[1, 0])
    axes[1, 0].set_title(f"{view_df['view'].iloc[0]} PCA by KMeans-2")

    sns.scatterplot(data=clustered_df, x="tsne_x", y="tsne_y", hue="kmeans_2", palette="Set2", s=35, alpha=0.8, ax=axes[1, 1])
    axes[1, 1].set_title(f"{view_df['view'].iloc[0]} t-SNE by KMeans-2")

    for ax in axes.flat:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / f"{view_df['view'].iloc[0]}_embedding_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="resnet18.a1_in1k")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records()

    all_embeddings = []
    metric_dfs = []
    for view in ["front", "top"]:
        view_records = [r for r in records if r.view == view]
        emb_df = extract_embeddings(view_records, args.model_name, args.img_size, args.batch_size)
        emb_df.to_csv(args.output_dir / f"{view}_embeddings.csv", index=False, encoding="utf-8-sig")
        all_embeddings.append(emb_df)
        metric_dfs.append(analyze_view(emb_df, args.output_dir))

    summary_df = pd.concat(metric_dfs, ignore_index=True)
    summary_df.to_csv(args.output_dir / "embedding_cluster_metrics.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
