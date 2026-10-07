"""PyTorch Dataset for narrative consistency data."""
from torch.utils.data import Dataset


class NarrativeDataset(Dataset):
    """Expects a DataFrame with `caption` and `content` columns, plus `label`
    (train) and optionally `id`."""

    LABEL_MAP = {"consistent": 0, "contradict": 1, "contradiction": 1, "neutral": 0}

    def __init__(self, df, train=True):
        self.df = df.fillna("").astype(str)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.LABEL_MAP.get(row["label"].lower(), 0) if self.train else -1
        return row["caption"], row["content"], label, row.get("id", idx)
