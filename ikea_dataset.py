from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from shutil import rmtree
from random import random
import requests
from tqdm import tqdm
from PIL import Image


class IkeaDataset(Dataset):

    def __init__(
        self,
        filepath: str,
        dataset_loc: str,
        train_data_ratio: float = 0.8,
    ):
        super().__init__()
        self.filepath = filepath
        self.dataset_loc = dataset_loc
        self.train_data_ratio = train_data_ratio
        # Integrate tqdm callbacks with pandas
        tqdm.pandas()

    def resolve_image_path(self, row: pd.Series) -> Path:
        return Path(
            self.dataset_loc, row.split, f"{row.product_name}_{row.id}.jpg"
        ).resolve()

    def save_image(self, row: pd.Series) -> None:
        # Set up file location, and create parents folders if needed
        file_loc = self.resolve_image_path(row)
        file_loc.parent.mkdir(parents=True, exist_ok=True)
        # Request the content of the 'image_link'
        image_data = requests.get(row.image_link).content
        # Save the image
        with open(file_loc, "wb") as file:
            file.write(image_data)

    def load_image_from_path(self, path: Path) -> Image.Image:
        return Image.open(path)

    def set_up_dataset(self) -> None:
        # Set up data directory, removing any existing data if necessary
        dataset_loc = Path(self.dataset_loc).resolve()
        rmtree(dataset_loc, ignore_errors=True)
        dataset_loc.mkdir(parents=True, exist_ok=True)

        # Load data, and create train/test splits based on 'product_name'
        self.data_df = pd.read_csv(self.filepath)
        split_df = (
            self.data_df.groupby("product_name")
            .apply(
                lambda g: "train" if random() < self.train_data_ratio else "test",
                include_groups=False,
            )
            .rename("split")
        ).reset_index()
        self.data_df = self.data_df.merge(
            split_df, on="product_name", how="left", validate="m:1"
        )

        self.data_df["id"] = self.data_df.groupby("product_name")[
            "image_link"
        ].transform(lambda g: range(len(g)))

        # For every row of data, download the image associated with 'image_link' and
        # store it as : <dataset_loc>/<split>/<product_name>XXX.jpg where XXX are numbers to differentiate duplicates
        self.data_df[["id", "product_name", "split", "image_link"]].progress_apply(
            lambda row: self.save_image(row),
            axis=1,  # Apply function for each row
        )

    def __getitem__(self, index):
        # Retrieve the index-th row from the dataframe
        row = self.data_df[index, :]
        # Retrieve matching image
        image_path = self.resolve_image_path(row)
        return row.product_name, self.load_image_from_path(image_path)

    def __len__(self):
        return len(self.data_df)


if __name__ == "__main__":
    IkeaDataset("./scrapped_data.csv", "./dataset").set_up_dataset()
