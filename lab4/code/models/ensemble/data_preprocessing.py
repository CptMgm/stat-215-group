import argparse

import numpy as np
import pandas as pd


def load_image_as_df(
    file_path: str,
    image_idx: int = None,
    use_only_engineered_features: bool=False
) -> pd.DataFrame:
    """
    Load an MISR image from a text file and return it as a DataFrame.

    :param file_path: The path to the text file containing the image data.
    :param image_idx: The index of the image.
                      If provided, an "image" column will be added to the DataFrame. (Default: None)
    :param use_only_engineered_features: Whether to use only the expert-engineered features. (Default: False)
    :return: The image data as a DataFrame.
    """
    data = np.loadtxt(file_path)
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Set column names
    df.columns = ["y", "x", "label", "NDAI", "SD", "CORR", "DF_angle", "CF_angle", "BF_angle", "AF_angle", "AN_angle"]
    # Add image column
    if image_idx is not None:
        df["image"] = image_idx
    # Drop rows with label 0
    df = df[df["label"] != 0]
    # Drop y and x columns
    df = df.drop(columns=["y", "x"])
    # Drop raw features
    if use_only_engineered_features:
        df = df.drop(columns=["DF_angle", "CF_angle", "BF_angle", "AF_angle", "AN_angle"])

    return df


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_only_engineered_features", action="store_true")
    args = parser.parse_args()
    only_engineered_features = args.use_only_engineered_features

    # Load and preprocess train images
    train_images = ["../../../data/image_data/image1.txt", "../../../data/image_data/image2.txt"]
    all_train_images = []
    for idx, train_image in enumerate(train_images):
        df = load_image_as_df(train_image, idx+1, use_only_engineered_features=only_engineered_features)
        print(f"Loaded train image {idx} with shape {df.shape}")
        df.to_csv(f"../../../data/preprocessed/image{idx+1}.csv", index=False)
        all_train_images.append(df)

    all_train_images_df = pd.concat(all_train_images)
    all_train_images_df.to_csv("../../../data/preprocessed/image1_2.csv", index=False)

    # Load and preprocess test image
    test_image = "../../../data/image_data/image3.txt"
    test_df = load_image_as_df(test_image, use_only_engineered_features=only_engineered_features)
    print(f"Loaded test image with shape {test_df.shape}")
    test_df.to_csv("../../../data/preprocessed/image3.csv", index=False)
