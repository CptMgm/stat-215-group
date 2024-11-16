import argparse

import pandas as pd

from data_preprocessing import load_image_as_df
from autogluon.tabular import TabularDataset, TabularPredictor


if __name__ == "__main__":
    # Parse arguments and set output path
    parser = argparse.ArgumentParser()
    parser.add_argument("test_img", type=int)
    args = parser.parse_args()
    test_img = args.test_img
    output_path = "StabilityAnalysisModels/run_{test_img}"

    # Load and preprocess data
    data_dir = "../../../data/image_data"
    image_files = [f"{data_dir}/image{i}.txt" for i in range(1, 4)]
    image_dfs = [load_image_as_df(image_file, image_idx=idx+1, use_only_engineered_features=True)
                 for idx, image_file in enumerate(image_files)]

    # Use one image as test set and the rest as training set
    train_imgs = [df for idx, df in enumerate(image_dfs) if idx != test_img-1]

    # Concatenate training images and create TabularDataset
    train_df = pd.concat(train_imgs)
    train_dataset = TabularDataset(train_df)

    # Fit the model
    predictor = TabularPredictor(label="label", groups="image", eval_metric="roc_auc", path=output_path)
    predictor.fit(train_dataset, presets="best_quality", auto_stack=True)
