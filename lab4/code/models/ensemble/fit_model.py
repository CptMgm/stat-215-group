import argparse
from autogluon.tabular import TabularDataset, TabularPredictor


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_only_engineered_features", action="store_true")
    args = parser.parse_args()
    only_engineered_features = args.use_only_engineered_features

    # Set output path
    if only_engineered_features:
        output_path = "AutogluonModels/engineered_features"
    else:
        output_path = "AutogluonModels/all_features"

    # Load preprocessed data
    data = TabularDataset("../../../data/preprocessed/image1_2.csv")

    # Fit the model
    predictor = TabularPredictor(label="label", groups="image", eval_metric="roc_auc", path=output_path)
    predictor.fit(data, presets="best_quality", auto_stack=True)
