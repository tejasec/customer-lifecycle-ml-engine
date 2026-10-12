import argparse
from mlo.pipeline.lifecycle import ProductionPipeline

def main():
    parser = argparse.ArgumentParser(description="Execute the 9-Step Production Lifecycle.")
    parser.add_argument("--data", type=str, default=None, help="Optional raw dataset path.")
    args = parser.parse_args()

    print("=" * 80)
    print("EXECUTING 9-STEP PRODUCTION LIFECYCLE (MLO)")
    print("=" * 80)
    pipeline = ProductionPipeline()
    report = pipeline.run(raw_data_path=args.data)
    print("\nPipeline Execution Successfully Completed.")
    print("Summary of Evaluation Report:")
    print("LTV Regression MSE:", report["regression_metrics"]["mse"])
    print("LTV Regression R2:", report["regression_metrics"]["r2"])
    print("Churn Classification Accuracy:", report["classification_metrics"]["accuracy"])
    print("Churn Classification F1:", report["classification_metrics"]["f1"])

if __name__ == "__main__":
    main()
