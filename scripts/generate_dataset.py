import argparse
from pathlib import Path
from mlo.data.generator import generate_customer_dataset
from mlo.config import ROOT_DIR

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customer dataset.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed.")
    parser.add_argument("--output", type=str, default="data/raw/customer_churn_data.csv", help="Output path relative to root.")
    args = parser.parse_args()

    out_path = ROOT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples} customer samples with seed {args.seed}...")
    df = generate_customer_dataset(n_samples=args.samples, random_state=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved successfully to: {out_path} (Shape: {df.shape})")

if __name__ == "__main__":
    main()
