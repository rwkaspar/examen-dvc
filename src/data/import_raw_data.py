import os
import argparse
import requests

def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()  # fail if download failed
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv")
    parser.add_argument("--output", type=str, default="data/raw_data/raw.csv")
    args = parser.parse_args()

    download_file(args.url, args.output)
