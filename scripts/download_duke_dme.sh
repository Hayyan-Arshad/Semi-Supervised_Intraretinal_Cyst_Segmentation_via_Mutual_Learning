#!/usr/bin/env bash
set -euo pipefail

mkdir -p datasets/duke_dme
cd datasets/duke_dme

curl -L -o Chiu_BOE_2014_dataset.zip \
  "https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.zip"

unzip -n Chiu_BOE_2014_dataset.zip

