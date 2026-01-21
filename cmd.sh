#!/bin/bash

apt update && apt install -y rclone
apt install -y unzip
curl https://rclone.org/install.sh | bash

mkdir -p /workspace/dataset
rclone copy gdrive:IXI-T1 /workspace/dataset --progress
