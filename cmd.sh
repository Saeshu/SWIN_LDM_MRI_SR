#!/bin/bash

apt update && apt install -y rclone
apt install -y unzip
curl https://rclone.org/install.sh | bash
