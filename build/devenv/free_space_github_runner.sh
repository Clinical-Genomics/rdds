#!/bin/bash
set -e
set -x

echo "Listing 100 largest packages"
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 100
df -h

echo "Freeing up space on runner ..."
sudo apt-get purge -y '^microsoft-edge-stable.*'
sudo apt-get purge -y '^azure-cli.*'
sudo apt-get purge -y '^google-chrome.*'
sudo apt-get purge -y '^google-cloud-cli.*'
sudo apt-get purge -y '^temurin-.*'
sudo apt-get purge -y '^firefox.*'
sudo apt-get purge -y '^powershell.*'
sudo apt-get purge -y '^snapd.*'
sudo apt-get purge -y '^mysql-server.*'
sudo apt-get purge -y '^kubectl.*'
sudo apt-get purge -y '^postgresql.*'
sudo apt-get purge -y '^podman.*'
sudo apt-get purge -y '^llvm.*'
sudo apt-get autoremove -y
sudo apt-get clean
df -h