#!/usr/bin/env bash
set -e
cd ~/hugo_blog
hugo
sudo rm -rf /var/www/lanshi.xyz/*
sudo cp -r public/* /var/www/lanshi.xyz/
echo "Blog updated successfully!"
