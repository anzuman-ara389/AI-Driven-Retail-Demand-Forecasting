@echo off

cd /d "C:\Users\Fusion\OneDrive\Desktop\AI-Driven Retail Demand Forecasting and Intelligent Inventory Decision Support for Food Waste Reduction"

call conda activate base

python -m src.external_client --mode normal --events 100 --delay 1

pause