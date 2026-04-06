#!/bin/bash

echo "Starting training with config_1"
python document_and_instructions.py --config config_1

echo "Finished training with config_1"

echo "Starting training with config_2"
python document_and_instructions.py --config config_2

echo "Finished training with config_2"

echo "Starting preference evaluation on trained models"
python eval/preference_misalignment/preference_eval.py

echo "Finished preference evaluation"