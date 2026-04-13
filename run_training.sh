#!/bin/bash

for i in {7..15}; do
    echo "Starting training with config_$i"
    python document_and_instructions.py --config config_$i
    echo "Finished training with config_$i"
done

echo "Starting preference evaluation on trained models"
python eval/preference_misalignment/preference_eval.py

echo "Finished preference evaluation"