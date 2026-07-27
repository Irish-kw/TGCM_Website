# Table XXII — unknown-K inference cost

Run `reproduce.ipynb`. Each TGCM forward is decoded at the known-K channel budget and at the default Up-to-6 budget. The notebook reports both values and Up-to-6 minus Known-K deltas without retraining or using a second model.

The optional `DOWNLOAD_FULL_CALDERA_CSV` cell verifies the complete source CSV snapshot but is not required for checkpoint-based timing and metric reproduction.
