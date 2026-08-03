# Table III — kill-chain coverage

Run `reproduce.ipynb` to reproduce the main-paper Table III exactly. The notebook reports the number of unique techniques mapped to each ATT&CK tactic for ATLAS, NODLINK, ProvCon, DARPA TC-E3, DARPA TC-E5, and CAPTure.

The output columns match the manuscript: `Dataset`, `Init. Access`, `Execution`, `Persistence`, `Def. Evasion`, `Cred. Access`, `Discovery`, `Lat. Move.`, `Collection`, `C2`, and `Exfiltration`.

The five zero-shot rows are derived from `paper_metadata/kill_chain_mapping.json`. CAPTure coverage is loaded from `paper_metadata/capture_kill_chain_coverage.json`, which summarizes the profile-level mapping reported in the appendix. The notebook asserts exact equality with the Overleaf Table III values before writing `table03_kill_chain_coverage.csv`.
