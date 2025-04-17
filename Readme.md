This code builds on the codebase from "Reward Model Ensembles Help Mitigate Overoptimization." It provides setup and scripts for:

1. Supervised fine-tuning (SFT)
2. Reward model (RM) training
3. Policy optimization

Pretrained SFT Models
For experiments involving Pythia models, you can use the following pretrained SFT checkpoints:

Pythia 70M SFT on AlpacaFarm:
https://huggingface.co/tlc4418/pythia_70m_sft

Pythia 1.4B SFT on AlpacaFarm:
https://huggingface.co/tlc4418/pythia_1.4b_sft_policy

Reward Model (RM) Training:
You can train the reward model using the following datasets:
1. AlpacaFarm preference data
2. Preferences collected from the 1.4B SFT policy on AlpacaFarm instructions
3. Custom training data
Refer to the dataset preparation guide here:
Dataset Guide (Coste et al.) and cqan filterd usingg filterdata.py 

EBM Model Training
Once RM data is prepared, you can train the Energy-Based Model (EBM) using:
python -m src.reward_modeling.ebm_training.ebm_nce_plus.py
