This code builds on the codebase from "[Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/abs/2310.02743v2)" It provides setup and scripts for:

1. Supervised fine-tuning (SFT)
2. Reward model (RM) training
3. Policy optimization

Pretrained SFT Models
For experiments involving Pythia models, you can use the following pretrained SFT checkpoints:

Pythia 70M SFT on AlpacaFarm:
https://huggingface.co/tlc4418/pythia_70m_sft

Pythia 1.4B SFT on AlpacaFarm:
https://huggingface.co/tlc4418/pythia_1.4b_sft_policy

For experiments on 44M, 1.3B RM we can use the given SFT's to train the RMs, for experiments on 2.6B RM you need to train the SFT and then use it for RM training.


Reward Model (RM) Training:
You can train the reward model using the following datasets:
1. AlpacaFarm preference data
2. Preferences collected from the 1.4B SFT policy on AlpacaFarm instructions
3. Custom training data
Refer to the dataset preparation guide here:
Dataset Guide (Coste et al.) and can be filterd using filterdata.py 

For experiments on skywork we feed the training data [Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) in the [Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) RM and save the final embedding and rewards, and then normailze the rewards before training the EBRM.

EBM Model Training
Once RM data is prepared, you can train the Energy-Based Model (EBM) using:
python -m src.reward_modeling.ebm_training.ebm_nce_plus.py
