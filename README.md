# ✅ Adversarial Prompt Shield
This is the official repository for ["**Robust Safety Classifier Against Jailbreaking Attacks: Adversarial Prompt Shield**"](https://aclanthology.org/2024.woah-1.pdf) by Jinhwa Kim,  Ali Derakhshan, and Ian G. Harris.
We provide our generated dataset,  APS training codes, and APS trained models to allow everyone to access and replicate our works.

This paper is published at WOAH (Workshop on Online Abuse and Harms), 2024. 

## Citation
```
@inproceedings{kim-etal-2024-robust,
    title = "Robust Safety Classifier Against Jailbreaking Attacks: Adversarial Prompt Shield",
    author = "Kim, Jinhwa  and
      Derakhshan, Ali  and
      Harris, Ian",
    editor = {Chung, Yi-Ling  and
      Talat, Zeerak  and
      Nozza, Debora  and
      Plaza-del-Arco, Flor Miriam  and
      R{\"o}ttger, Paul  and
      Mostafazadeh Davani, Aida  and
      Calabrese, Agostina},
    booktitle = "Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.woah-1.12",
    pages = "159--170",
}
```
## Data Preparation 
We utilized various safety-classification benchmarks including red-team datasets and  an jailbreak attack to train and evaluate our models. The following is the list of datasets. To replicate our training, kindly download each dataset from its original repository and store them in separate directories.

<details>
  <summary>WTC (Wikipedia Toxic Comment) </summary>

  - Please download WTC Data from the [Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge).
  - Place the dataset on the `"data/wikipedia_toxic_comment/"` directory 
</details>
<details>
  <summary> BBF (Build-it, Break it, Fix-it) (Dinan et al., 2019) </summary>
  
  - Please download [BBF data](https://parl.ai/projects/dialogue_safety/) through ParlAI API
  - Please copy the `"data/dialogue safety"` folder (including both single_turn_safety.json, multi_turn_safety.json) on the "data/dialogue_safety" directory.
  - BBF consists of (single) "standard","adversarial", and "multi-turn" dialogue datasets
</details>

<details>
  <summary> BAD (Bot Adversarial Dialogue) (Xu et al., 2021) </summary>
  
  - Please download [BAD data](https://parl.ai/projects/safety_recipes/) through ParlAI API
  - Please copy the `"/bot_adversarial_dialogue_datasets_with_persona"` folder on the "/data/bot_adversarial_dialogue_datasets_with_persona" directory.
</details>

<details>
  <summary> ANTHROPIC Red-team-attempts data (Ganguli et al., 2022) </summary>
  
  - Please download ["red-team-attempts" dataset](https://github.com/anthropics/hh-rlhf), and convert it to .csv files with splitting them to training, valid, and test data
  - Place the dataset on the `"data/red-team/"` directory under the parent data path
</details>
<details>
  <summary> AdvBench (Zou et al., 2023) </summary>
  
  - AdvBench is available at [here](https://github.com/llm-attacks/llm-attacks).
  - Please follow that repository to generate GCG adversarial suffix for each harmful_behavior instance : APS Random, APS Pseudo

</details>


## 📌 BAND Datasets
We propose two different strategies to generate synthetic datasets that improve the robustness of our safety classifiers. 
- BAND Random Suffix Generation : Applying randomly selected 20 strings as a suffix to each instance. 
- BAND Pseudo Attack Suffix Generation : Applying suffix generated with semi-optimization algorithm which is computationally efficient.

Our Pseudo Attack dataset on AdvBench dataset is available at [Huggingface datasets](https://huggingface.co/datasets/jinhwak/APS_pseudo_attack). 
**Note this dataset is for research purpose only.**
```Python
from datasets import load_dataset
dataset = load_dataset('jinhwak/APS_pseudo_attack')
```

---
## 🛡️ APS Models
we trained several different APS-based classifiers with different training datasets, and provide our SOTA model, APS Pseudo.
You can get the model from [Huggingface](https://huggingface.co/jinhwak/APS_pseudo).

### Run APS Pseudo Demo

You can run a demo code for our model. To start, 
```Shell
python APSdemo.py --model_path 'jinhwak/APS_pseudo' --use_auth_token {your huggingface auth token}
```
You can change `model_path` to path of your local repository. 
It will be ready to get a text from the user, and provide prediction result (binary) from the model.

`1` indicates Unsafe, and `0` indicates Safe.  


### Train your own safety classifier 
You can train your own safety classifier with `APStrain.py`. To start,  
```Shell
## python APStrain.py {data names to be used to train} {data folder path} {path to save a trained model} {number of turns to use for preprocessing of multi-turn dialogue data}
python APStrain.py --training_dt wtc,bbf_s --datapath /mnt/llm/data/ --model_savepath /mnt/llm/models/ --multiturn 8
```

If you plan to train a model using your own data, pleas refer to  `utils/dataloader.py` for preprocessing. 
The dataset format should be a `DataFrame` containing columns such as "transcript," "rating," and "datatype," following the structure of the red-team-attempts dataset (Ganguli et al., 2022).

- "Transcript" contains text data (e.g. an utterance or a dialogue). Each instance is annotated with either "Human:" or "Assistant:".
- For dialogue data, we utilized a delimiter "\n\n" to separate each utterance.
