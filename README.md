# HumT DumT
This repository contains code to compute HumT and SocioT, metrics of human-like tone and social perceptions in text. It also contains code to fine-tune DumT, a model that is steered such that the outputs are less human-like. These methods are introduced in the following paper:

## HumT DumT:  Measuring and controlling human-like language in LLMs
*Myra Cheng, Sunny Yu, Dan Jurafsky* (Stanford University)
### Abstract:
Should LLMs generate language that makes them seem human? Human-like language might improve user experience, but might also lead to overreliance and stereotyping. Assessing these potential impacts requires a systematic way to measure human-like tone in LLM outputs. We introduce HumT and SocioT, metrics for human-like tone and other dimensions of social perceptions in text data based on relative probabilities from an LLM. By measuring HumT across preference and usage datasets, we find that users prefer less human-like outputs from LLMs. HumT also offers insights into the impacts of anthropomorphism: human-like LLM outputs are highly correlated with warmth, social closeness, femininity, and low status, which are closely linked to the aforementioned harms. We show that HumT can be used to systematically control and reduce the degree of human-like tone in LLM outputs while preserving model performance, offering a practical tool for mitigating risks associated with anthropomorphic language generation.

# Setup
Clone the repository: ```git clone https://github.com/myracheng/humtdumt.git;```

HumT, SocioT and DumT require the following packages:
  `numpy pandas scipy torch transfomers`

Furthermore, DumT requires `tqdm`, `trl`, `wandb`, `peft`, and `datasets`.

The specific versions of packages that we use are:
    ```
    datasets==3.2.0 
    numpy==2.2.3
    pandas==2.2.3
    peft==0.10.0
    scipy==1.15.1
    torch==2.6.0
    tqdm==4.67.1
    transformers==4.43.4
    trl==0.8.7.dev0
    wandb==0.16.6
    ```
# HumT and SocioT

To compute HumT and/or SocioT on a CSV file, run the following command in your command line:

`python compute_humt_sociot.py FILENAME --input_column COLUMN --metric METRIC1 METRIC2...` 

`FILENAME` is the CSV containing with column `COLUMN` on which to run `METRIC 1`, `METRIC 2`, etc.  `METRIC1 METRIC2....` must be one or more of the following: 
    `humt sociot_status sociot_social_distance sociot_gender sociot_warmth`

The output will be saved to a new file (whose path you can also specify using `--save_path`) with 2 columns for each metric: the score in `METRIC_COLUMN` and its standard deviation over `100` runs in `std_METRIC_COLUMN`.

See the paper for descriptions of each metric. We recommend running the code on GPU for larger datasets as the computation may be prohibitively slow otherwise.[^1] We ran this on a NVIDIA RTX 600 (48 cores, 516GB RAM) using 1 GPU and 16GB memory. 


[^1]:You could also change the number of times that the computation is performed, i.e., averaging each score over fewer samples, by modifying the parameter `n_samples=100` on Line 40. However we haven't tested the quality of the results with fewer samples.


## Example usage
To compute HumT on the column `text` in `examples.csv`:

`python compute_humt_sociot.py examples.csv --input_column text --metric humt`

To compute HumT and SocioT on the column `text` in `examples.csv` and then save it to `all_metrics.csv`:

`python compute_humt_sociot.py examples.csv --input_column text --metric humt sociot_status sociot_social_distance sociot_gender sociot_warmth --save_path all_metrics.csv`

# DumT
The `DumT` folder contains code for fine-tuning DumT. The code that we used to construct the preference dataset and evaluation dataset for DumT in `DumT/construct_train_test_dataset.ipynb`. The code to fine-tune the model and then run inference on the evaluation set is `DumT/DPO.py`. It can be run as follows:

`python DPO.py preference_data_filename model_name evaluation_save_filepath`

We ran this on a NVIDIA A100-SXM4-80GB (127 cores, 1032GB RAM) using 1 GPU and 200 GB memory. 

## Example usage
For instance, to fine-tune the model on `dumt_training_data.json` (the prefrence data for the model we present in the paper):
`python DPO.py dumt_training_data.json dumt_model dumt_test_output`


# Contact
Please contact me if you have any questions or concerns! myra [at] cs [dot] stanford [dot] edu
