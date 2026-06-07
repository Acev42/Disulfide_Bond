# Disulfide Bond Prediction
## Note of Attention
-The `src` directory contains the model weights(not uploaded yet), model architecture, and auxiliary scripts. Both run.py and disulfide.sh are functional. For proper execution of ESM, the scripts must be run from within the main directory.

-The `README.md` file provides detailed instructions on how to use the tool.

-The `log.md` file explains the architecture and workflow of this model.

-The `requirements.txt` file has been tested and verified to work correctly.

-The `data/weights` directory is intended to contain the pretrained ESM-C model weight files.



## Installation
Step 1: Create conda environment

```` 
conda create -n disulfide python=3.12
````

Step 2: Activate the environment
````
conda activate disulfide
````
Step 3: Install dependencies
````
pip install -r requirements.txt
````

## Usage

Use run.py or disulfide.sh with the same arguments.

To avoid replication, here show the method of using sh script.

Run the prediction script:(at any conda env)

````
bash disulfide.sh [options]
````


## Arguments
| Argument      | Description                                                 | Default                             |
| ------------- | ----------------------------------------------------------- |-------------------------------------|
| `--index`     | Index of the input file for prediction                      | `None`                               |
| `--input`     | Path to the input PDB/CIF file                              | **Required**                        |
| `--output`    | Path to the output CSV result file                          | `./output`                          |
| `--save`      | Whether to save intermediate results                        | `False`                             |
| `--ignore`    | Whether to ignore existing disulfide bonds in the structure | `True`                              |
| `--model`     | Path to the pretrained model checkpoint                     | `./src/model/Disulfide_searching.ckpt` |
| `--threshold` | Probability threshold for disulfide bond prediction         | `0.5`                               |
| `--filetype`  | Input structure format: `PDB` or `CIF`                      | `PDB`                               |
| `--pdb`       | PDB ID (optional, for reference only)                       | `None`                                |
| `--chain`     | Chain ID (optional)                                         | `None`                                |

## Example
````
bash disulfide.sh --index 1234 --input ./1AHO.pdb --output ./result --threshold 0.6
````
output:1234_1AHO_results.csv
| residue1 id | residue1 name | residue2 id | residue2 name | probability | pdb_id | chain |
|-------------|---------------|-------------|---------------|-------------|--------|-------|
| 31          | GLY           | 48          | CYS           | 0.7384205880050742 | 1AHO | A |
| 34          | GLY           | 48          | CYS           | 0.5725554322907841 | 1AHO | A |

## Model performance

After setting the positive sample weight for the cross-entropy function (determined by the ratio of the number of negative samples to positive samples), the model's performance after one epoch of training is as follows:

-Accuracy: 99.0%

-False Positive Rate: 0.86%

-False Negative Rate: 15.2%

-AUC-ROC: 0.991
