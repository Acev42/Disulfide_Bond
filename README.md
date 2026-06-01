# Disulfide Bond Prediction
# Installation
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

# Usage

Use run.py Or disulfide.sh with the same arguments.

To avoid replication, here show the method of using sh script.

Run the prediction script:(at any conda env)

````
bash disulfide.sh [options]
````


# Arguments
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
| `--pdb`       | PDB ID (optional, for reference only)                       | None                                |
| `--chain`     | Chain ID (optional)                                         | None                                |

# Example
````
bash disulfide.sh --index 1234 --input ./1AHO.pdb --output ./result --threshold 0.6
````

