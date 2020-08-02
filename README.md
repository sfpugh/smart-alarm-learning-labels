# smart-alarm-learning-labels
The Smart Alarm 2.0 project aims to suppress false, invalid SpO2 low alarms emitted by pulse oximetry machines. Large SpO2 alarm datasets with labels of whether an alarm should be suppressed or not do not exist and are expensive to create. Hence we are using Snorkel to programmatically label datasets using several rules based on domain knowledge.

## Installation
Clone this repository.
```
git clone https://github.com/sfpugh/smart-alarm-learning-labels.git
```

Create a virtual environment using Python3 virtualenv and add it to Jupyter Notebook.
```
virtualenv <env_name>
source <env_name>/bin/activate
pip install -r requirements.txt

python -m ipykernel install --user --name=<env_name>

deactivate
```

## Usage
Unzip the data to avoid having to re-build it from scratch.
```
unzip data.zip
```

Open Jupyter notebook and open the .ipynb file of interest. Once you open a notebook, activate the virtual environment created in the installation steps by navigating to the 'Kernel' tab in the toolbar, then hover on 'Change kernel', then select the proper environment. Now you can run all the cells in the notebook. 
```
jupyter notebook
```

## Resources
- Snorkel https://www.snorkel.org 
- Matrix profile https://github.com/target/matrixprofile-ts
