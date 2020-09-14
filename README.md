# smart-alarm-learning-labels
The Smart Alarm 2.0 project aims to suppress false, invalid SpO2 low alarms emitted by pulse oximetry machines. Large SpO2 alarm datasets with labels of whether an alarm should be suppressed or not do not exist and are expensive to create. Hence we are using Snorkel to programmatically label datasets using several rules based on domain knowledge.

## Installation
Clone this repository.
```
git clone https://github.com/sfpugh/smart-alarm-learning-labels.git
```

Create a virtual environment using Python3 virtualenv and add it to Jupyter Notebook per example. Use the following instructions, replacing `<example_dir>` with the name of the directory the example is in and `<example_env_name>` with a name for the virtual environment of that example.
```
pip3 install --upgrade virtualenv
virtualenv -p python3 <example_env_name>
source <example_env_name>/bin/activate

pip install -r <example_dir>/requirements.txt

python -m ipykernel install --user --name=<example_env_name>

deactivate
```

## Usage
For each example, there exists a jupyter notebook `*_STABLE.ipynb` in the example's directory. You can open the notebook with the following command.
```
jupyter notebook <blah>_STABLE.ipynb
```

Then activate the virtual environment created in the installation step by navigating to the 'Kernel' tab in the toolbar, then hover on 'Change kernel', then select the proper environment.

To run experiments in the `exp` directory, append cells to the end of the notebook that invoke the experiment script like the following. 
```
%run -i ../exp/<E#-description>.ipynb <params_if_necessary>
```

## Notes
- To avoid re-building the smart alarm data (which is time consuming), unzip the data in directory `smart_alarm/data`.
```
cd smart_alarm/data/
unzip data.zip
```


## Resources
- Snorkel https://www.snorkel.org 
- Matrix profile https://github.com/target/matrixprofile-ts
