# Bolete Classifier

Bolete identification algorithm that utilizes modern deep learning techniques and domain knowledge about important mushroom characteristics.

## Paper

The preprint is available in PDF form in `Bolete_Species_and_Edibility_Classification.pdf` in the root of the project.

## Prerequisites

Codebase was created and has only been tested on Ubuntu 20.04.

Using a virtual env or conda environment, run 
```
pip install -r requirements.txt
```
This will install the python dependencies.

For the data preprocessing pipeline, install MATLAB R2020b.

## Pipeline

To run the pipeline from the raw dataset, which is unavailable to the public at this time:

```bash
cd data_pipeline/
sh run_pipeline.sh
```

This will:

1. Get the list of possible categories of mushroom for the system to predict from the tabular data files.
2. Collect valid labels for the data by cross referencing which mushrooms have (or do not have) those categories in the tabular data.
3. Copy the images from the original photos directory by the name that the photos are titled, for example ``Allesioporus-rubriflavus-Jane-Smith.jpeg'' would be copied to the folder ``allesioporus-rubriflavus''.
4. Copy the images referenced by the tabular data that correspond to our mushroom labels into the appropriate folder and copy non-duplicates from the folder created from filenames directly in step 3.
Now `clean_data/labels.csv` contains all the mushrooms and the mushroom identification features to predict. And the `data` folder contains photos for each mushroom by the newest name of the mushroom for data-augmentation purposes (add correct photos to these bins as needed).
5. Process all image files in `data` by resizing and padding them so they are 512 x 512. Write these compressed versions to an HDF5 dataset file with their corresponding labels.

## Building

To build the Cython files execute

```bash
python setup.py build_ext --inplace
```

## Running Models

The direct model variants can be run from the `direct.ipynb` file and the characteristic model variants can be run using the `characteristic.ipynb`.

Options:
* FINAL = True if you want to produce evaluation results on the test set, False otherwise
* EDIBILITY = True if the direct model should output edibility scores, False for species scores
* HIGH_RES = True to use 512x512 images, False to use 256x256 images.

## Evaluating Results

Running the cells in `performance.ipynb` will produce the performance data from the data in `evaluation_data/`. Output will be located in `performance_data/`

To generate all the plots used in the paper, execute:

```matlab
>> paired_history_plots
>> p_charts
```

## Acknowledgements

Data generously provided by the [Western Pennsylvania Mushroom Club](https://wpamushroomclub.org/) via collaboration with Scott Pavelle and Richard Jacob.
