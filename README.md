# Bolete Classifier

Bolete identification algorithm that utilizes modern deep learning techniques and domain knowledge about important mushroom characteristics.

## Pipeline

To run the pipeline from the raw dataset:

```bash
sh run_pipeline.sh
```

This will:

1. Get the list of possible categories of mushroom for the system to predict from the tabular data files.
2. Collect valid labels for the data by cross referencing which mushrooms have (or do not have) those categories in the tabular data.
3. Copy the images from the original photos directory by the name that the photos are titled, for example ``Allesioporus-rubriflavus-Jane-Smith.jpeg'' would be copied to the folder ``allesioporus-rubriflavus''.
4. Copy the images referenced by the tabular data that correspond to our mushroom labels into the appropriate folder and copy non-duplicates from the folder created from filenames directly in step 3.
Now `clean_data/labels.csv` contains all the mushrooms and the mushroom identification features to predict. And the `data` folder contains photos for each mushroom by the newest name of the mushroom for data-augmentation purposes (add correct photos to these bins as needed).
5. Process all image files in `data` by resizing and padding them so they are 512 x 512. Write these compressed versions to an HDF5 dataset file with their corresponding labels.

## Acknowledgements

Data generously provided by the [Western Pennsylvania Mushroom Club](https://wpamushroomclub.org/) via collaboration with Scott Pavelle and Richard Jacob.
