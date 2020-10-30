# Bolete Classifier

Bolete identification algorithm that utilizes modern deep learning techniques and domain knowledge about important mushroom characteristics.

## Pipeline

To run the pipeline from the raw dataset:

1. Get the list of valid features for the system to predict

```matlab
>> map_features
```

2. Collect valid labels for the data using those features

```matlab
>> label_mushrooms
```

3. Copy the images from the photos directory by the name that the photos are titled

```matlab
>> copy_images_from_files
```

4. Copy the images that correspond to our mushroom labels into the appropriate folder and copy non-duplicates from what was 

```matlab
>> copy_images_from_csv
```

Now, `clean_data/labels.csv` contains all the mushrooms and the mushroom identification features to predict. And the `data` folder contains photos for each mushroom by the newest name of the mushroom for data-augmentation purposes (add correct photos to these bins as needed).

## Acknowledgements

Data generously provided by the [Western Pennsylvania Mushroom Club](https://wpamushroomclub.org/) via collaboration with Scott Pavelle.
