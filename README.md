# Species Distribution Modeling for UK Wildlife

This project is dedicated to predicting the geographical distribution of various wildlife species, within the United Kingdom, using machine learning. It employs species occurrence data and environmental variables to train models that predict suitable habitats.

The primary output appears to be probability maps that visualize the likelihood of a species being present in a given area.

## Project Workflow

The project seems to follow a standard data science workflow for species distribution modeling:

1.  **Data Gathering & Preprocessing**: Raw species data (in `.csv` files) is collected and cleaned. Environmental data from raster files (`.tif`) is extracted and merged with the species data. The `preprocessing.ipynb` notebook contains the code for this.
2.  **Model Training**: Machine learning models are trained on the preprocessed data. Based on the file names (`.pth` files), the project uses PyTorch and explores different loss functions, such as `BCEWithLogitsLoss` and a custom `deepmaxent_loss`. The `deep_maxent.ipynb` and `MaxEnt.ipynb` notebooks are central to this stage.
3.  **Inference & Prediction**: The trained models are used to predict habitat suitability across the entire study area. This is handled by the species-specific inference notebooks (e.g., `bat_inference.ipynb`).
4.  **Evaluation & Visualization**: Models are evaluated using metrics like ROC AUC and the Boyce Index. The final predictions are saved as probability maps (`.png` files) for visualization.

## File Structure Overview

The repository is organized primarily by species, with a collection of scripts and notebooks that form the modeling pipeline.

-   `*.ipynb`: Jupyter Notebooks for various stages of the project, from initial data exploration (`TGB_*.ipynb`) to preprocessing, modeling (`deep_maxent.ipynb`), and inference.
-   `[species_name]_*.csv`: A variety of CSV files per species, including:
    -   `_full_data.csv`: The complete, raw dataset.
    -   `_final_data.csv`: Cleaned data ready for preprocessing.
    -   `_preprocessed.csv`: Data ready for model training.
    -   `_results.csv`: Stores the evaluation metrics for trained models.
-   `*.pth`: Saved PyTorch model weights for different species, loss functions, and evaluation criteria (e.g., `bat_BCEWithLogitsLoss_roc_best_model.pth`).
-   `*.pkl`: Saved Python objects, such as data scalers (`scaler.pkl`), train/test splits (`split.pkl`), and other model components.
-   `*.tif`: GeoTIFF raster files (`gblcm2023_10m.tif`, `nilcm2023_10m.tif`), which are environmental data layers (e.g., land cover, climate data) used as predictors in the models.
-   `data/`: Contains geospatial vector data (shapefiles) that define the study area boundaries (e.g., UK country borders).
-   `*probability_map.png`: Output images visualizing the predicted habitat suitability for each species.

## Species Studied

The following species are the focus of this study:

-   Bat (`bat_` / `Pipistrellus pygmaeus`)
-   Brown Hare (`brown_hare_` / `Lepus europaeus`)
-   Dormice (`dormice_` / `Muscardinus avellanarius`)
-   Hedgehog (`hedgehog_` / `Erinaceus europaeus`)
-   Red Squirrel (`red_squirrel_` / `Sciurus vulgaris`)

Note: you can find the gblcm2023_10m.tif file on https://catalogue.ceh.ac.uk/documents/7727ce7d-531e-4d77-b756-5cc59ff016bd