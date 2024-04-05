
# Project Title

SEQUENTIAL COMBUSTION IMAGE CLASSIFICATION USING LATENT VECTORS FROM AUTOENCODER

## Installation
```bash
pip install -r requirements.txt
```


## USAGE

```bash
python main.py [path_to_your_config]
```
## TRAINING

 For Autoencoder`[your_config.ini]` file:
```ini
[Type]
type = autoencoder
[PostProcess]
post_process = False
```
For Sequential Model`[your_config.ini]` file:
```ini
[Type]
type = sequential
ae_path = "path_to_your_autoencoder/models"
[PostProcess]
post_process = False
```

## TESTING
 For Autoencoder`[your_config.ini]` file:
```ini
[Type]
type = autoencoder
[PostProcess]
post_process = True
```
For Sequential Model`[your_config.ini]` file:
```ini
[Type]
type = sequential
ae_path = "path_to_your_autoencoder/models"
[PostProcess]
post_process = True
```
## HYPERPARAMETER ANALYSIS DIRECT CLASSIFICATION
[DIRECT CLASSIFICATION](./notebooks/)

## Courtesy
Dr Soumik Sarkar