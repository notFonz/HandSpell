# HandSpell - Reading hands, one letter at a time

American Sign Language (ASL) Package and API.
Provides:
- Links to public ASL datasets
- Python package to explore, process and train models based on those datasets
- API to assess the ASL letter from an image

![By Coloringbuddymike - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=24802810](data/.img/Asl-sign-language-coloring-at-coloring-pages-for-kids-boys-dotcom.svg)


## Why?
Around 70 million deaf people use sign language, yet most hearing people can't
read a single sign.
We want to build a bridge, one letter at a time.

## Who?

Originally an educational group project to practice Data Science & AI skills.

Project [presentation](
https://docs.google.com/presentation/d/1Xz8Q6KtlDj0mV3MGL5ygtARrU_AmtbRg9EsATKX838w/edit?usp=sharing)


Original team members:
- Afonso Silva - [LinkedIn](https://www.linkedin.com/in/notfonz/)
- Delphine Pessoa - [LinkedIn](https://www.linkedin.com/in/delphine-pessoa/)
- Maximilian Greger - [LinkedIn](https://www.linkedin.com/in/maximilian-greger-77595b289/)
- Kazim Hussain Mohammed - [LinkedIn](https://www.linkedin.com/in/kazim-mohammed-a7316856/)

Pitched project [presentation](
https://docs.google.com/presentation/d/1jcQMq_tmHvUgpGdeSIOcO710AYkrgIvgbfjN2BN1O7g/edit?slide=id.p1#slide=id.p1) by Afonso Silva in October 2026.

See "Contributions section" for individual contributions.

# API

Interactive API documentation:
- Google Cloud Run deployment [link](https://handspell-api-390650046032.europe-west1.run.app/docs)

- Prediction

      Method: POST
      Endpoint: /predict
      Local URL: http://localhost:8080/predict
      Purpose: Receives image, detects hand landmark coordinates and predicts the corresponding hand sign language letter.

- Return Image

      Method: GET
      Endpoint: /sign/{letter}
      Local URL: http://localhost:8080/sign/{Letter}
      Purpose: Receives letter, returns a fingerspelling image of that letter

# Usage

HandSpell provides the following core functionalities:

- Hand Detection: Detects hands in images using MediaPipe.

- Landmark Extraction: Extracts the coordinates of 21 hand landmarks to represent hand poses as numerical features.

- Sign Language Classification: Uses a trained XGBoost classifier to predict the corresponding sign language letter based on the extracted hand landmarks.

- API Integration: Provides access to prediction functionality through a REST API.


# Setup instructions

## Prerequisites

Make sure the following tools are installed:

- **Git**
  ```bash
  # Windows
  winget install --id Git.Git -e --source winget
  # Linux
  sudo apt-get install git
  # MacOS
  brew install git
  ```
- **uv**: a Python package and project manager
  ```bash
  pip install uv
  ```
- **Python** `3.12` or higher - tested with `3.12`
  ```bash
  uv python install 3.12
  ```

## 1. Get the project

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/notFonz/HandSpell.git
cd HandSpell
```

## 2. Install dependencies

Install the project dependencies using uv:

```bash
uv sync
```

## Dockerization

### Local

```
make docker_build_local
make docker_run_local
```

if `ModuleNotFoundError: No module named 'cv2'`, run `uv sync --reinstall-package opencv-python-headless`.

To clean docker space, run

```
make docker_clean
```

### Cloud

Set `GCP_PROJECT` in your local `.env` file.

```bash
# Build a docker image compatible with cloud environment
make docker_show_image_path
make docker_build

# Configure gcloud
make gcloud_login
make gcloud-set-project
## In case you don't have free trial credits anymore
# make gcloud_activate_service_account

# Create Docker backbone on gcloud
make docker_allow
make docker_create_repo

# Actually make it available at your URL
make docker_push
make docker_deploy
```

Upon changes, test locally outside of a docker image

```bash
make run_api
```

And locally inside a docker image

```bash
uv sync --no-dev
make docker_build_local
make docker_run_local
```

In case of issues, you can edit `debug_docker.py` and run the following to
reproduce errros and try fixes:
```bash
docker run --rm -v "$(pwd):/work" -w /app --entrypoint .venv/bin/python \
  handspell-api:local /work/debug_docker.py data/mapping/ASL\ Letter\ Reference/A.jpg
```

then deploy
```bash
uv sync --no-dev
make docker_build
make docker_push
make docker_deploy
```


# Data download

See each dataset's `description.md` for instructions to download each dataset.


### Contributions

Two very different pipelines were explored during the project weeks:

- **v1** = YOLO hand detection + cropping, then a CNN on 28×28 greyscale images (Sign Language MNIST); did not generalize well on pictures we took with our webcams
- **v2** = MediaPipe hand landmarks + XGBoost, trained on a merged multi-source dataset ("megadataset"); used for demo day - On held-out users (not seen in training): hands detected in 100% of images, and ~80% of detected signs classified correctly.

Contributions:
- **Afonso Silva**
  - Project author; repository setup, Makefile and README boilerplate
  - Dataset research, cleaning (`data_cleaning.py`) and merging into the megadataset (`merge_asl_datasets.py`)
  - v1 CNN classifier on Sign Language MNIST exploration (Kaggle and local notebook)
  - v1 CNN refactoring Kazim's solution into handspell package (`model.py`: build, train, save/load, predict)
  - v2 MediaPipe hand detection and MediaPipe + XGBoost working in local notebook
  - Text-to-sign feature: `text_to_image.py` and the `/sign/{letter}` API endpoint
  - Code review

- **Delphine Pessoa**
  - Project management (Trello); project structure; dataset naming convention and documentation
  - v1 YOLO hand detection and 28×28 preprocessing notebook
  - v1 API refactored to use the `handspell` package
  - v2 refactoring Colab notebook into `handspell` package (landmark detection `model_landmarks.py`, training and evaluation - `model_landmarks_training.py`,  `dataset.py`)
  - v2 API with MediaPipe + XGBoost; Docker image and Cloud Run deployment
  - Code review

- **Maximilian Greger**
  - v1 preprocessing pipeline: YOLO hand detection and cropping (`data_preprocessing.py`)
  - v2 training, XGBoost hyperparameter tuning and evaluation on megadataset
  - README

- **Kazim Hussain Mohammed**
  - Dataset research
  - v1 CNN classifier notebook and evaluation (Colab notebook)
  - v1 API with `/predict` endpoint; first Dockerfile and deployment configuration
  - v2 first MediaPipe + XGBoost notebook with model evaluation trained on one dataset (Colab)
  - Explored model for videos with MediaPipe

## License

- **Code** (`src/`, `api/`, `notebooks/`, `Dockerfile`, `Makefile`): [GPL-3.0](LICENSE)
- **Data, images and documentation** (`data/`, `img/`, `*.md`): [CC BY-SA 4.0](LICENSE-DATA)

Third-party datasets in `data/raw_data/` keep their original licenses; see each `description.md`.
