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




# API

Interactive API documentation:
- Google Cloud Run deployment [link](https://handspell-api-390650046032.europe-west1.run.app/docs)

- Prediction

      Method: POST
      Endpoint: /predict
      Local URL: http://localhost:8080/predict
      Purpose: Receives hand landmark coordinates and predicts the corresponding hand sign language letter.

      Method: POST
      Endpoint: /predict_and_save_temp
      Local URL: http://localhost:8080/predict_and_save_temp
      Purpose: Predicts the corresponding hand sign language letter from MediaPipe hand landmark coordinates using a trained XGBoost classifier and saves the prediction temporarily.

- Delete

      Method: DELETE
      Endpoint: /clear_temp
      Local URL: http://localhost:8080/clear_temp
      Purpose: Delete temporary files.

- Return Image

      Method: GET
      Endpoint: /sign/{letter}
      Local URL: http://localhost:8080/sign/{Letter}
      Purpose: Return image based on the given letter

# Usage

HandSpell provides the following core functionalities:

- Hand Detection: Detects hands in images using MediaPipe.

- Landmark Extraction: Extracts the coordinates of 21 hand landmarks to represent hand poses as numerical features.

- Sign Language Classification: Uses a trained XGBoost classifier to predict the corresponding sign language letter based on the extracted hand landmarks.

- API Integration: Provides access to prediction functionality through a REST API.


# Setup instructions

## Prerequisites

Make sure the following tools are installed:

Python (3.12.9)
Git (windows: sudo apt-get install git, macOS: brew install git)
uv, a Python package and project manager (pip install uv)

## 1. Get the project

Clone the repository and navigate to the project directory:

git clone notFonz/HandSpell
cd HandSpell

## 2. Install dependencies

Install the project dependencies using uv:

uv sync

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

On max to unrar a folder:
```bash
tar -xvf ASLYset.rar
```
