# HandSpell - Reading hands, one letter at a time

American Sign Language (ASL) Package and API.
Provides:
- Links to public ASL datasets
- Python package to explore, process and train models based on those datasets
- API to assess the ASL letter from an image

![ASL schematic by Ds13 at English Wikipedia - Transferred from en.wikipedia to Commons., Public Domain, https://commons.wikimedia.org/w/index.php?curid=79729279](img/Asl_alphabet_gallaudet.svg)

## Why?
Around 70 million deaf people use sign language, yet most hearing people can't
read a single sign.
We want to build a bridge, one letter at a time.

## Who?

Originally an educational group project to practice Data Science & AI skills.

Pitched project [presentation](
https://docs.google.com/presentation/d/1jcQMq_tmHvUgpGdeSIOcO710AYkrgIvgbfjN2BN1O7g/edit?slide=id.p1#slide=id.p1) by Afonso Silva in October 2026.

Original team members:
- Afonso Silva - [LinkedIn]
- Delphine Pessoa - [LinkedIn](https://www.linkedin.com/in/delphine-pessoa/)
- Maximilian Greger - [LinkedIn](https://www.linkedin.com/in/maximilian-greger-77595b289/)
- Kazim Hussain Mohammed - [LinkedIn]


# API

Interactive API documentation:
- Kazim deployment [link](https://handspell-api-77400974703.europe-west1.run.app/docs)
- Delphine deployment [temporary_link](https://handspell-api-390650046032.europe-west1.run.app/docs)

- [TODO] Document main API endpoints here


# Usage
- [TODO] Document main functionalities of the package here


# Setup instructions
- [ToDo] Document here for users who want to setup the package locally

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
