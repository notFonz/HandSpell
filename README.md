# Handspell

- [ToDo] Document your project here
- [ToDo] Description
- [ToDo] Data used
- [ToDo] Where your API can be accessed
- ...

# Usage
- [ToDo] Document main functionalities of the package here

# API
- [ToDo] Document main API endpoints here

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
