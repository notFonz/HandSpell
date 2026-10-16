# Project Name
- Document your project here
- Description
- Data used
- Where your API can be accessed
- ...

# API
Document main API endpoints here

# Setup instructions
Document here for users who want to setup the package locally

## Dockerization

Set `GCP_PROJECT` in your local `.env` file.

### Local

```
make gcloud-set-project
make docker_build_local
make docker_run_local
```

if `ModuleNotFoundError: No module named 'cv2'`, run `uv sync --reinstall-package opencv-python-headless`.

To clean docker space, run

```
make docker_clean
```

### Cloud

```bash
make docker_show_image_path
make docker_build

make gcloud_login
## In case you don't have free trial credits anymore
# make gcloud_activate_service_account
make docker_allow
make docker_create_repo
make docker_push
make docker_deploy
```

# Usage
Document main functionalities of the package here
