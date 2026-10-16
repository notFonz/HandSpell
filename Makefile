#======================#
# Install, clean, test #
#======================#

install_requirements:
	@pip install -r requirements.txt

install:
	@pip install . -U

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info

test_structure:
	@bash tests/test_structure.sh

#======================#
#          API         #
#======================#

include .env
export

run_api:
	uvicorn api.main:app --reload --port $(DOCKER_LOCAL_PORT)


#======================#
#          GCP         #
#======================#

gcloud-set-project:
	gcloud config set project $(GCP_PROJECT)

#======================#
#         Docker       #
#======================#

# Local images - using local computer's architecture
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local .

docker_run_local:
	docker run \
		-e PORT=$(DOCKER_LOCAL_PORT) -p $(DOCKER_LOCAL_PORT):$(DOCKER_LOCAL_PORT) \
		$(DOCKER_IMAGE_NAME):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=$(DOCKER_LOCAL_PORT) -p $(DOCKER_LOCAL_PORT):$(DOCKER_LOCAL_PORT) \
		$(DOCKER_IMAGE_NAME):local \
		bash

docker_clean:
	docker system df
	docker builder prune -af
	docker system prune -af

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

DOCKER_IMAGE_PATH := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME)

docker_show_image_path:
	@echo $(DOCKER_IMAGE_PATH)

docker_build:
	docker build \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

# Alternative if previous doesn´t work. Needs additional setup.
# Probably don´t need this. Used to build arm on linux amd64
docker_build_alternative:
	docker buildx build --load \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

docker_run:
	docker run \
		--platform linux/amd64 \
		-e PORT=$(DOCKER_LOCAL_PORT) -p $(DOCKER_LOCAL_PORT):$(DOCKER_LOCAL_PORT) \
		$(DOCKER_IMAGE_PATH):prod

docker_run_interactively:
	docker run -it \
		--platform linux/amd64 \
		-e PORT=$(DOCKER_LOCAL_PORT) -p $(DOCKER_LOCAL_PORT):$(DOCKER_LOCAL_PORT) \
		$(DOCKER_IMAGE_PATH):prod \
		bash

# Push and deploy to cloud

# Personal authentication
gcloud_login:
	gcloud auth login

# Service account authentication;
# GOOGLE_APPLICATION_CREDENTIALS (the path to a json file) need to be defined
# in ~/.zshrc or equivalent
# And the following needs to have been done previously:
# gcloud iam service-accounts keys create \
  > $GOOGLE_APPLICATION_CREDENTIALS \
  --iam-account=name@your-project-id.iam.gserviceaccount.com
# where the name@your-project-id.iam.gserviceaccount.com can be obtained using
# > gcloud iam service-accounts list
gcloud_activate_service_account:
	gcloud auth activate-service-account \
	  --key-file="${GOOGLE_APPLICATION_CREDENTIALS}"

docker_allow:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev

docker_create_repo:
	gcloud artifacts repositories create $(DOCKER_REPO_NAME) \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--description="Repository for storing docker images"

docker_push:
	docker push $(DOCKER_IMAGE_PATH):prod

docker_deploy:
	gcloud run deploy \
		--image $(DOCKER_IMAGE_PATH):prod \
		--memory $(GCP_MEMORY) \
		--region $(GCP_REGION)

cloud_logs:
	gcloud run services logs read handspell-api \
		--project $(GCP_PROJECT) \
		--region $(GCP_REGION) \
		--limit 100
