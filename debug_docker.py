# Helper file to debug Docker images
import faulthandler, sys; faulthandler.enable()

from handspell.model_landmarks import preprocess_and_predict_api, create_landmarker, load_asl_model

# image_path = "../data/raw_data/A05_notFonz_original/images/Afonso/H0.jpg"
image_path = "data/mapping/ASL Letter Reference/A.jpg"

with create_landmarker() as landmarker:
    asl_model = load_asl_model()
    r = preprocess_and_predict_api(image_path, landmarker, asl_model)
    print(r)

## Run in terminal:
# docker run --rm -v "$(pwd):/work" -w /app --entrypoint .venv/bin/python \
#  handspell-api:local /work/debug_docker.py data/mapping/ASL\ Letter\ Reference/A.jpg
