# Helper file to debug Docker images
import faulthandler, sys; faulthandler.enable()

# from handspell.data_preprocessing import load_model as preprocessing_load_model, image_preprocessing
# from keras.models import load_model as keras_load_model
from handspell.model import preprocessing_load_model, image_preprocessing

m = preprocessing_load_model("models/hand_detection_model.pt")
print(m.predict(sys.argv[1], verbose=False)[0].boxes)
p = image_preprocessing(sys.argv[1])

## Run in terminal:
# docker run --rm -v "$(pwd):/work" -w /app --entrypoint .venv/bin/python \
#  handspell-api:local /src/debug_utils/debug_docker.py data/mapping/ASL\ Letter\ Reference/A.jpg
