FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 \
    && pip install -r /tmp/requirements-api.txt

# Cache the ImageNet ResNet18 encoder during image build instead of first API request.
RUN python -c "from torchvision.models import ResNet18_Weights; ResNet18_Weights.DEFAULT.get_state_dict(progress=True)"

COPY sameobject_api.py /app/sameobject_api.py
COPY sameobject_training_web.py /app/sameobject_training_web.py
COPY tools/predict_sameobject_ensemble.py /app/tools/predict_sameobject_ensemble.py
COPY tools/train_sameobject_animal_classifier.py /app/tools/train_sameobject_animal_classifier.py
COPY tools/sameobject_preprocess_utils.py /app/tools/sameobject_preprocess_utils.py
COPY training_runs/sameobject_animal_classifier/first_full193/animal_classifier.pt /app/weights/first_full193/animal_classifier.pt
COPY training_runs/sameobject_animal_classifier/second_full193_parts_v1/animal_classifier.pt /app/weights/second_full193_parts_v1/animal_classifier.pt

EXPOSE 8090

ENTRYPOINT ["python", "sameobject_api.py", "--host", "0.0.0.0", "--port", "8090", "--full-weights", "/app/weights/first_full193/animal_classifier.pt", "--parts-weights", "/app/weights/second_full193_parts_v1/animal_classifier.pt"]
