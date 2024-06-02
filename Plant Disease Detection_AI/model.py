import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array , load_img
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
model = load_model("best_model.h5")
# ref = dict(zip(list(train.class_indices.values()) , list(train.class_indices.keys()) ))
data = [
    "apple apple scab",
    "apple black rot",
    "apple cedar apple rust",
    "apple healthy",
    "blueberry healthy",
    "cherry including sour powdery mildew",
    "cherry including sour healthy",
    "corn maize cercospora leaf spot gray leaf spot",
    "corn maize common rust",
    "corn maize northern leaf blight",
    "corn maize healthy",
    "grape black rot",
    "grape esca black measles",
    "grape leaf blight isariopsis leaf spot",
    "grape healthy",
    "orange haunglongbing citrus greening",
    "peach bacterial spot",
    "peach healthy",
    "pepper bell bacterial spot",
    "pepper bell healthy",
    "potato early blight",
    "potato late blight",
    "potato healthy",
    "raspberry healthy",
    "soybean healthy",
    "squash powdery mildew",
    "strawberry leaf scorch",
    "strawberry healthy",
    "tomato bacterial spot",
    "tomato early blight",
    "tomato late blight",
    "tomato leaf mold",
    "tomato septoria leaf spot",
    "tomato spider mites two spotted spider mite",
    "tomato target spot",
    "tomato tomato yellow leaf curl virus",
    "tomato tomato mosaic virus",
    "tomato healthy"
]

ref = {}
for idx, name in enumerate(data):
    ref[idx] = name

def prediction(path):
  img = load_img(path , target_size=(224,224))

  i = img_to_array(img)

  im = preprocess_input(i)

  img = np.expand_dims(im , axis=0)

  pred = np.argmax(model.predict(img))


  print(f"image belongs to {ref[pred]}")
  print("DOne",pred)

path="pepperbellbactspotjpg.jpg"
prediction(path)