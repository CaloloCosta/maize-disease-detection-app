# from imageai.Prediction import ImagePrediction
import os
from app import APP_ROOT
import tensorflow as tf
import numpy as np
from PIL import Image
 
print(APP_ROOT)






os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

static_loc=os.path.join(APP_ROOT,'static/')
new_model = tf.keras.models.load_model(os.path.join(APP_ROOT,'static/','maize_model.h5'))
def predict_img(filename):
    img = tf.keras.utils.load_img(os.path.join(APP_ROOT,'static/',filename), target_size=(180,180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = new_model.predict(img_array)
    print(predictions)
    score = tf.nn.softmax(predictions[0])
    # print(score)
    class_names = ['FAW', 'MSV', 'healthy', 'unknown']
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    description = {
        "FAW": "Causes: Geminivirus trasmitted by leafhoppers Symptoms:pale,circular spots on younger leaves Treatment:Insecticide (recommeded pesticide:carbofuran) and farming during early season",
        "MSV": "Causes: Burrowing into maize ears by Fall armyworm larvae Symptoms:ragged holes in the leaves Treatment: Pesticides (recommeded pesticide:halofenozide)",
        "healthy": "The leaf is in a healthy condition,keep on treating it well",
        "unknown": "Please upload an image of a maize leave."

    }
    res = {
        "class" : "Not a leave" if class_names[np.argmax(score)] == "unknown" else class_names[np.argmax(score)],
        "confidence": "The model is " + str(100 * np.max(score)) + "confident of the prediction.",
        "confidence": "The model is with a {:.2f} % confidence.".format(100 * np.max(score)),
        "description": description[class_names[np.argmax(score)]]
    }
    return res

# def predict_img(filename):
#     tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#     target=os.path.join(APP_ROOT,'temp/'+filename) #location of image present in temp directory
#     prediction = ImagePrediction()
#     prediction.setModelTypeAsResNet()

#     tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#     prediction.setModelPath(os.path.join(static_loc, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
#     prediction.loadModel()
#     predictions, probabilities = prediction.predictImage(target, result_count=1)
#     d={} #dictionary that will save results
#     for eachPrediction, eachProbability in zip(predictions, probabilities):
#         d[eachPrediction]=eachProbability #prediction output
#         #print(eachPrediction , " : " , eachProbability)

#     os.remove(target) #delete temporary file

#     return d