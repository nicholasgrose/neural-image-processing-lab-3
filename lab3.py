import os
import numpy as np
from scipy.optimize.optimize import fmin
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
import imageio
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SAMPLE = 1
CONTENT_IMG_PATH = f"./shared/style_transfer/transfer_sample{SAMPLE}/content.jpg"
STYLE_IMG_PATH = f"./shared/style_transfer/transfer_sample{SAMPLE}/style.jpg"


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3



#=============================<Helper Functions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    return Image.fromarray(img, 'rgb')


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================
class Wrapper:
    def __init__(self, content: np.ndarray, style: np.ndarray, gen: np.ndarray):
        tf.compat.v1.disable_eager_execution()
        K.set_floatx('float64')
        self.content = content
        self.style = style
        self.gen = gen
        print("   Building transfer model.")
        contentTensor = K.variable(content)
        styleTensor = K.variable(style)
        # genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        genTensor = K.variable(gen)
        inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
        self.model = self.construct_model(inputTensor)

    def construct_model(self, inputTensor: tf.Tensor) -> keras.Model:
        return vgg19.VGG19(include_top=False, input_tensor=inputTensor)

    def styleLoss(self, style: np.ndarray, gen: np.ndarray) -> tf.Tensor:
        styleShape = style.shape
        M = (styleShape[0] * styleShape[1]) ** 2
        N = styleShape[2] ** 2
        error = K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4 * N * M)
        return error

    def contentLoss(self, content: np.ndarray, gen: np.ndarray) -> tf.Tensor:
        return K.sum(K.square(gen - content))

    def totalLoss(self, output: np.ndarray) -> tf.Tensor:
        output = K.reshape(output, (3, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        self.model = self.construct_model(output)
        outputDict = dict([(layer.name, layer.output) for layer in self.model.layers])
        loss = 0.0
        styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
        contentLayerName = "block5_conv2"
        print("   Calculating content loss.")
        contentLayer = outputDict[contentLayerName]
        contentOutput = contentLayer[0, :, :, :]
        genOutput = contentLayer[2, :, :, :]
        loss += CONTENT_WEIGHT * self.contentLoss(contentOutput, genOutput) / 2
        print("   Calculating style loss.")
        for layerName in styleLayerNames:
            styleLayer = outputDict[layerName]
            styleOutput = styleLayer[0, :, :, :]
            genOutput = styleLayer[2, :, :, :]
            layerWeight = 1 / self.activeLayers(layerName)
            loss += STYLE_WEIGHT * layerWeight * self.styleLoss(styleOutput, genOutput)
        print(f"=========================================================================={loss.shape}")
        return loss

    def activeLayers(self, layerName: str) -> int:
        layerCount = 1
        for layer in self.model.layers:
            if layer.name == layerName:
                break
            layerCount += 1
        return layerCount

    def gradient(self, x):
        # self.model = self.construct_model(output)
        grads = K.gradients(self.totalLoss(x), self.model.inputs)
        grads = K.flatten(grads)
        return grads


#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_temp = Image.fromarray(img, 'RGB').resize((ih, iw))
        img = np.array(img_temp)
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and de-processed images.
'''
wrapper = None
def styleTransfer(cData, sData, tData):
    print("   VGG19 model loaded.")
    global wrapper
    wrapper = Wrapper(cData, sData, tData)
    # TODO: Setup gradients or use K.gradients().
    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        x, tLoss, d = fmin_l_bfgs_b(wrapper.totalLoss, np.array([cData, sData, tData]), fprime=wrapper.gradient, maxiter=1000, maxfun=30)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        print("      Loss: %f." % tLoss)
        img = deprocessImage(x)
        saveFile = f"./shared/style_transfer/transfer_sample{SAMPLE}/result.jpg"
        imageio.imwrite(saveFile, img)   #Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")

#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
