# Import standard libraries
import numpy as np
import urllib.request
from PIL import Image
from skimage import transform
import matplotlib.pylab as plt
import matplotlib.image as mpimg

# Import tensorflow libraries
import tensorflow as tf
import tensorflow_hub as hub

class ImageClassification:
    def __init__(self, params):
        """
        Initialization of parameters and layers.
        """
        self.params = params
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.normalization_layer = tf.keras.layers.Rescaling(1./255)

    def load_images(self, subset):
        """
        Function to load training or validation images.
        """
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.params['train_path'],
            validation_split=self.params['validation_split'],
            subset=subset,
            image_size=(self.params['img_height'], self.params['img_width']),
            batch_size=self.params['batch_size'],
            seed = self.params['seed_train_validation'],
            shuffle = self.params['shuffle_value'] )

        class_names = ds.class_names
        # Normalize and prefetch images
        ds = ds.map(lambda x, y: (self.normalization_layer(x), y)) # Where x—images, y—labels.
        ds = ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        return ds, class_names

    def create_model(self, num_classes):
        """
        Function to create the model.
        """
        feature_extractor_layer = hub.KerasLayer(
            self.params['efficientnet_v2_fv'],
            input_shape=(self.params['img_width'], self.params['img_height'], 3),
            trainable=False)

        model = tf.keras.Sequential([
            feature_extractor_layer,
            tf.keras.layers.Dense(num_classes)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])

        return model

    def train_model(self, model, train_ds, val_ds):
        """
        Function to train the model.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=self.params['NUM_EPOCHS'],
                            callbacks=[early_stopping])
        return model, history

    def image_class_prediction(self, url, model, class_names, expected_class=None):
        """
        Function for prediction and visualization.
        """
        test_image_filename = 'randomImage.jpg'
        urllib.request.urlretrieve(url, test_image_filename)
        test_img_path = './'+test_image_filename 

        test_image = Image.open(test_img_path)
        test_image = np.array(test_image).astype('float32')/255
        test_image = transform.resize(test_image, (self.params['img_width'], self.params['img_height'], 3))
        test_image = np.expand_dims(test_image, axis=0)

        prediction = model.predict(test_image)
        pred_class = prediction.argmax()
        print(f"The Predicted Class: {class_names[pred_class]}\n")

        plt.figure(figsize=(8,8))
        test_img = mpimg.imread(test_img_path)
        plt.imshow(test_img)
        plt.title("predicted class: "+class_names[pred_class]+" | expected class: "+str(expected_class))
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    params = {'train_path': 'Data Set Path Here',
              'batch_size': 64,
              'img_height': 224,
              'img_width': 224,
              'seed_train_validation': 1,
              'shuffle_value': True,
              'validation_split': 0.1,
              'efficientnet_v2_fv': 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2',
              'NUM_EPOCHS': 20}

    image_classifier = ImageClassification(params)

    # Load training and validation images
    (train_ds, class_names) = image_classifier.load_images("training")
    (val_ds, _) = image_classifier.load_images("validation")

    # Create and train the model
    model = image_classifier.create_model(len(class_names))
    model, history = image_classifier.train_model(model, train_ds, val_ds)

    # Print model accuracy
    model_acc = '{:.2%}'.format(history.history['acc'][-1])
    print(f"\n Model Accuracy Reached: {model_acc}")

    # Perform prediction on random images
    url_list = ['Put Test Image URL Here', 'Expected Class Here']

    url_dict = {url_list[i]: url_list[i + 1] for i in range(0, len(url_list), 2)}

    for url,label in url_dict.items():
        image_classifier.image_class_prediction(url, model, class_names, expected_class=label)
