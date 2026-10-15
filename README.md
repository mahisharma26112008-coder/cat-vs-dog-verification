# cat-vs-dog-verification

# CAT VS DOG IMAGE CLASSIFICATION USING DEEP LEARNING

## 1. Introduction

Cat vs Dog Image Classification is a Deep Learning project that identifies whether an input image contains a cat or a dog. It uses image processing and a Convolutional Neural Network (CNN) to learn the visual features of cats and dogs.

The model is trained using a dataset containing images of cats and dogs. After training, it can predict the class of a new image.

## 2. Objective

The main objectives of this project are:

- To understand the basic concepts of Deep Learning.
- To classify images into two categories: Cat and Dog.
- To learn how Convolutional Neural Networks (CNNs) work.
- To train a model using a labelled image dataset.
- To predict the category of an unseen image.

## 4. What is Image Classification?

Image Classification is a computer vision task in which a computer identifies the category of an image.

In this project, the model classifies each input image into one of the following categories:

- Cat
- Dog

The model learns from labelled training images and uses the learned patterns to classify new images.

## 5. What is a Convolutional Neural Network (CNN)?

A Convolutional Neural Network (CNN) is a type of Deep Learning model commonly used for image-related tasks.

A CNN automatically learns important features from images, such as edges, shapes, textures, and patterns.

The main layers of a CNN are:

1. **Convolutional Layer:** Extracts important features from an image.
2. **Activation Function (ReLU):** Helps the network learn complex patterns.
3. **Pooling Layer:** Reduces the spatial size of feature maps.
4. **Flatten Layer:** Converts feature maps into a one-dimensional representation.
5. **Dense Layer:** Uses the extracted features to make a prediction.
6. **Output Layer:** Predicts whether the image belongs to the Cat or Dog class.

## 6. Dataset Description

A dataset is a collection of images used to train and evaluate the model.

For this project, the dataset contains images of cats and dogs. Each image belongs to one of two classes.

The dataset is generally divided into:

- **Training Dataset:** Used to teach the model.
- **Validation Dataset:** Used to monitor model performance during training.
- **Testing Dataset:** Used to evaluate the trained model on separate images.

A balanced and diverse dataset can help the model learn more effectively.

## 7. Working Principle

The project works through the following steps:

1. Collect images of cats and dogs.
2. Load the dataset into the Python environment.
3. Resize images to a fixed size.
4. Normalize pixel values to help the model train.
5. Divide the dataset into training, validation, and testing sets.
6. Build a CNN model.
7. Train the model using labelled images.
8. Evaluate its performance on unseen images.
9. Provide a new image to the trained model.
10. Display the predicted class as Cat or Dog.

## 8. Tools and Technologies Used

- **Programming Language:** Python
- **Platform:** Google Colab
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib
- **Technique:** Deep Learning
- **Model:** Convolutional Neural Network (CNN)
- **Dataset:** Labelled Cat and Dog Images

## 9. Applications

Cat vs Dog Image Classification demonstrates techniques that are also useful in:

- Animal image recognition.
- Wildlife monitoring systems.
- Smart camera applications.
- Pet identification systems.
- Automated image categorization.
- Learning and research in computer vision.

## 10. Advantages

- Automatically identifies image categories.
- Reduces the need for manual image sorting.
- Helps understand CNN-based image recognition.
- Can process many images after training.
- Can be extended to classify more animal categories.

## 11. Limitations

- Prediction accuracy depends on the quality and diversity of the dataset.
- Poor-quality or unusual images may cause incorrect predictions.
- Training may require significant computing resources for large datasets.
- The model may perform poorly on images that differ greatly from its training data.

## 12. Conclusion

Cat vs Dog Image Classification is a beginner-friendly Deep Learning project that demonstrates how a computer can learn to recognize visual patterns in images. By using a Convolutional Neural Network (CNN), the model learns features from cat and dog images and predicts the category of a new image.


"""
Cat vs Dog Classifier (Deep Learning, single file)

Install:
    pip install tensorflow tensorflow-datasets matplotlib pillow

Train:
    python cat_vs_dog.py train

Predict on your own image:
    python cat_vs_dog.py predict my_photo.jpg
"""
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 3
MODEL_PATH = "cat_vs_dog_model.keras"
CLASS_NAMES = ["CAT", "DOG"]  # label 0 = cat, 1 = dog


def prepare(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(image, tf.float32))
    return image, label


def load_data():
    # Dataset downloads automatically (no Kaggle account needed)
    train, val, test = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        as_supervised=True,
    )
    train = (train.shuffle(1000).map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    val = (val.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    test = (test.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    return train, val, test


def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ])
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # keep pretrained knowledge, train only the top

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    train_ds, val_ds, test_ds = load_data()
    model = build_model()

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    loss, acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {acc:.2%}")

    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

    # Save accuracy graph
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Cat vs Dog - Training Accuracy")
    plt.legend()
    plt.savefig("training_plot.png", dpi=150, bbox_inches="tight")
    print("Graph saved as training_plot.png")


def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    prob_dog = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[1] if prob_dog >= 0.5 else CLASS_NAMES[0]
    confidence = prob_dog if prob_dog >= 0.5 else 1 - prob_dog
    print(f"Prediction: {label}  (confidence: {confidence:.1%})")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 3 and sys.argv[1] == "predict":
        predict(sys.argv[2])
    else:
        print("Usage:")
        print("  python cat_vs_dog.py train")
        print("  python cat_vs_dog.py predict <image_path>")

This project provides practical knowledge of image preprocessing, model training, evaluation, and prediction. It also builds a foundation for more advanced computer vision applications.
