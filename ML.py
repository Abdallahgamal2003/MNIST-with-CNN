import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize input images
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'],
)

model.summary()

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(test_images, test_labels)
print(f'loss : {loss * 100}% , accuracy : {accuracy * 100}%')

# Get the weights of the first convolutional layer
filters, biases = model.layers[0].get_weights()

# Plot the filters
plt.figure(figsize=(10, 10))
for i in range(32):
    f = filters[:, :, 0, i]
    ax = plt.subplot(4, 8, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(f, cmap='viridis')
    plt.title(f'Filter {i + 1}')
plt.show()

# Define the model for visualization
visualization_model = Model(inputs=model.inputs, outputs=model.layers[0].output)

# Get feature maps for the first hidden layer
feature_maps = visualization_model.predict(test_images)

# Plot feature maps
plt.figure(figsize=(15, 15))
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_maps[0, :, :, i], cmap='inferno')
    plt.title(f'Feature Map {i + 1}')
plt.show()
