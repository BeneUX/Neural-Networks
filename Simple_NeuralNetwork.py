# Importieren der notwendigen Bibliotheken
import tensorflow as tf
import matplotlib.pyplot as plt

#Datensatz von Tensorflow herunterladen
mnist = tf.keras.datasets.mnist

# Trainingdatensatz zum Lernen und Testdatensatz zum messen der Accuracy
(training_data, training_labels), (test_data, test_labels) = mnist.load_data()
# Durch 255 teilen um Pixel auf Fließkommazahl zu bringen
training_data, test_data = training_data / 255, test_data / 255

# Modell erstellen, sequential von tf benutzen um Layer zu kreieren.
model = tf.keras.Sequential([
    # Input Layer erstellen, Pixelgröße = 28x28 = 784 auf Array flatten
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Hidden Layer erstellen, Overfitting vermeiden, daher nicht zu viele Hidden Layer! Insgesamt 128 Layer mashed mit den 784 Neuronen
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # Output Layer mit Aktivierungsfunktion softmax um insgesamt auf 100& zu kommen.
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Optimizer, Loss und Metriken definieren
model.compile(
    optimizer= tf.optimizers.Adam(),
    # Fehler des Netzwerks erkennen und mit Accuracy bestimmen
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Neuronales Netzwerk trainieren indem erkannte der richtigen Antwort gegenüber gestellt wird.
model.fit(training_data, training_labels, epochs=5)

# Wie gut war das Netzwerk? Wie reagiert das Netzwerk wenn es Daten zum ersten mal sieht.
model.evaluate(test_data, test_labels)

# Alle Testdaten klassifizieren
predictions = model.predict(test_data)


image_index = 90
plt.imshow(test_data[image_index], cmap='Greys')


# Testen ob das neuronale Netz die Zahl richtig erkannt hat
plt.title('True: {} \nPredict: {}'.format(test_labels[image_index], np.argmax(predictions[image_index])))
plt.imshow(test_data[image_index], cmap='Greys')


import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, '/content/model/')
