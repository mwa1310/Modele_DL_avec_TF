import mlflow
#import mlflow.tenforflow
import tensorflow as tf
import keras
import numpy as np


# Variables pour les paramètres
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

optimizers = {
"SGD_with_momentum": keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
"RMSprop": 'rmsprop',
"Adam": 'adam'
}

# Lancement de la session de suivi MLflow
for opt_name, optimizer in optimizers.items():
    with mlflow.start_run(run_name = f"Optimizer_Comparison_{opt_name}"):
# Enregistrement des paramètres
        mlflow.log_param("epochs ", EPOCHS)
        mlflow.log_param("batch_size ", BATCH_SIZE)
        mlflow.log_param("dropout_rate ", DROPOUT_RATE)

# =========================================================================
# Construction et entrainement du modèle (utiliser les variables définies)
# =========================================================================

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data ()

        # Normalisation des données
        x_train = x_train.astype("float32") / 255.0
        x_test =x_test.astype("float32") / 255.0 # On ramène les données dans l'intervale [0, 1]

        # Redimensionnement des images pour les réseaux fully-connected
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

        # Construction du modèle
        model = keras.Sequential([
            keras.layers.Dense(512, activation = 'relu', input_shape =(784,)), # ici tous les neurones de la couche sont connectés à la couche précédente
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(DROPOUT_RATE), # couche sans paramètre, qui désactive un certain pourcentage d'activation
            keras.layers.Dense(10, activation ='softmax') # la fonction d'acti ici definit la loss à utiliser
        ])

        # Compilation du modèle
        model.compile( # on definit ici les props de l'optimiseur
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy', # ne fonctionne que pour le softmax
            metrics =['accuracy']
        ) #Avec tf on ne peut pas faire le gridsearch qui consiste à entrer des valeurs des valeurs d'accuracy et choisir à la fin le meilleur-modèle mais mlflow est une bonne alternative pour la tracabilité des resultats pour choisir le meilleur modèle"""

        # Entrainement du modèle
        history = model.fit(
        x_train ,
        y_train ,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE ,
        validation_split = 0.1
        )

        # Evaluation du modèle
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"Précision sur les données de test : {test_acc:.4f}")

        # Sauvegarde du modèle
        model.save("mnist_model.h5")
        print("Modèle sauvegardé sous mnist_model.h5")

        # Enregistrement des métriques
        mlflow.log_param(" optimizer ", opt_name )
        mlflow.log_metric(" final_test_accuracy ", test_acc )

        # Enregistrement du modèle complet
        mlflow.keras.log_model(model, "mnist-model")
