# **Deep Food — Classification de 101 Catégories Alimentaires avec CNN, Transfert d’Apprentissage & Précision Mixte**

Ce projet a pour objectif de **classer plus de 100 catégories d’aliments** à partir du dataset *Food101* en utilisant un modèle CNN moderne optimisé via :

* **Transfert d’apprentissage (EfficientNetV2B0)**
* **Entraînement en précision mixte (float16/float32)**
* **GPU acceleration (NVIDIA RTX)**
* **Pipeline TensorFlow Dataset performant**
* **Callbacks TensorBoard & ModelCheckpoint**

Il s’agit d’un projet **personnel** visant à développer de solides compétences en vision par ordinateur, deep learning et optimisation de modèles.


## 🎯 **Objectifs du projet**

* Utiliser le dataset **Food101** avec TensorFlow Datasets
* Prétraiter efficacement + normaliser + redimensionner toutes les images
* Construire un modèle CNN basé sur **EfficientNetV2B0** pré-entraîné
* Activer la **précision mixte** pour un entraînement plus rapide
* Entraîner et valider le modèle sur GPU
* Sauvegarder les meilleurs poids via *ModelCheckpoint*
* Visualiser et analyser les courbes d’apprentissage
* Obtenir une **précision élevée** malgré +100 classes


## 🧰 **Stack Technique**

* **Python 3**
* **TensorFlow 2 / Keras**
* **TensorFlow Datasets (TFDS)**
* **GPU NVIDIA (RTx)** + CUDA + cuDNN
* **EfficientNetV2B0** (pretrained ImageNet)
* **Mixed Precision Training**
* **Callbacks TensorBoard & Checkpoints**
* **Matplotlib**


## 📦 **Dataset : Food101**

Dataset officiel :
101 catégories d’aliments

* 1000 images par classe
  Images non normalisées (0–255), tailles variables.

Chargement via TFDS :

```python
(train_data, test_data), ds_info = tfds.load(
    name="food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
```

Nombre de classes :

```python
len(class_names)  # 101
```


## 🧹 **Prétraitement & Pipeline TensorFlow**

### Problèmes dans les données :

* images tailles variables
* pixels entiers (uint8)
* besoin de normalisation

### Étapes de prétraitement :

* **Redimensionnement** à 224×224
* **Cast en float32**
* **Batching + Prefetching GPU**

```python
def preprocess_img(image, label, img_shape=224):
    image = tf.image.resize(image, [img_shape, img_shape])
    return tf.cast(image, tf.float32), label
```

Pipeline optimisé :

```python
train_data = train_data.map(...).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_data  = test_data.map(...).batch(32).prefetch(tf.data.AUTOTUNE)
```


## ⚡ **Accélération : Entraînement en précision mixte**

Active :

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
```

Avantages :

* Entraînement **2x plus rapide**
* Réduction de l'utilisation mémoire GPU
* Bénéficiel pour EfficientNet


## 🧠 **Modèle CNN — EfficientNetV2B0 (Transfert d’Apprentissage)**

Base :

```python
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False
)
base_model.trainable = False  # Extraction de features
```

Architecture :

* Input Layer
* EfficientNetV2B0 gelé
* GlobalAveragePooling2D
* Dense(101)
* Softmax **float32** (pour éviter la perte de précision mixte)

Compilation :

```python
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
```


## 🔄 **Callbacks : TensorBoard & Model Checkpoints**

### TensorBoard

```python
create_tensorboard_callback()
```

### Checkpoints

```python
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model_checkpoints/cp.weights.h5",
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=True
)
```


## 🏋️ **Entraînement**

Modèle exécuté sur GPU NVIDIA RTX 2060 :

```bash
!nvidia-smi -L
```

Training :

```python
history = model.fit(train_data,
                    validation_data=test_data,
                    epochs=EPOCHS,
                    callbacks=[tensorboard, model_checkpoint])
```


## 📈 **Évaluation & Visualisation**

Les courbes d’apprentissage (loss/accuracy) sont affichées via :

```python
plot_loss_curves(history)
```

Possibilité de comparer plusieurs phases d’entraînement :

```python
compare_historys(...)
```


## 🧠 **Compétences démontrées**

✔ Vision par ordinateur avancée
✔ CNN avec TensorFlow/Keras
✔ Transfert d’apprentissage EfficientNet
✔ Optimisation GPU + précision mixte
✔ Manipulation TFDS & pipelines haute performance
✔ Prétraitement d’images deep learning
✔ Callbacks professionnels (TensorBoard, checkpointing)
✔ Classification multi-classes (101 catégories)
✔ Programmation orientée performance (prefetch, AUTOTUNE)


## 🚀 **Pistes d’amélioration**

* Ajout d’un scheduler (ReduceLROnPlateau)
* Augmentation de données (tf.image)
* Test d’EfficientNetV2M ou V2L
* Export en format TF-Lite pour mobile
* Déploiement API (FastAPI) + front-end


## 👤 **À propos**

Projet réalisé par **Alex Alkhatib**, passionné par la vision par ordinateur et les modèles deep learning modernes.


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib

Souhaites-tu l’un de ces bonus ?
