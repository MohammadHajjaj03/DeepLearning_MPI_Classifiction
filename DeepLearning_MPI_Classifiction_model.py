from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,GlobalAveragePooling2D
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#تثبيت الراندوم سيد عشان ما يعطينا نتيجة مختلفة كل Run
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
#نورملايزيشن للصورة تحويلها ما بين 0 و 1
datagen = ImageDataGenerator(rescale=1./255)
#هون عملنا اونلاين اوقمنتيشن عشان نزيد عدد صور التدريب
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.01,
    height_shift_range=0.01,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
#طبقنا زيادة العدد على الصور
train_generator = train_aug.flow_from_directory(
    'C:/Users/mohma/OneDrive/Desktop/MPi/MP_split/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)
#طبقنا النورملايزيشن الطبيعي هون
val_generator = datagen.flow_from_directory(
    'C:/Users/mohma/OneDrive/Desktop/MPi/MP_split/val',
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical'
)
#طبقنا النورملايزيشن الطبيعي هون

test_generator = datagen.flow_from_directory(
    'C:/Users/mohma/OneDrive/Desktop/MPi/MP_split/test',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

#استخدمنا موبايل نت كونها LIGHT WIEGHT PRETRAINED MODLE
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(3, activation='softmax')(x)

early = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)
model = Model(inputs=base_model.input, outputs=predictions)
#عملنا فريز لاول 60 وخلينا بس اخر 26 يتعلموا عشان نستفيد من اول 60 لير الي هما اصلا متعلمين ومخلصين
base_model.trainable = True
for layer in base_model.layers[:60]:
    layer.trainable = False


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# طبقنا الكلاس ويت عشان نحل مشكل تباين عدد الصور
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Train Model
history = model.fit(train_generator,
          epochs=50,
          validation_data=val_generator,
          class_weight=class_weights,
          callbacks=[early])

# Evaluate on Test Set
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))



cm = confusion_matrix(y_true, y_pred)
labels = list(test_generator.class_indices.keys())

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Set')
plt.show()
# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
model.save('model.h5')


