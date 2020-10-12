import tensorflow as tf
from sklearn.ensemble import VotingClassifier
from testModel1 import x_train, y_train, x_test, y_test


# Load 84% acc model
json_file = open('Saved-Models/noContempt', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model1 = tf.keras.models.model_from_json(loaded_model_json)
loaded_model1.load_weights("Saved-Models/noContempt.h5")
loaded_model1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load 83.82% acc model
json_file = open('Saved-Models/model8382.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = tf.keras.models.model_from_json(loaded_model_json)
loaded_model2.load_weights("Saved-Models/model8382.h5")
loaded_model2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])


model = VotingClassifier(estimators=[('84%', loaded_model1), ('83.82%', loaded_model2)], voting='hard')
model.fit(x_train, y_train)
model.score(x_test, y_test)
