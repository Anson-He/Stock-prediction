from keras.models import load_model
import numpy as np
data = [0.20]
data = np.array(data)
data = data.reshape(data.shape[0], 1, 1)
print(data)
model = load_model('../../model/day/SSE.600000.h5')
print(model.predict(data)[0][0])