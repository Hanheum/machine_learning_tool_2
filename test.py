import ai_tool_library
import numpy as np
from getpass import getuser

username = getuser()

x = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
]

y = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
]

x = np.asarray(x)
y = np.asarray(y)


#=============================================
#make a model and train it, and save it
layers = [
    ai_tool_library.layer().plain(nods=30, activation=ai_tool_library.ReLU, input_shape=(2, )),
    ai_tool_library.layer().plain(nods=20, activation=ai_tool_library.ReLU),
    ai_tool_library.layer().plain(nods=10, activation=ai_tool_library.ReLU),
    ai_tool_library.layer().plain(nods=3, activation=ai_tool_library.ReLU)
]

model = ai_tool_library.Model(layers)

model.train(ai_tool_library.optimizer, x, y, epochs=1000)

model.save('C:\\Users\\{}\\Desktop\\test_model_save'.format(username))

#=============================================
#load a new model from previous save, a new model should share same layers with the original model to use the save files

model2 = ai_tool_library.Model(layers)

model2.load('C:\\Users\\{}\\Desktop\\test_model_save'.format(username))

result = model2.predict(x)
for i in result:
    miniList = []
    for a in i:
        miniList.append(round(a, 3))
    print(miniList)