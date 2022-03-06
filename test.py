import ai_tool_library
import numpy as np

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

layers = [
    ai_tool_library.layer().plain(nods=30, activation=ai_tool_library.ReLU, input_shape=(2, )),
    ai_tool_library.layer().plain(nods=20, activation=ai_tool_library.ReLU),
    ai_tool_library.layer().plain(nods=10, activation=ai_tool_library.ReLU),
    ai_tool_library.layer().plain(nods=3, activation=ai_tool_library.ReLU)
]

model = ai_tool_library.Model(layers)

model.train(ai_tool_library.optimizer, x, y, epochs=1000)

result = model.predict(x)
print(result)