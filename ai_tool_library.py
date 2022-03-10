import numpy as np
import os

class layer():
    def plain(self, nods, activation, input_shape=None):
        def layer(x, variables):
            w, b = variables
            Z = x.dot(w)+b
            A = activation(Z)
            return (Z, A)

        return layer, nods, input_shape

def ReLU(X):
    return np.maximum(X, 0)

def ReLU_deriv(X):
    return X>0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

class Model():
    def __init__(self, layers):
        input_shape = layers[0][2]
        self.trainable_variables = []
        self.layers = layers
        for count, Layer in enumerate(layers):
            if count == 0:
                index = int(len(np.shape(input_shape))-1)
                W = np.random.rand(input_shape[index], Layer[1])-0.5
                b = np.random.rand(Layer[1])
                package = [W, b]
                self.trainable_variables.append(package)
            else:
                previous_layer_shape = np.shape(self.trainable_variables[int(count-1)][1])
                index = int(len(np.shape(previous_layer_shape))-1)
                W = np.random.rand(previous_layer_shape[index], Layer[1])-0.5
                b = np.random.rand(Layer[1])
                package = [W, b]
                package = np.asarray(package)
                self.trainable_variables.append(package)

    def calculate_train(self, x):
        Zs = []
        As = []
        target = x
        for count, Layer in enumerate(self.layers):
            function = Layer[0]
            Z, A = function(target, self.trainable_variables[count])
            target = A
            Zs.append(Z)
            As.append(A)
        return Zs, As

    def predict(self, x):
        target = x
        for count, Layer in enumerate(self.layers):
            function = Layer[0]
            Z, A = function(target, self.trainable_variables[count])
            target = A
            if count == len(self.layers)-1:
                return A

    def train(self, optimizer, x_data, y_data, epochs=1):
        for epoch in range(epochs):
            print('epoch:{}'.format(int(epoch+1)))
            optimizer(x_data, y_data, self.trainable_variables, self.calculate_train)

    def save(self, saving_path):
        os.makedirs(saving_path)
        for a, package in enumerate(self.trainable_variables):
            os.makedirs(saving_path+'\\{}'.format(a))
            for b, variable in enumerate(package):
                os.makedirs(saving_path+'\\{}\\{}'.format(a, b))
                file = open(saving_path+'\\{}\\{}\\save_file'.format(a, b), 'wb')
                file.write(variable.tobytes())
                file2 = open(saving_path+'\\{}\\{}\\shape.txt'.format(a, b), 'wb')
                file2.write('{}'.format(np.shape(variable)).encode())

    def load(self, file_path):
        layer_folders = os.listdir(file_path)
        loaded_variables = []
        for folder in layer_folders:
            elements = os.listdir(file_path+'\\{}'.format(folder))
            package = []
            for element in elements:
                file_name = os.listdir(file_path+'\\{}\\{}'.format(folder, element))[0]
                variable_shape_file_name = os.listdir(file_path+'\\{}\\{}'.format(folder, element))[1]
                variable_shape_file_dir = file_path+'\\{}\\{}\\{}'.format(folder, element, variable_shape_file_name)
                shape = open(variable_shape_file_dir, 'rb').read().decode()
                shape = eval(shape)
                file = open(file_path+'\\{}\\{}\\{}'.format(folder, element, file_name), 'rb').read()
                variable = np.frombuffer(file, np.float64)
                variable = np.reshape(variable, shape)
                package.append(variable)
            loaded_variables.append(package)
        self.trainable_variables = loaded_variables

def optimizer(x_data, y_data, trainable_variables, calculate_function, learning_rate=0.01):
    Zs, As = calculate_function(x_data)
    As.append(x_data)
    m = len(y_data)
    last_index = int(len(trainable_variables)-1)
    packages = []
    dzs = []
    for count in range(len(Zs)):
        index = int(last_index-count)
        if index == last_index:
            dz = As[index].T-y_data.T
            dw = 1/m * dz.dot(As[int(index-1)]).T
            db = 1/m * np.sum(dz)
        else:
            dz = trainable_variables[int(index+1)][0].dot(dzs[0])*ReLU_deriv(Zs[index]).T
            dw = 1/m * dz.dot(As[int(index-1)]).T
            db = 1/m * np.sum(dz)
        dzs.insert(0, dz)
        package = [dw, db]
        packages.insert(0, package)

    for a, package in enumerate(trainable_variables):
        trainable_variables[a][0] = package[0]-learning_rate*packages[a][0]
        trainable_variables[a][1] = package[1]-learning_rate*packages[a][1]
    return trainable_variables