import numpy as np
import os

class layer():
    def plain(self, nods, activation, input_shape=None):
        def layer(x, variables):
            w, b = variables
            Z = x.dot(w)+b
            A = activation(Z)
            return (Z, A)

        return layer, nods, input_shape, 'plain'

    def conv2D(self, characteristics, filter_size_tuple, activation=None, stride=1, padding=0, input_shape=None):
        filter_size = filter_size_tuple[0]
        def layer(x, filters):
            Z = []
            for one_x in x:
                multiple_filters = []
                for one_real_x in one_x:
                    for i in range(padding):
                        one_real_x = np.insert(one_real_x, one_real_x.shape[1], 0, axis=1)
                        one_real_x = np.insert(one_real_x, 0, 0, axis=1)
                        one_real_x = np.insert(one_real_x, one_real_x.shape[0], 0, axis=0)
                        one_real_x = np.insert(one_real_x, 0, 0, axis=0)
                    width, height = len(one_real_x[0]), len(one_real_x)
                    for Filter in filters:
                        filtered_one_x = []
                        h = 0
                        while (h+filter_size) <= height:
                            one_line = []
                            w = 0
                            while (w+filter_size) <= width:
                                cropped_data = one_real_x[h:int(h+filter_size), w:int(w+filter_size)]
                                filtered_area = cropped_data*Filter
                                filtered_area = np.sum(filtered_area)
                                one_line.append(filtered_area)
                                w += stride
                            filtered_one_x.append(one_line)
                            h += stride
                        filtered_one_x = np.asarray(filtered_one_x)
                        multiple_filters.append(filtered_one_x)
                multiple_filters = np.asarray(multiple_filters)
                Z.append(multiple_filters)
            Z = np.asarray(Z)
            if activation == None:
                A = Z
            else:
                A = activation(Z)
            return (Z, A)

        return layer, filter_size_tuple, input_shape, 'conv2D', characteristics, stride, padding

    def maxpool2D(self, filter_size_tuple, stride=1, padding=0):
        filter_size = filter_size_tuple[0]
        def layer(x, trainable_varaible = None):
            Z = []
            for one_x in x:
                filtered_one_x = []
                for one_real_x in one_x:
                    for i in range(padding):
                        one_real_x = np.insert(one_real_x, one_real_x.shape[1], 0, axis=1)
                        one_real_x = np.insert(one_real_x, 0, 0, axis=1)
                        one_real_x = np.insert(one_real_x, one_real_x.shape[0], 0, axis=0)
                        one_real_x = np.insert(one_real_x, 0, 0, axis=0)
                    width, height = len(one_real_x[0]), len(one_real_x)
                    h = 0
                    pooled_one_x = []
                    while (h+filter_size) <= height:
                        one_line = []
                        w = 0
                        while (w+filter_size) <= width:
                            cropped_data = one_real_x[h:int(h+filter_size), w:int(w+filter_size)]
                            filtered_area = np.max(cropped_data)
                            one_line.append(filtered_area)
                            w += stride
                        pooled_one_x.append(one_line)
                        h += stride
                    filtered_one_x.append(pooled_one_x)
                Z.append(filtered_one_x)
            Z = np.asarray(Z)
            return Z, Z

        return layer, filter_size_tuple, None, 'maxpool2D', stride, padding

    def flatten(self):
        def layer(x, trainable_variable = None):
            flatten_array = []
            for one_x in x:
                size_one_x = one_x.shape
                size_one_x = size_one_x[0]*size_one_x[1]*size_one_x[2]
                one_x = np.reshape(one_x, (size_one_x, ))
                flatten_array.append(one_x)
            flatten_array = np.asarray(flatten_array)
            return flatten_array, flatten_array
        return layer, None, None, 'flatten'

def ReLU(X):
    return np.maximum(X, 0)

def ReLU_deriv(X):
    return X>0

def softmax(Z):
    A = np.exp(Z.T) / sum(np.exp(Z.T))
    return A.T

class Model():
    def __init__(self, layers):
        input_shape = layers[0][2]
        self.trainable_variables = []
        self.layers = layers
        self.layers_types = []
        self.list_of_returned_image_sizes = []
        for count, Layer in enumerate(layers):
            self.layers_types.append(Layer[3])
            if Layer[3] == 'plain':
                if count == 0:
                    index = int(len(np.shape(input_shape)) - 1)
                    W = np.random.rand(input_shape[index], Layer[1]) - 0.5
                    b = np.random.rand(Layer[1])
                    package = [W, b]
                    self.trainable_variables.append(package)
                elif self.layers_types[int(count-1)] == 'flatten':
                    previous_size = self.list_of_returned_image_sizes[int(count-1)]
                    W = np.random.rand(previous_size, Layer[1])-0.5
                    b = np.random.rand(Layer[1])
                    package = [W, b]
                    self.trainable_variables.append(package)
                else:
                    previous_layer_shape = np.shape(self.trainable_variables[int(count - 1)][1])
                    index = int(len(np.shape(previous_layer_shape)) - 1)
                    W = np.random.rand(previous_layer_shape[index], Layer[1]) - 0.5
                    b = np.random.rand(Layer[1])
                    package = [W, b]
                    self.trainable_variables.append(package)

            if Layer[3] == 'conv2D':
                filter_size_tuple = Layer[1]
                characteristics = Layer[4]
                if input_shape != None and count == 0:
                    fake_w, fake_h = 0, 0
                    fake_stride, fake_padding = Layer[5], Layer[6]
                    x_height, x_width = input_shape[1]+2*fake_padding, input_shape[2]+2*fake_padding
                    filter_size_int = filter_size_tuple[0]
                    while (fake_w+filter_size_int) <= x_width:
                        fake_w += fake_stride
                    while (fake_h+filter_size_int) <= x_height:
                        fake_h += fake_stride
                    new_image_size = (int(input_shape[0]*characteristics), int(fake_h/fake_stride), int(fake_w/fake_stride))
                    self.list_of_returned_image_sizes.append(new_image_size)
                else:
                    fake_w, fake_h = 0, 0
                    fake_stride, fake_padding = Layer[5], Layer[6]
                    x_height, x_width = self.list_of_returned_image_sizes[int(count-1)][1] + 2 * fake_padding, self.list_of_returned_image_sizes[int(count-1)][2] + 2 * fake_padding
                    filter_size_int = filter_size_tuple[0]
                    while (fake_w + filter_size_int) <= x_width:
                        fake_w += fake_stride
                    while (fake_h + filter_size_int) <= x_height:
                        fake_h += fake_stride
                    new_image_size = (int(self.list_of_returned_image_sizes[int(count-1)][0]*characteristics), int(fake_h / fake_stride), int(fake_w / fake_stride))
                    self.list_of_returned_image_sizes.append(new_image_size)

                trainable_variable = []
                for i in range(characteristics):
                    if len(filter_size_tuple) == 2:
                        one_filter = np.random.rand(filter_size_tuple[0], filter_size_tuple[1])
                    else:
                        one_filter = np.random.rand(filter_size_tuple[0], filter_size_tuple[1], filter_size_tuple[2])
                    trainable_variable.append(one_filter)
                trainable_variable = np.asarray(trainable_variable)
                self.trainable_variables.append(trainable_variable)

            if Layer[3] == 'maxpool2D':
                self.trainable_variables.append(None)
                filter_size_tuple = Layer[1]
                fake_w, fake_h = 0, 0
                fake_stride, fake_padding = Layer[4], Layer[5]
                x_height, x_width = self.list_of_returned_image_sizes[int(count - 1)][1] + 2 * fake_padding, self.list_of_returned_image_sizes[int(count - 1)][2] + 2 * fake_padding
                filter_size_int = filter_size_tuple[0]
                while (fake_w + filter_size_int) <= x_width:
                    fake_w += fake_stride
                while (fake_h + filter_size_int) <= x_height:
                    fake_h += fake_stride
                new_image_size = (int(self.list_of_returned_image_sizes[int(count-1)][0]), int(fake_h / fake_stride), int(fake_w / fake_stride))
                self.list_of_returned_image_sizes.append(new_image_size)

            if Layer[3] == 'flatten':
                self.trainable_variables.append(None)
                previous_size = self.list_of_returned_image_sizes[int(count-1)]
                new_size = previous_size[0]*previous_size[1]*previous_size[2]
                self.list_of_returned_image_sizes.append(new_size)

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
        try:
            os.makedirs(saving_path)
        except:
            pass
        for a, package in enumerate(self.trainable_variables):
            try:
                os.makedirs(saving_path+'\\{}'.format(a))
            except:
                pass
            for b, variable in enumerate(package):
                try:
                    os.makedirs(saving_path+'\\{}\\{}'.format(a, b))
                except:
                    pass
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