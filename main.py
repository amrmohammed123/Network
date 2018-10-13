import network
import mnist_loader

choice = int(input('Enter 0 for training the network and 1 for loading the existing model:'))
net = network.Network([784, 250, 10])
if choice == 0:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(True)
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)
    net.SGD(training_data, 10 , 1 , 0.5 , validation_data)
    net.save('data.json')
else:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)
    net.load('data.json')
print('number of correctly classified data = ' + str(net.evaluate(test_data)) + str(' from ') + str(len(test_data)))
print('accuracy of test_data = ' + str(net.accuracy(test_data)) + str(' %'))

