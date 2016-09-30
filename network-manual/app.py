import data_loader
import network


def main():
    training_data, validation_data, test_data = data_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.sgd(training_data, 30, 10, 3.0, test_data=test_data)


if __name__ == "__main__":
    main()
