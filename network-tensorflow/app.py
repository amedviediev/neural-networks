from tensorflow.examples.tutorials.mnist import input_data

import multilayer_convolutional_network
import simple_network


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    simple_network.run(mnist)
    multilayer_convolutional_network.run(mnist)


if __name__ == "__main__":
    main()
