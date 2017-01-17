from __future__ import print_function
import tensorflow as tf
from util import *


def logistic_regresion(x):
    # Set model weights
    W = tf.Variable(tf.zeros([2, 1]))
    b = tf.Variable(tf.zeros([1]))

    return tf.matmul(x, W) + b


def deep_network_easy_way(x):
    fc1 = tf.contrib.layers.fully_connected(x, num_outputs=10, activation_fn=tf.nn.relu)
    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=10, activation_fn=tf.nn.relu)
    fc3 = tf.contrib.layers.fully_connected(fc2, num_outputs=1, activation_fn=None)
    return fc3


def main(show_plots=False):
    # Import our data
    dataset = read_file("dataset2")
    if show_plots:
        plot_dataset(dataset, False)
        plot_dataset(dataset, True)

    # Parameters
    learning_rate = 0.01
    training_epochs = 100

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 2])  # our data
    labels = tf.placeholder(tf.float32, [None, 1])  # our labels

    h = deep_network_easy_way(x)

    pred = tf.greater(tf.nn.sigmoid(h), 0.5)

    # Minimize error using cross entropy
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(h, labels))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    # Test model
    correct_prediction = tf.equal(tf.cast(pred, tf.float32), labels)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Epoch: 0")
        print("\tValidaton accuracy: {:.3f}".format(accuracy.eval({x: dataset.valid[0], labels: dataset.valid[1]})))
        print("\tTrain accuracy: {:.3f}".format(accuracy.eval({x: dataset.train[0], labels: dataset.train[1]})))

        # Training cycle
        for epoch in range(training_epochs):
            avg_loss = 0.
            # Loop over all batches
            for train_x, train_label in dataset.get_train_batches():
                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([train_step, loss], feed_dict={x: train_x, labels: train_label})
                # Compute average loss
                avg_loss += l * len(train_x)
            # Display logs per epoch step
            print("Epoch:", '%d' % (epoch + 1), "loss=", "{:.10f}".format(avg_loss / len(dataset.train[0])))
            print("\tValidaton accuracy: {:.3f}".format(accuracy.eval({x: dataset.valid[0], labels: dataset.valid[1]})))
            print("\tTrain accuracy: {:.3f}".format(accuracy.eval({x: dataset.train[0], labels: dataset.train[1]})))


if __name__ == "__main__":
    main()
