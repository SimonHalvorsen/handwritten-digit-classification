import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Location where the data is saved locally
DATA_DIR = '/tmp/data'

# number of steps in the gradient descent approach
# could have used more sophisticated methods to decide when to stop
NUM_STEPS = 1000

# number of examaples in each learning step
MINIBATCH_SIZE = 100


# downloads dataset and saves it locally(DATA_DIR)
# one_hot: writes categorical variables in a one-hot vector format where the vector
# is all-zero apart from one element
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# x: placeholder for images of size 784 (28x28 pixels unrolled into a single vector)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

# elements representing the true labels
y_true = tf.placeholder(tf.float32, [None, 10])

# representing the predicted values
y_pred = tf.matmul(x, W)

# measuring similarity for the model (loss function).
# cross entropy is a natural choice when the model outputs class probabilities
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

# using gradient descent optimization with a learning rate of 0.5 to minimize the loss function
# learning rate controls how fast our gradient descent optimizer shits model weights to reduce overall loss
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:

    # initializes all variables
    sess.run(tf.global_variables_initializer())

    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        #feed_dict supplies the placeholder elements with values
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # feeds the model with images it has never seen
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))
