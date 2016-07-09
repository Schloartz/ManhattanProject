import numpy as np

import tensorflow as tf
import pandas
from sklearn import metrics

# Import data
data_x = np.genfromtxt("data/train_spectrogram.csv", delimiter=",")
data_x = np.log(data_x)
gender = np.genfromtxt("data/labels_train.csv", delimiter=",", dtype=None, names=True)
data_y = np.array(pandas.get_dummies(gender['x']))

test_x = np.genfromtxt("data/devel_spectrogram.csv", delimiter=",")
test_x = np.log(test_x)
gender_test = np.genfromtxt("data/labels_devel.csv", delimiter=",", dtype=None, names=True)
test_y = np.array(pandas.get_dummies(gender_test['x']))

# Mode: Train / Restore model
#mode = "train"
mode = "restore"

# Parameters
learning_rate = 0.01
training_epochs = 2500
display_step = 100
batch_size = 394/4

# Network Parameters
n_hidden_1 = 1024 # 1st layer num features
n_hidden_2 = 1024 # 2nd layer num features
n_hidden_3 = 1024 # 3rd layer num features
n_hidden_4 = 1024
n_input = 2048 # Agender data input (spectrogram: 2048 pixel)
n_classes = 2 # Agender data ouput (gender: m/w)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
   # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
   # layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])) #Hidden layer with RELU activation
    #layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4']))
    return tf.matmul(layer_1, _weights['out']) + _biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    #'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	#'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    #'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.3).minimize(cost)
# Initializing the variables
init = tf.initialize_all_variables()

# Initializing saver
saver = tf.train.Saver()

# Evaluation helpers
prediction = tf.argmax(pred, 1)
observation = tf.argmax(y, 1)

def next_batch(batch_size):
    """Return the next `batch_size` examples from this data set."""
    x = data_x
    y = data_y

    perm = np.arange(394)
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    return x[0:batch_size], y[0:batch_size]

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    if mode=="train": #Train the model
	    
	    for epoch in range(training_epochs):
	        avg_cost = 0.
	        # Fit training using batch data
	        batch_x, batch_y = next_batch(batch_size)
	        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
	        # Compute average loss
	        avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y})
	        # Display logs per epoch step
	        if epoch % display_step == 0:
	            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
	            # Test model
	            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
				# Calculate accuracy
	            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	            print("Acc_Test:", accuracy.eval({x: test_x, y: test_y}), "Acc_Train:", accuracy.eval({x: data_x, y: data_y}))

		# Save the variables to disk.
		save_path = saver.save(sess, "save/model.ckpt")
		print("Model saved in file: %s" % save_path)

    elif mode=="restore": #Load the model
		saver.restore(sess, "save/model.ckpt")
		print("Model restored.")
	
	
	# Final test of model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Acc_Test:", accuracy.eval({x: test_x, y: test_y}), "Acc_Train:", accuracy.eval({x: data_x, y: data_y}))
    predicted = prediction.eval({x: test_x})
    observed = observation.eval({y: test_y})
    print("Classification report:\n%s\n" % metrics.classification_report(observed, predicted))

    print("Optimization Finished!")
    