import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_net
import os


batch_size = 50
lr = 0.001
n_iterations = 10000

checkpoint_path = './'

data = input_data.read_data_sets('mnist_data', one_hot=True)

x_input = tf.placeholder(tf.float32, shape=[None, 784])
y_label = tf.placeholder(tf.float32, shape=[None, 10])
prob = tf.placeholder(tf.float32)

output = mnist_net.mnist_net(x_input, prob)
output_ = tf.nn.softmax(output)

# loss function and testing accuracy define
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_label))

pred = tf.equal(tf.argmax(output, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


# image generation function defined here
def gen_fool_img(start, target, step_s, steps):

    target_hot = np.zeros([1, 10])
    print(target_hot.shape)
    target_hot[0][target] = 1

    start_hot = np.zeros((1, 10))
    start_hot[0][start] = 1
    index = np.nonzero(data.test.labels[0:1000][:, start])[0]
    rand_num = np.random.randint(0, len(index))

    image_of_s = data.test.images[index[rand_num]]
    label_of_s = data.test.labels[index[rand_num]]
    image_of_s = np.reshape(image_of_s, (1, 784))
    print(label_of_s)
    loss_gen = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target_hot)
    d = tf.gradients(loss_gen, x_input)

    img_gen = tf.stop_gradient(x_input - tf.sign(d) * step_s / steps)
    img_gen = tf.clip_by_value(img_gen, 0, 1)
    original_image = np.reshape(image_of_s, (28, 28))

    show = plt.figure(1, (15., 15.))
    row = 10
    col = 3
    grid = ImageGrid(show, 111,
                     nrows_ncols=(row, col),
                     axes_pad=0.5,
                     )

    for j in range(steps):

        deriv = sess.run(d, feed_dict={x_input: image_of_s, prob: 1.})
        img_gen_r = sess.run(img_gen, feed_dict={x_input: image_of_s, prob: 1.})

        img = np.reshape(img_gen_r, (28, 28))
        image_of_s = np.reshape(img_gen_r, (1, 784))

        img_noise = np.reshape(deriv, (28, 28))

        # save generated images
        # img2 = img_noise * 255
        # img2 = np.uint8(img2)
        # img2 = Image.fromarray(img2)
        # img2 = img2.convert("L")
        # img2.save('./imgs/' + 'result_noise' + str(j) + '.bmp', 'bmp')
        #
        # img = img * 255
        # img = np.uint8(img)
        # img = Image.fromarray(img)
        # img = img.convert("L")
        # img.save('./imgs/' + 'result' + str(j) + '.bmp', 'bmp')

        acc_gen = sess.run(accuracy,
                           feed_dict={x_input: image_of_s, y_label: target_hot, prob: 1.})

        # plot the result and save figure
        grid[0+j*3].imshow(original_image)
        grid[0+j*3].set_title('Label: {0}' \
                          .format(start))

        grid[1+j*3].imshow(img_noise)
        grid[1+j*3].set_title('Adversarial \nNoise')

        grid[2+j*3].imshow(img)
        grid[2+j*3].set_title('Label: {0} \nAccuracy: {1}' \
                          .format(target, acc_gen))

    plt.savefig('result.png')


saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()

sess = tf.Session(config=config)
sess.run(init)

# training of the benchmark network
for i in range(n_iterations):
    batch_x_input, batch_y_label = data.train.next_batch(batch_size)

    sess.run(optimizer, feed_dict={x_input: batch_x_input, y_label: batch_y_label, prob: 1.})

    if i % 100 == 0:
        loss_train = sess.run(loss, feed_dict={x_input: batch_x_input, y_label: batch_y_label, prob: 1.})
        acc = sess.run(accuracy, feed_dict={x_input: batch_x_input, y_label: batch_y_label, prob: 1.})
        print('Iter: %d\tTraining Accuracy= %.5f\tTrain Loss= %.6f' % ((i*batch_size), acc, loss_train))

# test 200 numbers
test_acc = sess.run(accuracy, feed_dict={x_input: data.test.images[:200], y_label: data.test.labels[:200], prob: 1.})
print("Testing Accuracy:", test_acc)

# fool image generation function
gen_fool_img(2, 6, 0.8, 10)

print('generation successful!')

# save model
checkpoint_name = os.path.join(checkpoint_path, 'mnist_model' + '.ckpt')
saver.save(sess, checkpoint_name)

sess.close()
