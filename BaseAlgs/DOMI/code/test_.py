import tensorflow as tf


def test_gpu():
    m1 = tf.constant([[3, 3]])
    m2 = tf.constant([[2], [3]])
    product = tf.matmul(m1, m2)
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()

test_gpu()
