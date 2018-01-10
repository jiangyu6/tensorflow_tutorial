import tensorflow as tf
'''
# create a graph
a=tf.constant([1,2,3,4,5,6],shape=[2,3],name='a')
b=tf.constant([1,2,3,4,5,6],shape=[3,2],name='b')
c=tf.matmul(a,b)
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
'''


#with tf.device('/gpu:0'):
# using a single GPU on a multi-GPU system
with tf.device('/device:GPU:2'):
  a=tf.constant([1,2,3,4,5,6],shape=[2,3],name='a')
  b=tf.constant([1,2,3,4,5,6],shape=[3,2],name='b')
c=tf.matmul(a,b)
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))


# allocate very little memory, and then extend GPU memory as SEssions get run and more memory is needed. 
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.Session(config=config,...)

# only allocate 40% memory of each GPU
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
session=tf.Session(config=config,..)



# using a single GPU on a multi-GPU system
with tf.device('/device:GPU:2'):
  a=tf.constant([1,2,3,4,5,6],shape=[2,3],name='a')
  b=tf.constant([1,2,3,4,5,6],shape=[3,2],name='b')
c=tf.matmul(a,b)
# let tf to automatically choose an existing and supported divece to run in case the specified one don't exit
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
print(sess.run(c))


# use multiple GPUs
c=[]
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a=tf.constant([1,2,3,4,5,6],shape=[2,3])
    b=tf.constant([1,2,3,4,5,6],shape=[3,2])
    c.append(tf.matmul(a,b))
with tf.device('/cpu:0'):
  sum=tf.add_n(c)
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(sum))



