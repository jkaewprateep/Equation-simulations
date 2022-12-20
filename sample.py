import tensorflow as tf

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
n_sample = 2048
n_loop = 10
sample_x_scales = tf.range( start=0, limit=n_sample * ( n_loop + 1 ), delta=1, dtype=tf.int32, name='x-axis' )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Function / Class
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def create_sine_data(n = 2048, n_loop = 10 ):
	pi = 3.141592653589793
	start = 0.0
	stop = 1.0 * 2.0 * pi
	num = n
	
	x = tf.linspace( start, stop, num, name='linspace', axis=0 )
	y_temp = 3 * tf.math.sin( x )
	y_1 = [ ]
	
	for i in range( n_loop + 1 ):
		y_1 = tf.concat( [ y_temp, y_1 ], axis=0 )
	
	escape_sine = tf.random.normal(
		shape=( n * ( n_loop + 1 ), ),
		mean=0.0,
		stddev=0.15 * tf.math.abs( y_1, name='abs' ),
		dtype=tf.dtypes.float32,
		seed=32,
		name=None
	)
	
	return y_1
	
def create_upper_slope_data( n = 2048, n_time = 10, shift=1, reverse=False ) :

	start = 0.0
	stop = 4
	num = n * ( n_time + 1 )

	x = tf.linspace( start, stop, num, name='linspace', axis=0 )
	x = tf.concat([tf.zeros([ shift, ]), x[:-shift]], axis=0)

	if reverse == True :
		x = - x * 100
	else :
		x = x * 100
	
	y = tf.ones([x.shape[0], ]) * 3.2
	z = tf.zeros([x.shape[0], ])
	x = tf.where(tf.math.greater_equal(x, y), z, x)
	x = tf.where(tf.math.less_equal(x, -y), z, x)
	
	return x

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Graph plottings / Display
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

y1 = create_sine_data( n_sample, n_loop )
plt.plot( sample_x_scales, y1, label="line 1" )

y2 = create_upper_slope_data( n_sample, n_loop, 1, False )
plt.plot( sample_x_scales, y2, label="line 2" )

y3 = create_upper_slope_data( n_sample, n_loop, 1024, True )
plt.plot( sample_x_scales, y3, label="line 3" )

y4 = create_upper_slope_data( n_sample, n_loop, 2048, False )
plt.plot( sample_x_scales, y4, label="line 3" )

y5 = create_upper_slope_data( n_sample, n_loop, 3072, True )
plt.plot( sample_x_scales, y5, label="line 3" )

y6 = create_upper_slope_data( n_sample, n_loop, 4096, False )
plt.plot( sample_x_scales, y6, label="line 3" )

plt.show()
