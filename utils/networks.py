import tensorflow as tf

def build_neural_networks_for_ppo_agent(observation_shape, action_shape, learning_rate=0.0003):
    input_state = tf.keras.layers.Input(shape=observation_shape)
    convolutional_layer_1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')(input_state)
    convolutional_layer_2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')(convolutional_layer_1)
    convolutional_layer_3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')(convolutional_layer_2)
    flatten_layer = tf.keras.layers.Flatten()(convolutional_layer_3)
    dense_layer = tf.keras.layers.Dense(512, activation='relu')(flatten_layer)
    
    critic_network_output_layer = tf.keras.layers.Dense(1, activation='linear', name='value')(dense_layer)
    actor_network_output_layer = tf.keras.layers.Dense(action_shape[0], activation='softmax', name='policy')(dense_layer)

    critic_network = tf.keras.models.Model(inputs=input_state, outputs=critic_network_output_layer)
    critic_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    actor_network = tf.keras.models.Model(inputs=input_state, outputs=actor_network_output_layer)
    actor_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return critic_network, actor_network