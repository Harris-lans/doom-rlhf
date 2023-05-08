import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from utils.time import *
from utils.memory import Memory

class ActorNetwork(tf.keras.Model):
    def __init__(self, observation_shape, number_of_actions):
        super(ActorNetwork, self).__init__()

        self.network = tf.keras.Sequential();
        self.network.add(tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)))
        self.network.add(tf.keras.layers.Input(shape=observation_shape));
        self.network.add(tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu'))
        self.network.add(tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'))
        self.network.add(tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu'))
        self.network.add(tf.keras.layers.Flatten())
        self.network.add(tf.keras.layers.Dense(512, activation='relu'))
        self.network.add(tf.keras.layers.Dense(number_of_actions, activation='softmax', name='policy'))

    def call(self, input):
        return self.network(input)
    

class CriticNetwork(tf.keras.Model):
    def __init__(self, observation_shape):
        super(CriticNetwork, self).__init__()

        self.network = tf.keras.Sequential();
        self.network.add(tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)))
        self.network.add(tf.keras.layers.Input(shape=observation_shape))
        self.network.add(tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu'))
        self.network.add(tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'))
        self.network.add(tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu'))
        self.network.add(tf.keras.layers.Flatten())
        self.network.add(tf.keras.layers.Dense(512, activation='relu'))
        self.network.add(tf.keras.layers.Dense(1, activation='linear', name='value'))

    def call(self, input):
        return self.network(input)

class Ppo2Agent:
    def __init__(self, env, actor_model_path=None, critic_model_path=None, gamma=0.99, gae_lambda=0.95, epsilon=0.2, value_coef=0.5, entropy_coef=0.01, learning_rate=0.0003, training_epochs=10):
        self.env = env
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.training_epochs = training_epochs

        if actor_model_path:
            print(f"loading actor network from `{actor_model_path}` model...")
            self.actor = tf.keras.models.load_model(actor_model_path)
            print(f"successfully loaded actor network from `{actor_model_path}` model!")
        else:
            print('creating actor network...')
            self.actor = ActorNetwork(self.env.observation_space.shape, self.env.action_space.n)
            print('successfully created actor network!')
            
        if critic_model_path:
            print(f"loading critic network from `{critic_model_path}` model...")
            self.critic = tf.keras.models.load_model(critic_model_path)
            print(f"successfully loaded critic network from model `{critic_model_path}` model!")
        else:
            print('creating critic network...')
            self.critic = CriticNetwork(self.env.observation_space.shape)
            print('successfully created critic network!')

        # Creating optimizer for networks
        print('creating optimizer for actor network...')
        self.actor.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))
        print('successfully created optimizer for actor network!')
        print('creating optimizer for critic network...')
        self.critic.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))
        print('successfully created optimizer for critic network!')

        self.memory = Memory()

    def save_models(self, save_dir='models/'):
        print('saving models ...')
        self.actor.save(save_dir + f"actor_{current_timestamp_ms()}")
        self.critic.save(save_dir + f"critic_{current_timestamp_ms()}")
        print('successfully saved models!')
    
    def get_optimal_action(self, observation):
        state = tf.convert_to_tensor([observation])

        action_probabilities = self.actor(state)
        # action_probabilities = np.random.normal(action_probabilities, self.entropy_coef)
        distribution = tfp.distributions.Categorical(action_probabilities)
        
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        log_probability = log_probability.numpy()[0]
        value = value.numpy()[0]

        return action, log_probability, value
    
    def remember_experience(self, state, action, probability, value, reward, done):
        self.memory.remember(state, action, probability, value, reward, done)

    def num_of_experiences(self):
        return self.memory.num_experiences()
    
    def try_learning_from_experience(self, experience_batch_size):
        for _ in range(self.training_epochs):
            states_arr, actions_arr, old_probabilities_arr, values_arr, rewards_arr, dones_arr, num_batches = self.memory.generate_batches(experience_batch_size)

            values = values_arr
            advantage = np.zeros(len(rewards_arr), dtype=np.float32)

            for t in range(len(rewards_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr) - 1):
                    a_t += discount*(rewards_arr[k] + self.gamma * values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            for batch in num_batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(states_arr[batch])
                    old_probabilities = tf.convert_to_tensor(old_probabilities_arr[batch])
                    actions = tf.convert_to_tensor(actions_arr[batch])

                    probabilities = self.actor(states)
                    distributions = tfp.distributions.Categorical(probabilities)
                    new_probabilities = tf.cast(distributions.log_prob(actions), dtype=tf.float32)

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1)
                    
                    probabilities_ratio = tf.math.exp(new_probabilities - old_probabilities)
                    weighted_probabilities = advantage[batch] * probabilities_ratio
                    clipped_probabilities = tf.clip_by_value(probabilities_ratio, 1 - self.epsilon, 1 + self.epsilon)
                    weighted_clipped_probabilities = clipped_probabilities * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probabilities, weighted_clipped_probabilities)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    critic_loss = tf.keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_gradients = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_gradients = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(zip(actor_gradients, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_gradients, critic_params))

        self.memory.clear()