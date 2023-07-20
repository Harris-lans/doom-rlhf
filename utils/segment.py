class Segment:
	def __init__(self):
		self._buffer = []

	def add(self, observation, action, log_prob, reward, value, done):
		"""Add experience to buffer.

		Args:
			observation (array): The observation.
			action (integer): The action.
			log_prob (float): The log probability of the action.
			reward (float): The reward.
			value (float): The estimated value of the state.
			done (integer): The done flag indicating if the episode terminated.
		"""
		self._buffer.append((observation, action, log_prob, reward, value, done))
	
	def __getitem__(self, index):
		return self._buffer[index]

	def __setitem__(self, index, value):
		self._buffer[index] = value

	def __len__(self):
		return len(self._buffer)
	
	def __iter__(self):
		self.index = 0
		return self

	def __next__(self):
		if self.index < len(self._buffer):
			result = self._buffer[self.index]
			self.index += 1
			return result
		else:
			raise StopIteration
