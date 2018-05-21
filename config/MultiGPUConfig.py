#coding:utf-8

import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
import h5py as h5
import time
import threading
import re


MOVING_AVERAGE_DECAY = 0.9999

TOWER_NAME = 'tower'


def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.
	Note that this function provides a synchronization point across all towers.
	Args:
	    tower_grads: List of lists of (gradient, variable) tuples. The outer list
	    is over individual gradients. The inner list is over the gradient
	    calculation for each tower.
	Returns:
	    List of pairs of (gradient, variable) where the gradient has been averaged
	    across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads


def tower_loss(scope, loss_op):
	tf.add_to_collection('losses', loss_op)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	tf.add_n(tf.get_collection('losses'), name='total_loss')

	# Assemble all of the losses for the current tower only.
	losses = tf.get_collection('losses', scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
		# session. This helps the clarity of presentation on tensorboard.
		loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
		tf.summary.scalar(loss_name, l)

	return total_loss


class MultiGPUConfig(object):

	def __init__(self):
		self.lib = ctypes.cdll.LoadLibrary("./release/Base.so")
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.testHead.argtypes = [ctypes.c_void_p]
		self.lib.testTail.argtypes = [ctypes.c_void_p]
		self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.test_flag = False
		self.in_path = None
		self.out_path = None
		self.bern = 0
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 0
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.001
		self.lmbda = 0.000
		self.log_on = 1
		self.exportName = None
		self.importName = None
		self.export_steps = 0
		self.opt_method = "SGD"
		self.optimizer = None
		self.test_link_prediction = False
		self.test_triple_classification = False
		self.early_stopping_rounds = None
		self.train_subset = None
		self.num_gpus = 1
	def init(self):
		self.trainModel = None
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path, len(self.in_path) * 2))
			if self.train_subset:
				self.lib.setTrainSubset(ctypes.create_string_buffer(self.train_subset, len(self.train_subset) * 2))
				self.local_entities = list()
				self.entity2id = dict()
				with open(os.path.join(self.in_path, self.train_subset, 'entity2id.txt')) as entity2id:
					entity2id.next()
					for l in entity2id:
						e, _ = l.split()
						self.local_entities.append(e)
				with open(os.path.join(self.in_path, 'entity2id.txt')) as entity2id:
					entity2id.next()
					for l in entity2id:
						e, n = l.split()
						self.entity2id[e] = int(n)
			self.lib.setBern(self.bern)
			self.lib.setWorkThreads(self.workThreads)
			self.lib.randReset()
			self.lib.importTrainFiles()
			self.relTotal = self.lib.getRelationTotal()
			self.entTotal = self.lib.getEntityTotal()
			self.trainTotal = self.lib.getTrainTotal()
			self.testTotal = self.lib.getTestTotal()
			self.validTotal = self.lib.getValidTotal()
			self.batch_size = self.lib.getTrainTotal() / self.nbatches
			self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
			self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
			self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
			self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
			self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
			self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
		if self.test_link_prediction:
			self.lib.importTestFiles()
			self.test_h = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
			self.test_t = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
			self.test_r = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
			self.test_h_addr = self.test_h.__array_interface__['data'][0]
			self.test_t_addr = self.test_t.__array_interface__['data'][0]
			self.test_r_addr = self.test_r.__array_interface__['data'][0]
		if self.test_triple_classification:
			self.lib.importTestFiles()
			self.lib.importTypeFiles()

			self.test_pos_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_pos_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_pos_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_neg_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_neg_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_neg_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
			self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
			self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
			self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
			self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
			self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]

			self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
			self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
			self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
			self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
			self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
			self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]

	def get_ent_total(self):
		return self.entTotal

	def get_rel_total(self):
		return self.relTotal

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_in_path(self, path):
		self.in_path = path

	def set_out_files(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern

	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim

	def set_ent_dimension(self, dim):
		self.ent_size = dim

	def set_rel_dimension(self, dim):
		self.rel_size = dim

	def set_train_times(self, times):
		self.train_times = times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_margin(self, margin):
		self.margin = margin

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path, steps = 0):
		self.exportName = path
		self.export_steps = steps

	def set_export_steps(self, steps):
		self.export_steps = steps

	def set_early_stopping_rounds(self, rounds):
		self.early_stopping_rounds = rounds

	def set_train_subset(self, subset):
		self.train_subset = subset;

	def set_num_gpus(self, gpus):
		self.num_gpus = gpus

	def sampling(self):
		self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

	def save_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, self.exportName)

	def restore_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, self.importName)


	def export_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.save(self.sess, self.exportName)
				else:
					self.saver.save(self.sess, path)

	def import_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.restore(self.sess, self.importName)
				else:
					self.saver.restore(self.sess, path)

	def get_parameter_lists(self):
		return self.trainModel.parameter_lists

	def get_parameters_by_name(self, var_name):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					return self.sess.run(self.trainModel.parameter_lists[var_name])
				else:
					return None

	def get_parameters(self, mode = "numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = self.get_parameters_by_name(var_name)
			else:
				res[var_name] = self.get_parameters_by_name(var_name).tolist()
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def save_embeddings(self):
		print("Saving parameters to h5 store ...")
		# .json to .h5
		with h5.File(self.out_path[:-4] + 'h5', 'a') as store:
			with self.graph.as_default():
				with self.sess.as_default():
					for var_name in self.get_parameter_lists():
						if var_name[:3] == 'rel':
							if var_name in store:
								store.pop(var_name)
							embeddings = self.get_parameters_by_name(var_name)
							store.create_dataset(var_name, data=embeddings)
						else:
							if var_name not in store:
								embeddings = np.zeros((len(self.entity2id), self.ent_size), dtype=np.float32)
								store.create_dataset(var_name, data=embeddings)

						params = self.get_parameters_by_name(var_name)
						for i, param in enumerate(params):
							global_id = self.entity2id[self.local_entities[i]]
							store[var_name][global_id] = param

	def set_parameters_by_name(self, var_name, tensor):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default(), tf.device('/cpu:0'):
			self.global_step = tf.contrib.framework.get_or_create_global_step()
			config = tf.ConfigProto(allow_soft_placement=True,
						log_device_placement=False)
			self.sess = tf.Session(config=config)
			with self.sess.as_default():
				tower_grads = []
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", initializer = initializer):
					self.trainModel = self.model(config = self, init_op=False)
					# CPU Operations
					with tf.name_scope("input"):
						self.trainModel.input_def()
					with tf.name_scope('embedding'):
						self.trainModel.embedding_def()
					for i in range(self.num_gpus):
						with tf.device('/gpu:%d' % i):
							with tf.name_scope('%s_%d' % ('tower', i)) as scope:
								with tf.name_scope("loss"):
									self.trainModel.loss_def()
								with tf.name_scope("predict"):
									self.trainModel.predict_def()
								
								self.loss_op = tower_loss(scope, self.trainModel.loss)

								tf.get_variable_scope().reuse_variables()
								
								if self.optimizer != None:
									pass
								elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
									self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-20)
								elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
									self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
								elif self.opt_method == "Adam" or self.opt_method == "adam":
									self.optimizer = tf.train.AdamOptimizer(self.alpha)
								else:
									self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
								grads = self.optimizer.compute_gradients(self.loss_op)
								tower_grads.append(grads)

				grads = average_gradients(tower_grads)
				apply_gradient_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)
				# Track the moving averages of all trainable variables.
				variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
				variables_averages_op = variable_averages.apply(tf.trainable_variables())

				self.train_op = tf.group(apply_gradient_op, variables_averages_op)

				self.saver = tf.train.Saver()

				self.summary_op = tf.summary.merge_all()
				self.summary_writer = tf.summary.FileWriter(self.out_path[:self.out_path.rindex('/')], graph=self.sess.graph)

				self.sess.run(tf.initialize_all_variables())

	def load_parameters(self, embeddings_path=None):
		if embeddings_path is None:
			if self.out_path is None or os.path.exists(self.out_path[:-4] + 'h5') is False:
				return
			else:
				embeddings_path = self.out_path[:-4] + 'h5'

		print("Loading parameters from h5 store ...")
		with h5.File(embeddings_path, 'r') as store:
			with self.graph.as_default():
				with self.sess.as_default():
					for var_name in self.trainModel.parameter_lists:
						tensor = store[var_name].value
						var = self.trainModel.parameter_lists[var_name]
						if var_name[:3] == 'ent':
							if self.train_subset:
								update_indices = []
								updates = np.empty((self.entTotal, self.ent_size))
								for n in xrange(self.entTotal):
									global_id = self.entity2id[self.local_entities[n]]
									update_indices.append(n)
									updates[n] = tensor[global_id]
								tf.scatter_update(var, update_indices, updates).eval()
							else:
								var.assign(tensor).eval()
						else:
							var.assign(tensor).eval()

	def train_step(self, train_op, loss_op, global_step, feed_dict):
		_, loss, step = self.sess.run([train_op, loss_op, global_step], feed_dict)
		return loss, step

	def test_step(self, test_h, test_t, test_r):
		feed_dict = {
			self.trainModel.predict_h: test_h,
			self.trainModel.predict_t: test_t,
			self.trainModel.predict_r: test_r,
		}
		predict = self.sess.run(self.trainModel.predict, feed_dict)
		return predict

	def train(self, train_op, loss_op, summary_writer, summary_op, global_step, epoch, is_chief=False):
		reports = 0
		cum_loss = 0.0
		nbatches = self.nbatches
		start = time.time()
		for local_step in range(nbatches):
	   		self.sampling()
			feed_dict = {
				self.trainModel.batch_h: self.batch_h,
				self.trainModel.batch_t: self.batch_t,
				self.trainModel.batch_r: self.batch_r,
				self.trainModel.batch_y: self.batch_y
			}
			loss, step = self.train_step(train_op, loss_op, global_step, feed_dict)
			cum_loss += loss

			if is_chief:
				if self.exportName != None and local_step >= self.export_steps * (reports+1):
					reports += 1
					summary = self.sess.run(summary_op, feed_dict)
					self.summary_writer.add_summary(summary, global_step=step)
					self.summary_writer.flush()

		mean_loss = cum_loss/float(self.nbatches)

		if is_chief and self.log_on:
			print("Epoch. {:d}, Step: {:d}, Loss: {:.3f} Elapsed: {:.3f} sec".format(epoch, step, mean_loss, time.time() - start))

		return mean_loss

	def run(self):
		best_loss = None
		stopping_step = 0
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				for epoch in range(self.train_times):
					loss = self.train(self.train_op, self.loss_op, self.summary_writer, self.summary_op, self.global_step, epoch, is_chief=True)

					if self.exportName != None:
						self.save_tensorflow()

					if self.early_stopping_rounds:
						if best_loss is None or loss < best_loss:
							best_loss = loss
							stopping_step = 0
						else:
							stopping_step += 1
						if stopping_step >= self.early_stopping_rounds:
							break
				if self.exportName != None:
					self.save_tensorflow()
				if self.out_path != None:
					self.save_parameters(self.out_path)
					if self.train_subset:
						self.save_embeddings()

	def test(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if self.test_link_prediction:
					total = self.lib.getTestTotal()
					for times in range(total):
						self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
						res = self.test_step(self.test_h, self.test_t, self.test_r)
						self.lib.testHead(res.__array_interface__['data'][0])

						self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
						res = self.test_step(self.test_h, self.test_t, self.test_r)
						self.lib.testTail(res.__array_interface__['data'][0])
						if self.log_on:
							print times
					self.lib.test_link_prediction()
				if self.test_triple_classification:
					self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
					res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
					res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
					self.lib.getBestThreshold(res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

					self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)

					res_pos = self.test_step(self.test_pos_h, self.test_pos_t, self.test_pos_r)
					res_neg = self.test_step(self.test_neg_h, self.test_neg_t, self.test_neg_r)
					self.lib.test_triple_classification(res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])


	def predict(self, h, t, r, n=10):
		if h is None:
			for i in xrange(self.entTotal):
				self.test_h[i] = i
		else:
			self.test_h[:] = h
		if t is None:
			for i in xrange(self.entTotal):
				self.test_t[i] = i
		else:
			self.test_t[:] = t
		self.test_r[:] = r
		scores = self.test_step(self.test_h, self.test_t, self.test_r)
		top_n = np.argpartition(scores, n)[:n]
		sorted_top_n = top_n[scores[top_n].argsort()]
		top_n_score = scores[sorted_top_n]
		for i in range(sorted_top_n.shape[0]):
			print sorted_top_n[i], top_n_score[i]
