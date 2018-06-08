#coding:utf-8

import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
import h5py as h5
import sys
import time

from tensorflow.python.client import timeline
from threading import Thread
from tqdm import tqdm


class AsyncMultiGPUConfig(object):

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
		self.num_train_threads = 1
		self.per_process_gpu_memory_fraction = None
	def init(self):
		self.trainModel = None
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path, len(self.in_path) * 2))
			if self.train_subset:
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
		self.train_subset = subset
                if subset:
                        self.lib.setTrainSubset(ctypes.create_string_buffer(subset, len(subset) * 2))
                else:
                        self.lib.setTrainSubset("")

	def set_num_gpus(self, gpus):
		self.num_gpus = gpus

	def set_num_train_threads(self, threads):
		self.num_train_threads = threads

	def set_per_process_gpu_memory_fraction(self, fraction):
		self.per_process_gpu_memory_fraction = fraction

	def sampling(self, batch_h_addr, batch_t_addr, batch_r_addr, batch_y_addr):
		self.lib.sampling(batch_h_addr, batch_t_addr, batch_r_addr, batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

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
                store_path = self.out_path[:-4] + 'h5'
                print("Saving parameters to {:s}.".format(store_path))
		with h5.File(store_path, 'a') as store:
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
			if self.per_process_gpu_memory_fraction:
				config.gpu_options.per_process_gpu_memory_fraction = self.per_process_gpu_memory_fraction
			self.sess = tf.Session(config=config)
			with self.sess.as_default():
				self.gpu_ops = []
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", initializer = initializer):
					self.trainModel = self.model(config = self, init_op=False)
					with tf.name_scope("input"):
						self.trainModel.input_def()
					with tf.name_scope('embedding'):
						self.trainModel.embedding_def()
					for i in range(self.num_gpus):
						with tf.device('/gpu:%d' % i):
							with tf.name_scope('%s_%d' % ('tower', i)):
								with tf.name_scope("loss"):
									self.trainModel.loss_def()
								with tf.name_scope("predict"):
									self.trainModel.predict_def()
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
								grads_and_vars = self.optimizer.compute_gradients(self.trainModel.loss)
								self.train_op = self.optimizer.apply_gradients(grads_and_vars, self.global_step)
								self.gpu_ops.append((self.train_op, self.trainModel.loss))

								tf.get_variable_scope().reuse_variables()
				self.saver = tf.train.Saver()

				tf.summary.scalar('loss', self.trainModel.loss)
				self.summary_op = tf.summary.merge_all()
				self.summary_writer = tf.summary.FileWriter(self.out_path[:self.out_path.rindex('/')], graph=self.sess.graph)

                                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                #run_metadata = tf.RunMetadata()
				#self.sess.run(tf.initialize_all_variables(), options=run_options, run_metadata=run_metadata)
				self.sess.run(tf.initialize_all_variables())

                                #step_stats = run_metadata.step_stats
                                #tl = timeline.Timeline(step_stats)

                                #ctf = tl.generate_chrome_trace_format(show_memory=True, show_dataflow=True)

                                #with open('./timeline.json', 'w') as f:
                                #    f.write(ctf)


	def load_parameters(self, embeddings_path=None):
		if embeddings_path is None:
			if self.out_path is None or os.path.exists(self.out_path[:-4] + 'h5') is False:
				return
			else:
				embeddings_path = self.out_path[:-4] + 'h5'

                print("Loading parameters from {:s}.".format(embeddings_path))
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
		batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
		batch_h_addr = batch_h.__array_interface__['data'][0]
		batch_t_addr = batch_t.__array_interface__['data'][0]
		batch_r_addr = batch_r.__array_interface__['data'][0]
		batch_y_addr = batch_y.__array_interface__['data'][0]
		reports = 0
		cum_loss = 0.0
		ob = 0
		ob_threshold = 100000.0
		nbatches = self.nbatches
		start = time.time()
                if is_chief and self.log_on:
                        pbar = tqdm(range(nbatches))
                else:
                        pbar = range(nbatches)
		for local_step in pbar:
	   		self.sampling(batch_h_addr, batch_t_addr, batch_r_addr, batch_y_addr)
			feed_dict = {
				self.trainModel.batch_h: batch_h,
				self.trainModel.batch_t: batch_t,
				self.trainModel.batch_r: batch_r,
				self.trainModel.batch_y: batch_y
			}
			loss, step = self.train_step(train_op, loss_op, global_step, feed_dict)
			# TODO sometimes return OB loss ...
			if loss < ob_threshold:
				cum_loss += loss
				if is_chief:
					if self.exportName != None and local_step >= self.export_steps * (reports+1):
						reports += 1
						summary = self.sess.run(summary_op, feed_dict)
						self.summary_writer.add_summary(summary, global_step=step)
						self.summary_writer.flush()
			else:
			    ob += 1

                if ob < self.nbatches:
		        mean_loss = cum_loss/float(self.nbatches - ob)
		        if is_chief and self.log_on:
		                print("Epoch. {:d}, Step: {:d}, Loss: {:.5f}, OB Loss: {:d}, Elapsed: {:.3f} sec, Total Elapsed: {:.3f} sec"
				        .format(epoch, step, mean_loss, ob, time.time() - start, time.time() - self.train_start))
                else:
                        # too many OB loss ...
                        mean_loss = None
		        print("Epoch. {:d} Failed !! Step: {:d}, OB Loss: {:d}, Elapsed: {:.3f} sec, Total Elapsed: {:.3f} sec"
		                .format(epoch, step,  ob, time.time() - start, time.time() - self.train_start))

		return mean_loss

	def run(self):
                if self.train_subset:
		    print('Training begins ({:s}).'.format(self.train_subset))
                else:
		    print('Training begins.')
		print('  max epoch: {:d}'.format(self.train_times))
		print('  epoch length: {:d}'.format(self.nbatches))
		print('  batch size: {:d}'.format(self.batch_size))
		print('  batch seq size: {:d}'.format(self.batch_seq_size))
		print('  GPUs: {:d}'.format(self.num_gpus))
		print('  train threads: {:d}'.format(self.num_train_threads))
		print('  sampling threads: {:d}'.format(self.workThreads))
		print('  total entity: {:d}'.format(self.entTotal))
		print('  total relation: {:d}'.format(self.relTotal))
		print('  total train triple: {:d}'.format(self.trainTotal))
		print('  total test triple: {:d}'.format(self.testTotal))
		print('  total valid triple: {:d}'.format(self.validTotal))
		best_loss = None
		stopping_step = 0
		self.train_start = time.time()
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				for epoch in range(self.train_times):
					if self.num_train_threads == 1:
						loss = self.train(self.train_op, self.trainModel.loss, self.summary_writer, self.summary_op, self.global_step, epoch, is_chief=True)
					else:
						train_threads = []
						for i in range(self.num_train_threads):
							if i == 0:
								is_chief = True
							else:
								is_chief = False

							ops = self.gpu_ops[i % self.num_gpus]

							train_args = (ops[0], ops[1], self.summary_writer, self.summary_op, self.global_step, epoch, is_chief)
                                                        train_thread = ThreadWithReturnValue(name='train_thread_{:d}'.format(i), target=self.train, args=train_args)
							train_threads.append(train_thread)

						for tt in train_threads:
							tt.start()

						cum_loss = 0.0
						for tt in train_threads:
                                                        loss = tt.join()
                                                        if loss is not None:
							    cum_loss += loss
                                                        else:
                                                            sys.stderr.write('{:s} return None loss.\n'.format(tt.name))

						loss = cum_loss / float(self.num_train_threads)

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
						#if self.log_on:
						#	print times
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


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
		 args=(), kwargs={}, Verbose=None):
	Thread.__init__(self, group, target, name, args, kwargs, Verbose)
	self._return = None
    def run(self):
	if self._Thread__target is not None:
	    self._return = self._Thread__target(*self._Thread__args,
						**self._Thread__kwargs)
    def join(self):
	Thread.join(self)
	return self._return


