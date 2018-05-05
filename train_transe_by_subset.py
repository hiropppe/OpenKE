import config
import models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

con = config.Config()
con.set_in_path("./benchmarks/FB15K/")

con.set_work_threads(4)
con.set_train_times(100)
con.set_nbatches(20)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

con.set_export_files("./res/model.vec.tf", 0)
con.set_out_files("./res/embedding.vec.json")

# train0
con.set_train_subset("train0")
con.init()
con.set_model(models.TransE)
con.run()

# train1
con.set_train_subset("train1")
con.init()
con.set_model(models.TransE)
con.load_parameters()
con.run()

# train2
con.set_train_subset("train2")
con.init()
con.set_model(models.TransE)
con.load_parameters()
con.run()

# train3
con.set_train_subset("train3")
con.init()
con.set_model(models.TransE)
con.load_parameters()
con.run()

# train4
con.set_train_subset("train4")
con.init()
con.set_model(models.TransE)
con.load_parameters()
con.run()
