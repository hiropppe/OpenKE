import config
import models

con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(False)
con.set_work_threads(4)
con.init()
con.set_model(models.TransE)
con.load_parameters('./res/embedding.vec.h5')

# con.predict(453, None, 37, n=10)
con.test()
