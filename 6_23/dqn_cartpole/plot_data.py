import pickle
import matplotlib.pyplot as plt
import numpy as np
import os



def plot_algo(data, name):
	mag_adv = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]

	for mean, std, algo in data:
		plt.plot(mag_adv, mean, label = algo)
		plt.fill_between(mag_adv, mean - std, mean + std, alpha = 0.5)
	plt.xlabel('adv_mag')
	plt.ylabel('reward')
	plt.title(name)
	plt.legend()
	if not os.path.exists('result_image'):
	    os.makedirs('result_image')
	path = 'result_image/' + str(name) + '.png'
	plt.savefig(path)
	# plt.show()

# matrix = [trial0, trial1, ..., trial9]
# trial0 = [adv_mag = 0.01, ..., adv_mag = 0.5]
van = []
samp = []
grad = []
pa = []
sgd = []
mag_adv = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]

van_temp = []
samp_temp = []
grad_temp = []
pa_temp = []
sgd_temp = []

# read data from pickle files and load to the matrices
for j in range(10):
	
	for i in range(len(mag_adv)):
		path = 'result_data/adv_mag=' + str(mag_adv[i]) + '/adv_test_mag_{}_iteration_{}.pickle'.format(mag_adv[i], j)
		# print(path)
		with open(path, 'rb') as f: data = pickle.load(f)
        
	        van_temp.append(data['rewardList_van'])
	        samp_temp.append(data['rewardList_samp'])
	        grad_temp.append(data['rewardList_grad'])
	        pa_temp.append(data['rewardList_pa'])
	        sgd_temp.append(data['rewardList_sgd'])
	       	
	      
	van_temp_cpy = list(van_temp)
	samp_temp_cpy = list(samp_temp)
	grad_temp_cpy = list(grad_temp)
	pa_temp_cpy = list(pa_temp)
	sgd_temp_cpy = list(sgd_temp)

	van.append(van_temp_cpy)
	samp.append(samp_temp_cpy)
	grad.append(grad_temp_cpy)
	pa.append(pa_temp_cpy)
	sgd.append(sgd_temp_cpy)
	del van_temp[:]
	del samp_temp[:]
	del grad_temp[:]
	del pa_temp[:]
	del sgd_temp[:]	


# print(len(van_temp))
# plot average
# avg = []
# avg.append(np.mean(van, axis = 0))
# avg.append(np.mean(samp, axis = 0))
# avg.append(np.mean(grad, axis = 0))
# avg.append(np.mean(pa, axis = 0))
# avg.append(np.mean(sgd, axis = 0))




# average across episodes
van_epi = np.mean(van, axis = 2)
samp_epi = np.mean(samp, axis = 2)
grad_epi = np.mean(grad, axis = 2)
pa_epi = np.mean(pa, axis = 2)
sgd_epi = np.mean(sgd, axis = 2)

algo_mean = []
algo_mean.append((np.mean(van_epi, axis = 0), np.std(van_epi, axis = 0, ddof = 1), "vanilla testing"))
algo_mean.append((np.mean(samp_epi, axis = 0), np.std(samp_epi, axis = 0, ddof = 1), "naive sampling"))
algo_mean.append((np.mean(grad_epi, axis = 0), np.std(grad_epi, axis = 0, ddof = 1), "gradient based attack"))
algo_mean.append((np.mean(pa_epi, axis = 0), np.std(pa_epi, axis = 0, ddof = 1), "arXiv paper attack"))
algo_mean.append((np.mean(sgd_epi, axis = 0), np.std(sgd_epi, axis = 0, ddof = 1), "gradient descent based attack"))
plot_algo(algo_mean, "reward vs adv_mag")

# algo_var = []
# algo_var.append((np.mean(van_epi, axis = 0),"vanilla testing"))
# algo_var.append((np.mean(samp_epi, axis = 0), "naive sampling"))
# algo_var.append((np.mean(grad_epi, axis = 0), "gradient based attack"))
# algo_var.append((np.mean(pa_epi, axis = 0), "arXiv paper attack"))
# algo_var.append((np.mean(sgd_epi, axis = 0), "gradient descent based attack"))

# plot_algo(algo_var, "variance")


