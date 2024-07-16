import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.metrics import det_curve
from utils.test_config import test_configs

genuine_distances = np.load(test_configs.results_dir + 'genuine_distances_{}.npy'.format(test_configs.enrolment_samples), allow_pickle=True)
impostor_distances = np.load(test_configs.results_dir + 'impostor_distances_{}.npy'.format(test_configs.enrolment_samples), allow_pickle=True)


fontsizebigtitle = 16
fontsizetitle = 14
fontsizelegend = 10
fontsizeaxis = 12
fontsizeticks = 9



line_kwargs = {}
ticks = [0.01, 0.1, 0.5, 0.9, 0.99]  # [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
tick_locations = sp.stats.norm.ppf(ticks)
tick_labels = ['{:.0%}'.format(s) if (100 * s).is_integer() else '{:.1%}'.format(s) for s in ticks]
plt.xticks(tick_locations, labels=tick_labels, fontsize=fontsizeticks)
plt.xlim(-3, 3)
plt.yticks(tick_locations, labels=tick_labels, fontsize=fontsizeticks)
plt.ylim(-3, 3)

plt.plot([tick_locations[0], tick_locations[0]], [-3.2, 3.2], color='grey', linestyle='dotted', label = 'FRR at 1% FAR')
plt.plot([tick_locations[1], tick_locations[1]], [-3.2, 3.2], color='grey', linestyle='dashed', label = 'FRR at 10% FAR')
plt.plot([-3.2, 3.2], [-3.2, 3.2], color='k', linestyle='solid', label = 'EER: FAR = FRR')


plt.xlabel("False Acceptance Rate (%)", fontsize=fontsizeaxis)
plt.ylabel("False Rejection Rate (%)", fontsize=fontsizeaxis)
plt.grid()
specific_results = np.concatenate((np.ravel(genuine_distances), np.ravel(impostor_distances)))

labels = [0 for x in range(len(np.ravel(genuine_distances)))] + [1 for x in range(len(np.ravel(impostor_distances)))]

fpr, fnr, threshold = det_curve(labels, specific_results)


line_, = plt.plot(sp.stats.norm.ppf(fpr), sp.stats.norm.ppf(fnr), linewidth = 1.75, label = 'TypeFormer, E = 5', **line_kwargs)


# plt.title('DET Curves', fontsize = fontsizetitle)
# plt.axes().set_aspect('equal')
plt.legend(loc='upper right', fontsize=fontsizelegend)# title = "Performance Comparison: EER [%]")
plt.savefig('DET_curves.pdf')
# plt.close()
