import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

def set_plot_details(axs, xticks=None, yticks=None, xlabel=None, xlabel_fontsize=None, ylabel=None, 
	ylabel_fontsize=None, title=None, title_fontsize=None, grid=False, legend=None):
	

	"""
	Function to modularize a few plot details.
	"""
	
	if xticks is not None:
		axs.set_xticks(xticks)

	if yticks is not None:
		axs.set_yticks(yticks)

	if xlabel is not None:
		axs.set_xlabel(xlabel, fontsize=xlabel_fontsize)

	if ylabel is not None:
		axs.set_ylabel(ylabel, fontsize=ylabel_fontsize)

	if title is not None:
		axs.set_title(title, fontsize=title_fontsize)

	if grid:
		axs.grid()

	if legend is not None:
		axs.legend(legend)


class Stat_Analizer_TLC():

	"""
	Class designed to implement an analizer with a few statistical tests.
	"""

    def __init__(self, alpha, sample_size, n_sample):
        self.alpha = alpha
        self.sample_size = sample_size
        self.n_sample = n_sample

    def sampling(self, data):
        sample_size = self.sample_size
        index = np.random.choice(range(0, data.shape[0]), size=sample_size)
        sample_df = data.iloc[index]

        return sample_df

    def sample_means(self, data):

    	"""
    	Function designed to collect samples from a dataset within Central Limit Theorem hypothesis
    	"""
    	
        n_sample = self.n_sample
        sample_size = self.sample_size
        samp_means = []

        for sample in range(n_sample):
            sample = self.sampling(data)
            samp_means.append(sample.mean())

        return samp_means

    def kurtosis_check(self, dist, title):
        kurt = st.kurtosis(dist, fisher=False)
        print('Distribuição de ' + title + ': Curtose = {0:.3f}'.format(kurt))

    def perform_t_test_1_sample(self, dist, pop_dist, title):

    	"""
		Performs t-test to compare one sample with the population it was drawn from.
    	"""

        alpha = self.alpha
        sample_size = self.sample_size

        t_test = st.ttest_1samp(dist, np.array(pop_dist).mean())

        if t_test[1] < 1-alpha:
            result = ' A amostra é estatisticamente diferente da população! (rejeitamos a hipótese nula).'
        else:
            result = 'As amostras não é estatisticamente diferente da população! (não rejeitamos a hipótese nula).'

        interval = st.t.interval(alpha=alpha, df=st.kurtosis(pop_dist, fisher=False), loc=np.array(
            pop_dist).mean(), scale=np.array(pop_dist).std()/np.sqrt(sample_size))

        print('>>> ' + title+':')
        print('Média: {0:.3f}'.format(np.array(dist).mean()))
        print(
            'Intervalo de confiança: {0:.3f} <------------> {1:.3f}'.format(interval[0], interval[1]))
        print('p-valor = {0:.3f}'.format(t_test[1]))
        print('Resultado: '+result)

    def perform_t_test_2_sample(self, dist1, dist2, title1, title2):

    	"""
		Performs t-test comparing two independent samples from the same population.
    	"""

        alpha = self.alpha
        sample_size = self.sample_size

        t_test = st.ttest_ind(dist1, dist2)

        if t_test[1] < 1-alpha:
            result = ' As amostra são estatisticamente diferentes entre si! (rejeitamos a hipótese nula).'
        else:
            result = 'As amostras não são estatisticamente diferentes entre si! (não rejeitamos a hipótese nula).'

        interval1 = st.t.interval(alpha=alpha, df=st.kurtosis(dist1, fisher=False), loc=np.array(
            dist1).mean(), scale=np.array(dist1).std()/np.sqrt(sample_size))
        interval2 = st.t.interval(alpha=alpha, df=st.kurtosis(dist2, fisher=False), loc=np.array(
            dist2).mean(), scale=np.array(dist2).std()/np.sqrt(sample_size))

        print('Analisando ' + title1 + ' contra '+title2+':')
        print('>>> ' + title1+':')
        print('Média: {0:.3f}'.format(np.array(dist1).mean()))
        print(
            'Intervalo de confiança: {0:.3f} <------------> {1:.3f}'.format(interval1[0], interval1[1]))
        print('>>> ' + title2+':')
        print('Média: {0:.3f}'.format(np.array(dist2).mean()))
        print(
            'Intervalo de confiança: {0:.3f} <------------> {1:.3f}'.format(interval2[0], interval2[1]))
        print('>>> Resultado:')
        print('p-valor = {0:.3f}'.format(t_test[1]))
        print('Conclusão: '+result)

