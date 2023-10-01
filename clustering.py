from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
def kmeans_clustering(df):
	stopwords = set(['new', 'people', 'researchers', 'size', 'and', 'the', 'scientists'])
	num_clusters = 6
	kmeans = KMeans(n_clusters=num_clusters)
	# Step 1: Convert String to List of Floats
	#df['document_vecs'] = df['document_vecs'].apply(lambda x: [float(i) for i in x.strip("[]").split()])
	df['Keyword'] = df['Keyword'].apply(lambda x: x.strip("[]"))
	# Step 2: Convert List of Floats to NumPy Array
	#df['document_vecs'] = df['document_vecs'].apply(lambda x: np.array(x))
	vectors = np.vstack(df['document_vecs'])
	df['cluster'] = kmeans.fit_predict(vectors)
	return df
	
if __name__ == "__main__":
	df = pd.read_csv('result_data_with vectors.csv')
	kmeans_clustering(df)
