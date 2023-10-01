import time
import re
import pandas as pd
import spacy
import sqlalchemy
from sqlalchemy import text
import pymysql
import sys
import select
from collections import Counter
import clustering
import scraper
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*cryptography', )
#nltk.download('stopwords')
#nltk.download('punkt')

# get vectors
def get_model(df):
	print('Getting vectors for messages...')
	model = Doc2Vec(vector_size=150, window=10, min_count=1, epochs=100)
	documents = []
	for i, row in df.iterrows():
		documents.append(TaggedDocument(words=row['content'].split(), tags=[i]))
	model.build_vocab(documents)  # 'documents' is your list of TaggedDocument objects
	model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
	document_vectors = [model.infer_vector(doc.words) for doc in documents]
	df['document_vecs'] = document_vectors
	return [model, df]
	
def storing(df):
	print('Storing to database...')
	pymysql.install_as_MySQLdb()
	engine = sqlalchemy.create_engine("mysql+mysqldb://root:Dsci-560@localhost/information_clustering")
	
	#df.to_csv('result_data.csv')
	df.to_sql('reddit_result', engine, if_exists = 'replace', index=True)
	print('Successfully updated!')

def fetch():
	print('Getting data from database...')
	pymysql.install_as_MySQLdb()
	engine = sqlalchemy.create_engine("mysql+mysqldb://root:Dsci-560@localhost/information_clustering")    
	with engine.connect() as conn:
		result = conn.execute(text("SELECT * FROM reddit_result")).fetchall()
	return result
	
if __name__ == "__main__":
	#pd.set_option('display.max_colwidth', None)
	if sys.argv[1] != 'quit' and sys.argv[1].isnumeric():
		while True:
			print('Updating... Please type the word quit and press enter if you want to stop. (The data would be automatically updated if no input in next 5 secs)')
			time.sleep(5)
			ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
			if ready_to_read:
				user = sys.stdin.readline().strip()
				if user == 'quit':
					break
			else:
				# print('System sleeping... Please type your keyword if you want to check the results.')
				df = scraper.get_data()
				#df = pd.DataFrame(df)
				model, df = get_model(df)
				df = clustering.kmeans_clustering(df)
				storing(df)
				print('System sleeping... Please type the word quit and press enter if you want to stop after sleeping.')
				time.sleep(60*int(sys.argv[1]))
				ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
				if ready_to_read:
					user = sys.stdin.readline().strip()
					if user == 'quit':
						break	
	if not sys.argv[1].isnumeric():
		stopwords = set(['new', 'people', 'researchers', 'size', 'and', 'the', 'scientists'])
		search = sys.argv[1:]
		search.append('tech')
		df = fetch()
		df = pd.DataFrame(df)
		#df, cluster_df = clustering.kmeans_clustering(df)
		model, df = get_model(df)
		inferred_vector = model.infer_vector(search)
		#similar_docs = model.docvecs.most_similar([inferred_vector], topn=1)[0]
		similar_doc_index = model.dv.most_similar([inferred_vector], topn=1)[0][0]
		cluster = df.iloc[similar_doc_index][['cluster']].iloc[0]
		cluster_df = df.groupby(['cluster']).agg({'Keyword':list, 'content':list}).reset_index()
		cluster_df['Keyword'] = cluster_df['Keyword'].apply(lambda x: [i for i in x if i not in stopwords])
		cluster_df['Keyword'] = cluster_df['Keyword'].apply(lambda x: Counter(x).most_common(5))
		display = cluster_df[cluster_df['cluster'] == cluster][['Keyword','content']]
		print(display.to_string(header=False))
		

