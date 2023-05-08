import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_data():
        job_data = pd.read_csv('Datasets/dice_com-job_us_sample.csv')
        job_data.dropna(inplace=True)
        job_data.reset_index(inplace=True)
        job_data['jobtitle'] = job_data['jobtitle'].str.lower()
        return job_data

def combine_data(data):
        data_recommend = data.loc[:,['joblocation_address','skills','company','shift','jobdescription']]
        data_recommend['combine'] = data_recommend[data_recommend.columns[0:4]].apply(
                                                                        lambda x: ','.join(x.dropna().astype(str)),axis=1)
        
        data_recommend = data_recommend.drop(columns=['joblocation_address','skills','company','shift','jobdescription'])
        return data_recommend
        
def transform_data(data_combine, data_plot):
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(data_combine['combine'])

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data_plot['jobdescription'])

        combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
        cosine_sim = cosine_similarity(combine_sparse, combine_sparse)
        
        return cosine_sim


def recommend_jobs(title, data, combine, transform):
        indices = pd.Series(data.index, index = data['jobtitle'])
        index = indices[title]



        sim_scores = list(enumerate(transform[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1].any(), reverse=True)
        sim_scores = sim_scores[1:11]


        job_indices = [i[0] for i in sim_scores]

        job_id = data['jobid'].iloc[job_indices]
        job_title = data['jobtitle'].iloc[job_indices]
        job_address = data['joblocation_address'].iloc[job_indices]

        recommendation_data = pd.DataFrame(columns=['Id','Name', 'Address'])

        recommendation_data['Id'] = job_id
        recommendation_data['Name'] = job_title
        recommendation_data['Address'] = job_address

        return recommendation_data

def results(job_name):
        job_name = job_name.lower()

        find_job = get_data()
        combine_result = combine_data(find_job)
        transform_result = transform_data(combine_result,find_job)

        if job_name not in find_job['jobtitle'].unique():
                return 'job not in Database'

        else:
                recommendations = recommend_jobs(job_name, find_job, combine_result, transform_result)
                return recommendations.to_dict('records')

