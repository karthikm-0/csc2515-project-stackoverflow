from google.cloud import bigquery
from google.cloud import storage
import os
import bq_helper
from bq_helper import BigQueryHelper
import pandas as pd

# SET UP AUTHENTICATION
path = 'C:\\Users\\karthikm\\Downloads\\glass-mirror-api-174106-65832b594e66.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
client = bigquery.Client()

# LOAD DATASET
'''so_dataset_ref = client.dataset('stackoverflow', project='bigquery-public-data')
print(type(so_dataset_ref))

so_dset = client.get_dataset(so_dataset_ref)
print(type(so_dset))

print([x.table_id for x in client.list_tables(so_dset)])

# We need two tables: tags and stackoverflow_posts
so_full = client.get_table(so_dset.table('tags'))
print(type(so_full))

print(so_full.schema)

schema_subset = [col for col in so_full.schema if col.name in ('id', 'tag_name', 'count')]
results = [x for x in client.list_rows(so_full, start_index=100, selected_fields=schema_subset, max_results=10)]
print(results)
'''

# QUERYING FOR MOST POPULAR TAGS
QUERY = (
    'SELECT tag_name '
    'FROM `bigquery-public-data.stackoverflow.tags` '
    'ORDER BY count DESC '
    'LIMIT 20'
)

query_job = client.query(QUERY)
rows = query_job.result()
tag_list_tuple = []

for row in rows:
    tag_list_tuple.append(row.tag_name)

#tag_list_tuple = ["vba", "na"]
tag_list_tuple = tuple(tag_list_tuple)

# QUERYING FOR MOST POPULAR TAGS
QUERY_TWO = (
    'SELECT title, body, tags '
    'FROM `bigquery-public-data.stackoverflow.posts_questions` '
    'WHERE tags IN {}'.format(tag_list_tuple)
)

query_job = client.query(QUERY_TWO)
rows = query_job.result()
print(type(rows))

df = rows.to_dataframe()
print(df.shape)
#df.to_csv('query.csv')