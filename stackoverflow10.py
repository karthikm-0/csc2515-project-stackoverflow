import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tags = pd.read_csv('Tags.csv', encoding="ISO-8859-1", dtype={'Tag': str})
questions = pd.read_csv('Questions.csv', encoding="ISO-8859-1")
#print(tags)

#count the 100 most frequent tags
n = 100
#print(tags.value_counts()[:n].index.tolist())


# Group by tags automatically
'''top_ten_tags = tags['Tag'].value_counts().nlargest(20).index.tolist()
print(top_ten_tags)

tags_filtered = tags[tags['Tag'].isin(top_ten_tags)]
print(tags_filtered)

tags_filtered_duplicates = tags_filtered[tags_filtered.duplicated(['Id'])]
print(tags_filtered_duplicates)'''

# Group by tags and questions (CUSTOM)
custom_tags_filter = ['javascript', 'java', 'c#', 'php', 'python', 'html', 'c++', 'ruby-on-rails', 'c']
tags_filtered = tags[tags['Tag'].isin(custom_tags_filter)]
#print(tags_filtered)

tags_filtered_duplicates = tags_filtered[tags_filtered.duplicated(['Id'])]
tags_filtered = tags_filtered.drop_duplicates(subset=['Id'], keep='first')
print(tags_filtered_duplicates)
print(tags_filtered)

questions_filtered = questions[questions['Id'].isin(tags_filtered['Id'])]
print(questions_filtered)

questions_tags_combined = questions_filtered.merge(tags_filtered)
print(questions_tags_combined.head(5))

questions_tags_combined.to_csv('top_nine_languages.csv', index=False)