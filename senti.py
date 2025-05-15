import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import re
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

def prepro(data):
    pattern='\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    df=pd.DataFrame({'user_messages':messages})

    users = []
    messages = []
    for i in df['user_messages']:
        entry = re.split('([\w\W]+?):\s', i)
        if entry[1:]:
            users.append(entry[1])
            messages.append(''.join(entry[2]))
        else:
            users.append('grp_notification')
            messages.append(entry[0])
    df['users']=users
    df['messages']=messages
    df.drop(columns='user_messages',inplace=True)
    hashmap = defaultdict(list)

    # Populate the defaultdict
    for key, value in zip(df['users'], df['messages']):
        if value!='<Media omitted>\n':
            hashmap[key].append(value)
    del hashmap['grp_notification']


    data_for_df = [{'user': key, 'message': message} for key, messages in hashmap.items() for message in messages]

    # Create DataFrame from the list of dictionaries
    df_result = pd.DataFrame(data_for_df)
    return df_result

def sentimentA(df_result):

# VADER
    sia= SentimentIntensityAnalyzer()
# Run the Polarity Score on the Whole dataset
    res={}
    for i,row in tqdm(df_result.iterrows(),total=len(df_result)):
        messg=row['message']
        user=row['user']
        res[user]=sia.polarity_scores(messg)

    vaders=pd.DataFrame(res).T
    vaders=vaders.reset_index().rename(columns={'index':'user'})
    vaders=vaders.merge(df_result,how='left')

    graph = defaultdict(list)

# Populate the defaultdict
    for key, value in zip(vaders['user'], vaders['compound']):
        graph[key].append(value)

    for i, j in graph.items():
        graph[i] = sum(j)/len(j)
    df_graph = pd.DataFrame(list(graph.items()), columns=['user', 'average_compound'])
    return df_graph
