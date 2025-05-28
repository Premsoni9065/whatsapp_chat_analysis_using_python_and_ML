import pandas as pd
from collections import Counter
import emoji
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Basic stats
    num_messages = df.shape[0]
    words = [word for message in df['message'] for word in message.split()]
    num_media = df[df['message'] == '<Media omitted>'].shape[0]

    # Count links
    link_pattern = r'https?://\S+|www\.\S+'
    links = sum(len(re.findall(link_pattern, msg)) for msg in df['message'])

    return num_messages, len(words), num_media, links


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df_percent = (df['user'].value_counts(normalize=True) * 100).reset_index()
    df_percent.columns = ['name', 'percent']
    return x, df_percent


def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Remove media and notifications
    temp = df[~df['message'].str.contains('<Media omitted>')]

    # Combine all messages
    text = " ".join(message for message in temp['message'])

    # Generate wordcloud
    wc = WordCloud(width=500, height=500, min_font_size=10,
                   background_color='white', stopwords=set(stopwords.words('english')))
    wc.generate(text)
    return wc


def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Remove media and stopwords
    temp = df[~df['message'].str.contains('<Media omitted>')]
    words = []

    for message in temp['message']:
        words.extend([word.lower() for word in message.split()
                      if word.lower() not in stopwords.words('english')])

    return pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    return pd.DataFrame(Counter(emojis).most_common(), columns=['emoji', 'count'])


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period',
                                  values='message', aggfunc='count').fillna(0)
    return user_heatmap