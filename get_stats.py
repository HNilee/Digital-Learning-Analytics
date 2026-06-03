import pandas as pd

# Load only necessary columns to save memory
cols = [
    'total_learning_hours',
    'daily_app_minutes',
    'engagement_consistency',
    'gamification_engagement',
    'forum_posts',
    'video_completion_pct',
    'skill_pre_score',
    'content_recommendations_followed'
]

df = pd.read_csv('digital_learning_analytics_100k.csv', usecols=cols)
stats = df.describe().loc[['mean', 'std']]
print(stats.to_json())
