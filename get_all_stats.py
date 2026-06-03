import pandas as pd

cols = [
    'total_learning_hours',
    'daily_app_minutes',
    'engagement_consistency',
    'gamification_engagement',
    'forum_posts',
    'video_completion_pct',
    'skill_pre_score',
    'content_recommendations_followed',
    'score_improvement',
    'total_interactions'
]

df = pd.read_csv('digital_learning_analytics_100k.csv')

# Pre-calculate score_improvement and total_interactions if they are not in CSV (they were engineered in notebook)
if 'score_improvement' not in df.columns:
    df['score_improvement'] = df['skill_post_score'] - df['skill_pre_score']
if 'total_interactions' not in df.columns:
    df['total_interactions'] = df['forum_posts'] + df['peer_review_given'] + df['content_recommendations_followed']

stats = df[cols].describe().loc[['mean', 'std']]
print(stats.to_json())
