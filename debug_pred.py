import joblib
import pandas as pd

model = joblib.load('xgboost_mooc_model.pkl')
base_data = joblib.load('base_data.pkl')

scaler_stats = {
    'total_learning_hours': {'mean': 44.954331, 'std': 40.708444},
    'daily_app_minutes': {'mean': 45.403519, 'std': 22.550576},
    'engagement_consistency': {'mean': 0.490421, 'std': 0.199641},
    'gamification_engagement': {'mean': 39.971342, 'std': 19.963302},
    'forum_posts': {'mean': 1.745746, 'std': 2.202564},
    'video_completion_pct': {'mean': 40.853429, 'std': 18.718410},
    'skill_pre_score': {'mean': 44.964121, 'std': 17.586624},
    'content_recommendations_followed': {'mean': 56.067160, 'std': 18.472460},
    'total_interactions': {'mean': 58.846994, 'std': 18.668226}
}

def get_prediction(hours, mins, consistency, gamification, forum, video, pre_score, recom):
    input_df = base_data.copy()
    
    total_interactions = forum + 0.95 + recom
    
    input_df['total_learning_hours'] = (hours - scaler_stats['total_learning_hours']['mean']) / scaler_stats['total_learning_hours']['std']
    input_df['daily_app_minutes'] = (mins - scaler_stats['daily_app_minutes']['mean']) / scaler_stats['daily_app_minutes']['std']
    input_df['engagement_consistency'] = (consistency - scaler_stats['engagement_consistency']['mean']) / scaler_stats['engagement_consistency']['std']
    input_df['gamification_engagement'] = (gamification - scaler_stats['gamification_engagement']['mean']) / scaler_stats['gamification_engagement']['std']
    input_df['forum_posts'] = (forum - scaler_stats['forum_posts']['mean']) / scaler_stats['forum_posts']['std']
    input_df['video_completion_pct'] = (video - scaler_stats['video_completion_pct']['mean']) / scaler_stats['video_completion_pct']['std']
    input_df['skill_pre_score'] = (pre_score - scaler_stats['skill_pre_score']['mean']) / scaler_stats['skill_pre_score']['std']
    input_df['content_recommendations_followed'] = (recom - scaler_stats['content_recommendations_followed']['mean']) / scaler_stats['content_recommendations_followed']['std']
    input_df['total_interactions'] = (total_interactions - scaler_stats['total_interactions']['mean']) / scaler_stats['total_interactions']['std']
    
    pred = model.predict(input_df)
    return pred[0]

# User's inputs
res = get_prediction(300, 70, 0.90, 60, 40, 92.18, 85, 91.16)
print(f"User Input Prediction: {res}")

# "Average" student
res_avg = get_prediction(44.95, 45.40, 0.49, 39.97, 1.74, 40.85, 44.96, 56.06)
print(f"Average Student Prediction: {res_avg}")
