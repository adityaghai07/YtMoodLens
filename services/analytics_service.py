import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io


def generate_advanced_analytics_dashboard(sentiment_data, comments_data):
    """Generate comprehensive analytics dashboard with multiple visualizations."""
    # Create comprehensive analytics dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#1e1e1e')
    fig.suptitle('Advanced Comment Analytics Dashboard', color='white', fontsize=20, fontweight='bold')

    # 1. Sentiment vs Comment Length Analysis
    df = pd.DataFrame(sentiment_data)
    comment_lengths = [len(comment['text'].split()) for comment in comments_data]
    df['comment_length'] = comment_lengths

    sns.boxplot(data=df, x='sentiment', y='comment_length', ax=ax1)
    ax1.set_title('Comment Length by Sentiment', color='white', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment', color='white', fontsize=12)
    ax1.set_ylabel('Comment Length (words)', color='white', fontsize=12)
    ax1.set_facecolor('#2c2c2c')
    ax1.tick_params(colors='white')

    # 2. Hourly sentiment patterns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    hourly_sentiment = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)

    if not hourly_sentiment.empty:
        hourly_sentiment_pct = hourly_sentiment.div(hourly_sentiment.sum(axis=1), axis=0) * 100
        hourly_sentiment_pct.plot(kind='area', ax=ax2, alpha=0.7,
                                color=['#E74C3C', '#95A5A6', '#2ECC71'])
        ax2.set_title('Sentiment Distribution by Hour of Day', color='white', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour of Day', color='white', fontsize=12)
        ax2.set_ylabel('Percentage', color='white', fontsize=12)
        ax2.set_facecolor('#2c2c2c')
        ax2.tick_params(colors='white')
        ax2.legend(['Negative', 'Neutral', 'Positive'], loc='upper right')

    # 3. Comment velocity over time
    df_daily = df.set_index('timestamp').resample('D').size()
    df_daily.plot(ax=ax3, color='#3498DB', linewidth=3)
    ax3.set_title('Comment Velocity (Comments per Day)', color='white', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', color='white', fontsize=12)
    ax3.set_ylabel('Comments Count', color='white', fontsize=12)
    ax3.set_facecolor('#2c2c2c')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.3, color='white')

    # 4. Sentiment correlation with engagement metrics (if available)
    # For now, create a mock correlation heatmap
    engagement_data = np.random.rand(3, 3)  # Mock data
    labels = ['Likes', 'Replies', 'Mentions']
    sentiments = ['Negative', 'Neutral', 'Positive']

    sns.heatmap(engagement_data, annot=True, fmt='.2f', ax=ax4,
               xticklabels=labels, yticklabels=sentiments,
               cmap='RdYlGn', center=0.5)
    ax4.set_title('Sentiment vs Engagement Correlation', color='white', fontsize=14, fontweight='bold')
    ax4.tick_params(colors='white')

    plt.tight_layout()

    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight',
               facecolor='#1e1e1e', edgecolor='none')
    img_io.seek(0)
    plt.close()

    return img_io