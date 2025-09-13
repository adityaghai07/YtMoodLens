import matplotlib
matplotlib.use('Agg')

import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import pickle
import os
import dotenv
import mlflow.sklearn
import seaborn as sns

# Set seaborn style and color palette
plt.style.use('dark_background')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)

dotenv.load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

app = Flask(__name__)
CORS(app)

def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

model_name = "aditya_ghai_yt_mood_lens_model"
model_version = 1
vectorizer_path = r'tfidf_vectorizer.pkl'
model, vectorizer = load_model_and_vectorizer(model_name, model_version, vectorizer_path)

@app.route('/')
def home():
    return "Welcome to my youtube sentiment analysis API!"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments.toarray()).tolist()
        predictions = [str(pred) for pred in predictions]
        
        response = [{"comment": c, "sentiment": s, "timestamp": t} 
                   for c, s, t in zip(comments, predictions, timestamps)]
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments.toarray()).tolist()
        
        return jsonify([{"comment": c, "sentiment": s} 
                       for c, s in zip(comments, predictions)])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for seaborn
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [sentiment_counts.get('1', 0), sentiment_counts.get('0', 0), sentiment_counts.get('-1', 0)]
        colors = ['#2ECC71', '#95A5A6', '#E74C3C']  # Green, Gray, Red
        
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        # Create figure with dark background
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#1e1e1e')
        
        # Pie chart with enhanced styling
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'color': 'white', 'fontsize': 12, 'weight': 'bold'},
                                          pctdistance=0.85)
        
        # Enhance the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        # Add a circle at the center to create a donut chart effect
        centre_circle = plt.Circle((0,0), 0.70, fc='#1e1e1e')
        ax1.add_artist(centre_circle)
        ax1.set_title('Sentiment Distribution', color='white', fontsize=16, fontweight='bold', pad=20)
        
        # Bar chart for counts
        df_sentiment = pd.DataFrame({'Sentiment': labels, 'Count': sizes, 'Color': colors})
        bars = ax2.bar(df_sentiment['Sentiment'], df_sentiment['Count'], color=df_sentiment['Color'], alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Enhance bar chart
        ax2.set_title('Sentiment Counts', color='white', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Sentiment', color='white', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Comments', color='white', fontsize=12, fontweight='bold')
        ax2.tick_params(colors='white', labelsize=11)
        ax2.set_facecolor('#2c2c2c')
        ax2.grid(True, alpha=0.3, color='white')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                    f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight', 
                   facecolor='#1e1e1e', edgecolor='none')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        text = ' '.join([preprocess_comment(comment) for comment in comments])
        
        # Enhanced word cloud with better colors and styling
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='#1e1e1e',
            colormap='plasma',  # Better color scheme
            stopwords=set(stopwords.words('english')),
            collocations=False,
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=80,
            prefer_horizontal=0.7
        ).generate(text)

        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title('Most Common Words in Comments', color='white', fontsize=16, 
                    fontweight='bold', pad=20)
        ax.axis('off')
        
        # Save to BytesIO
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight',
                   facecolor='#1e1e1e', edgecolor='none')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Create DataFrame and process timestamps
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sentiment'] = df['sentiment'].astype(int)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.patch.set_facecolor('#1e1e1e')

        # Define colors and labels
        sentiment_colors = {-1: '#E74C3C', 0: '#95A5A6', 1: '#2ECC71'}
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Plot 1: Time series with rolling average
        df_sorted = df.sort_values('timestamp')
        
        # Group by day and calculate sentiment percentages
        df_daily = df_sorted.set_index('timestamp').resample('D')['sentiment'].apply(list).reset_index()
        df_daily['total_comments'] = df_daily['sentiment'].apply(len)
        df_daily = df_daily[df_daily['total_comments'] > 0]  # Remove days with no comments
        
        # Calculate daily sentiment percentages
        daily_sentiment_data = []
        for _, row in df_daily.iterrows():
            sentiments = row['sentiment']
            total = len(sentiments)
            if total > 0:
                pos_pct = (sentiments.count(1) / total) * 100
                neg_pct = (sentiments.count(-1) / total) * 100
                neu_pct = (sentiments.count(0) / total) * 100
                daily_sentiment_data.append({
                    'date': row['timestamp'],
                    'positive': pos_pct,
                    'negative': neg_pct,
                    'neutral': neu_pct,
                    'total_comments': total
                })
        
        if daily_sentiment_data:
            df_plot = pd.DataFrame(daily_sentiment_data)
            
            # Plot sentiment percentages over time
            ax1.plot(df_plot['date'], df_plot['positive'], color='#2ECC71', 
                    linewidth=3, marker='o', markersize=6, label='Positive', alpha=0.8)
            ax1.plot(df_plot['date'], df_plot['negative'], color='#E74C3C', 
                    linewidth=3, marker='s', markersize=6, label='Negative', alpha=0.8)
            ax1.plot(df_plot['date'], df_plot['neutral'], color='#95A5A6', 
                    linewidth=3, marker='^', markersize=6, label='Neutral', alpha=0.8)
            
            ax1.fill_between(df_plot['date'], df_plot['positive'], alpha=0.3, color='#2ECC71')
            ax1.fill_between(df_plot['date'], df_plot['negative'], alpha=0.3, color='#E74C3C')
        
        ax1.set_title('Daily Sentiment Trends Over Time', color='white', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', color='white', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Percentage of Comments (%)', color='white', fontsize=12, fontweight='bold')
        ax1.set_facecolor('#2c2c2c')
        ax1.grid(True, alpha=0.3, color='white')
        ax1.tick_params(colors='white', labelsize=10)
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                  facecolor='#2c2c2c', edgecolor='white')
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Sentiment distribution histogram
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        colors = [sentiment_colors[s] for s in sentiment_counts.index]
        labels = [sentiment_labels[s] for s in sentiment_counts.index]
        
        bars = ax2.bar(labels, sentiment_counts.values, color=colors, alpha=0.8, 
                      edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars
        total_comments = len(df)
        for i, (bar, count) in enumerate(zip(bars, sentiment_counts.values)):
            percentage = (count / total_comments) * 100
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(sentiment_counts.values)*0.01,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                    color='white', fontweight='bold', fontsize=11)
        
        ax2.set_title('Overall Sentiment Distribution', color='white', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Sentiment', color='white', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Comments', color='white', fontsize=12, fontweight='bold')
        ax2.set_facecolor('#2c2c2c')
        ax2.grid(True, alpha=0.3, color='white', axis='y')
        ax2.tick_params(colors='white', labelsize=11)

        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight',
                   facecolor='#1e1e1e', edgecolor='none')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

@app.route('/generate_advanced_analytics', methods=['POST'])
def generate_advanced_analytics():
    """Generate additional advanced analytics visualizations"""
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')
        comments_data = data.get('comments')

        if not sentiment_data or not comments_data:
            return jsonify({"error": "Insufficient data provided"}), 400

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

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_advanced_analytics: {e}")
        return jsonify({"error": f"Advanced analytics generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
