import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from wordcloud import WordCloud
import nltk
import io
from .helpers import preprocess_comment

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords


def generate_sentiment_chart(sentiment_counts):
    """Generate pie chart and bar chart for sentiment distribution."""
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
    centre_circle = plt.Circle((0, 0), 0.70, fc='#1e1e1e')
    ax1.add_artist(centre_circle)
    ax1.set_title('Sentiment Distribution', color='white', fontsize=16, fontweight='bold', pad=20)

    # Bar chart for counts
    df_sentiment = pd.DataFrame({'Sentiment': labels, 'Count': sizes, 'Color': colors})
    bars = ax2.bar(df_sentiment['Sentiment'], df_sentiment['Count'], color=df_sentiment['Color'],
                   alpha=0.8, edgecolor='white', linewidth=1.5)

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

    return img_io


def generate_wordcloud_image(comments):
    """Generate word cloud from comments."""
    if not comments:
        raise ValueError("No comments provided for word cloud generation")

    # Process comments and combine text
    processed_comments = []
    for comment in comments:
        try:
            processed = preprocess_comment(comment)
            if processed and processed.strip():  # Only add non-empty processed comments
                processed_comments.append(processed)
        except Exception as e:
            print(f"Error processing comment: {e}")
            continue

    if not processed_comments:
        raise ValueError("No valid text found after preprocessing comments")

    text = ' '.join(processed_comments)

    if not text or not text.strip():
        raise ValueError("Text is empty after preprocessing")

    try:
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
    except ValueError as e:
        if "empty" in str(e).lower() or "no" in str(e).lower():
            raise ValueError("Insufficient text content to generate word cloud")
        raise e

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

    return img_io


def generate_trend_chart(sentiment_data):
    """Generate trend graph for sentiment analysis over time."""
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

    return img_io