// popup.js

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  
  // Use configuration from config.js
  const API_KEY = CONFIG.API_KEY;
  const API_URL = CONFIG.API_URL;

  // Utility function to show loading state
  function showLoading(message) {
    return `<div class="loading">${message}</div>`;
  }

  // Utility function to get sentiment class
  function getSentimentClass(sentiment) {
    switch(sentiment) {
      case '1': return 'sentiment-positive';
      case '-1': return 'sentiment-negative';
      default: return 'sentiment-neutral';
    }
  }

  // Utility function to get sentiment label
  function getSentimentLabel(sentiment) {
    switch(sentiment) {
      case '1': return 'Positive 😊';
      case '-1': return 'Negative 😞';
      default: return 'Neutral 😐';
    }
  }

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      
      outputDiv.innerHTML = `
        <div class="section-title">📹 YouTube Video Analysis</div>
        <div class="video-id">${videoId}</div>
        ${showLoading("Fetching comments...")}
      `;

      try {
        const comments = await fetchComments(videoId);
        if (comments.length === 0) {
          outputDiv.innerHTML += '<div class="error">No comments found for this video.</div>';
          return;
        }

        outputDiv.innerHTML = outputDiv.innerHTML.replace(showLoading("Fetching comments..."), 
          `<div class="success">✅ Fetched ${comments.length} comments</div>${showLoading("Performing sentiment analysis...")}`
        );

        const predictions = await getSentimentPredictions(comments);

        if (predictions) {
          // Process the predictions to get sentiment counts and sentiment data
          const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
          const sentimentData = [];
          const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);
          
          predictions.forEach((item, index) => {
            sentimentCounts[item.sentiment]++;
            sentimentData.push({
              timestamp: item.timestamp,
              sentiment: parseInt(item.sentiment)
            });
          });

          // Compute metrics
          const totalComments = comments.length;
          const uniqueCommenters = new Set(comments.map(comment => comment.authorId)).size;
          const totalWords = comments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(word => word.length > 0).length, 0);
          const avgWordLength = (totalWords / totalComments).toFixed(1);
          const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
          
          // Normalize the average sentiment score to a scale of 0 to 10
          const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(1);

          // Remove loading message and add results
          outputDiv.innerHTML = outputDiv.innerHTML.replace(showLoading("Performing sentiment analysis..."), "");

          // Add the Comment Analysis Summary section
          outputDiv.innerHTML += `
            <div class="section">
              <div class="section-title">📊 Comment Analysis Summary</div>
              <div class="metrics-container">
                <div class="metric">
                  <div class="metric-title">Total Comments</div>
                  <div class="metric-value">${totalComments.toLocaleString()}</div>
                </div>
                <div class="metric">
                  <div class="metric-title">Unique Users</div>
                  <div class="metric-value">${uniqueCommenters.toLocaleString()}</div>
                </div>
                <div class="metric">
                  <div class="metric-title">Avg Length</div>
                  <div class="metric-value">${avgWordLength} words</div>
                </div>
                <div class="metric">
                  <div class="metric-title">Sentiment Score</div>
                  <div class="metric-value">${normalizedSentimentScore}/10</div>
                </div>
              </div>
            </div>
          `;

          // Add the Sentiment Distribution section
          outputDiv.innerHTML += `
            <div class="section">
              <div class="section-title">📈 Sentiment Distribution</div>
              <div class="chart-container" id="chart-container">
                ${showLoading("Generating sentiment chart...")}
              </div>
            </div>`;

          // Fetch and display the pie chart
          await fetchAndDisplayChart(sentimentCounts);

          // Add the Sentiment Trend Graph section
          outputDiv.innerHTML += `
            <div class="section">
              <div class="section-title">📉 Sentiment Trend Over Time</div>
              <div class="trend-graph-container" id="trend-graph-container">
                ${showLoading("Generating trend analysis...")}
              </div>
            </div>`;

          // Fetch and display the sentiment trend graph
          await fetchAndDisplayTrendGraph(sentimentData);

          // Add the Word Cloud section
          outputDiv.innerHTML += `
            <div class="section">
              <div class="section-title">☁️ Comment Word Cloud</div>
              <div class="wordcloud-container" id="wordcloud-container">
                ${showLoading("Generating word cloud...")}
              </div>
            </div>`;

          // Fetch and display the word cloud
          await fetchAndDisplayWordCloud(comments.map(comment => comment.text));

          // Add the top comments section
          outputDiv.innerHTML += `
            <div class="section">
              <div class="section-title">💬 Top 25 Comments</div>
              <ul class="comment-list">
                ${predictions.slice(0, 25).map((item, index) => `
                  <li class="comment-item">
                    <div class="comment-text">${index + 1}. ${item.comment}</div>
                    <span class="comment-sentiment ${getSentimentClass(item.sentiment)}">
                      ${getSentimentLabel(item.sentiment)}
                    </span>
                  </li>`).join('')}
              </ul>
            </div>`;
        }
      } catch (error) {
        console.error("Analysis error:", error);
        outputDiv.innerHTML += `<div class="error">❌ Error during analysis: ${error.message}</div>`;
      }
    } else {
      outputDiv.innerHTML = '<div class="error">❌ This is not a valid YouTube video URL.</div>';
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < CONFIG.MAX_COMMENTS) {
        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await response.json();
        
        if (!response.ok) {
          throw new Error(data.error?.message || 'Failed to fetch comments');
        }
        
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp: timestamp, authorId: authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      throw error;
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      const result = await response.json();
      if (response.ok) {
        return result;
      } else {
        throw new Error(result.error || 'Error fetching predictions');
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
      throw error;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const chartContainer = document.getElementById('chart-container');
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch chart image');
      }
      
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.alt = "Sentiment Distribution Chart";
      
      chartContainer.innerHTML = '';
      chartContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching chart image:", error);
      document.getElementById('chart-container').innerHTML = '<div class="error">Failed to generate chart</div>';
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const wordcloudContainer = document.getElementById('wordcloud-container');
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch word cloud image');
      }
      
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.alt = "Comment Word Cloud";
      
      wordcloudContainer.innerHTML = '';
      wordcloudContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching word cloud image:", error);
      document.getElementById('wordcloud-container').innerHTML = '<div class="error">Failed to generate word cloud</div>';
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const trendGraphContainer = document.getElementById('trend-graph-container');
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch trend graph image');
      }
      
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.alt = "Sentiment Trend Over Time";
      
      trendGraphContainer.innerHTML = '';
      trendGraphContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching trend graph image:", error);
      document.getElementById('trend-graph-container').innerHTML = '<div class="error">Failed to generate trend graph</div>';
    }
  }
});