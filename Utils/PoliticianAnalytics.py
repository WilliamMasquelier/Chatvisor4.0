import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import logging
import traceback
from typing import List, Dict, Any
import seaborn as sns

from Classes.SimplifiedChatbots import ChatGPT, BigSummarizerGPT
from Classes.DocumentProcessor import DocumentProcessor, CitationProcessor
from Utils.Helpers import *

def generate_politician_analytics_output(articles_sorted: List[Dict], politician_name: str, general_folder: str, 
                                   region: str, political_party: str = None, language: str = 'English') -> str:
    """
    Generate political media analytics report with sentiment analysis and visualizations.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed and sorted articles.
        politician_name (str): Name of the politician being analyzed.
        general_folder (str): Base path for output files.
        region (str): Geographic region of interest.
        political_party (str, optional): Political party of the politician. Defaults to None.
        language (str, optional): Output language for the analytics report. Defaults to 'English'.
        
    Returns:
        str: Generated markdown content with political media analytics.
    """
    try:
        logging.info(f"Starting political media analytics generation for {politician_name}")
        
        # Ensure articles have sentiment analysis
        for article in articles_sorted:
            if 'tone' not in article or not article['tone']:
                logging.warning(f"Article missing tone analysis. Consider preprocessing articles first.")
                break
            if 'sentiment_score' not in article and 'sentiment score' not in article:
                logging.warning(f"Article missing sentiment score. Consider preprocessing articles first.")
                break
        
        # Standardize sentiment score key (some may use 'sentiment score' others 'sentiment_score')
        for article in articles_sorted:
            if 'sentiment score' in article and 'sentiment_score' not in article:
                article['sentiment_score'] = article['sentiment score']
        
        # Create DataFrame for analysis
        df = pd.DataFrame(articles_sorted)
        
        # Ensure date column is datetime
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        elif 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Set visualization style
        plt.style.use('ggplot')
        
        # Initialize charts dictionary
        charts = {}
        
        # Function to safely generate chart
        def generate_chart_safely(chart_name: str, chart_function, *args, **kwargs):
            try:
                logging.info(f"Generating chart: {chart_name}")
                if df.empty:
                    logging.warning(f"DataFrame is empty, skipping chart: {chart_name}")
                    return None
                chart = chart_function(*args, **kwargs)
                logging.info(f"Successfully generated chart: {chart_name}")
                return chart
            except Exception as e:
                logging.error(f"Error generating {chart_name}: {str(e)}")
                logging.error(traceback.format_exc())
                return None
        
        # Generate politician-specific charts
        chart_generators = {
            'media_outlet_pie': (generate_media_outlet_pie_chart, [df]),
            'media_outlet_tone': (generate_media_outlet_tone_chart, [df]),
            'overall_sentiment': (generate_political_sentiment_trend, [df, politician_name]),
            'narrative_distribution': (generate_narrative_distribution_chart, [df]),
            'narrative_sentiment': (generate_narrative_sentiment_chart, [df]),
            'policy_focus': (generate_policy_focus_chart, [df, politician_name]),
            'stakeholder_network': (generate_stakeholder_network_chart, [df, politician_name, general_folder]),
            'political_reputation': (generate_political_reputation_chart, [df, politician_name])
        }
        
        for chart_name, (func, args) in chart_generators.items():
            charts[chart_name] = generate_chart_safely(chart_name, func, *args)
        
        # Calculate statistics (only if we have data)
        if df.empty:
            logging.warning("DataFrame is empty, using default statistics")
            stats = {
                'total_articles': 0,
                'date_range': "No data available",
                'avg_sentiment': 0,
                'median_sentiment': 0
            }
        else:
            try:
                # Use sentiment_score instead of sentiment score for consistency
                sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment score'
                
                stats = {
                    'total_articles': len(df),
                    'date_range': f"from {df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}",
                    'avg_sentiment': df[sentiment_col].mean(),
                    'median_sentiment': df[sentiment_col].median()
                }
            except Exception as e:
                logging.error(f"Error calculating statistics: {str(e)}")
                stats = {
                    'total_articles': len(df),
                    'date_range': "Error calculating date range",
                    'avg_sentiment': 0,
                    'median_sentiment': 0
                }
        
        # Generate media outlet statistics
        try:
            media_outlet_stats = []
            sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment score'
            
            for outlet in df['media_outlet'].unique():
                try:
                    outlet_df = df[df['media_outlet'] == outlet]
                    media_outlet_stats.append({
                        'outlet': outlet,
                        'articles': len(outlet_df),
                        'avg_sentiment': outlet_df[sentiment_col].mean(),
                        'median_sentiment': outlet_df[sentiment_col].median()
                    })
                except Exception as e:
                    logging.error(f"Error processing outlet {outlet}: {str(e)}")
                    continue
            
            media_outlet_stats.sort(key=lambda x: x['articles'], reverse=True)
        except Exception as e:
            logging.error(f"Error generating media outlet statistics: {str(e)}")
            media_outlet_stats = []
        
        # Generate narrative statistics
        try:
            narrative_stats = []
            narrative_col = 'narrative_category' if 'narrative_category' in df.columns else None
            
            if narrative_col:
                for narrative in df[narrative_col].unique():
                    try:
                        narrative_df = df[df[narrative_col] == narrative]
                        narrative_stats.append({
                            'narrative': narrative,
                            'articles': len(narrative_df),
                            'avg_sentiment': narrative_df[sentiment_col].mean(),
                            'pos_percent': (narrative_df['tone'] == 'Positive').mean() * 100,
                            'neu_percent': (narrative_df['tone'] == 'Neutral').mean() * 100,
                            'neg_percent': (narrative_df['tone'] == 'Negative').mean() * 100
                        })
                    except Exception as e:
                        logging.error(f"Error processing narrative {narrative}: {str(e)}")
                        continue
                
                narrative_stats.sort(key=lambda x: x['articles'], reverse=True)
        except Exception as e:
            logging.error(f"Error generating narrative statistics: {str(e)}")
            narrative_stats = []
        
        # Generate final report with error handling for missing charts
        try:
            markdown_content = generate_politician_markdown_report(
                politician_name=politician_name,
                political_party=political_party,
                region=region,
                total_articles=stats['total_articles'],
                date_range=stats['date_range'],
                avg_sentiment=stats['avg_sentiment'],
                median_sentiment=stats['median_sentiment'],
                media_outlet_pie_chart=charts.get('media_outlet_pie', ''),
                media_outlet_tone_chart=charts.get('media_outlet_tone', ''),
                overall_sentiment_trend=charts.get('overall_sentiment', ''),
                narrative_distribution_chart=charts.get('narrative_distribution', ''),
                narrative_sentiment_chart=charts.get('narrative_sentiment', ''),
                policy_focus_chart=charts.get('policy_focus', ''),
                stakeholder_network_chart=charts.get('stakeholder_network', ''),
                political_reputation_chart=charts.get('political_reputation', ''),
                media_outlet_stats=media_outlet_stats,
                narrative_stats=narrative_stats,
                df=df,
                general_folder=general_folder,
                language=language
            )
        except Exception as e:
            logging.error(f"Error generating markdown report: {str(e)}")
            markdown_content = f"Error generating political media analytics report: {str(e)}"
        
        # Save the report
        try:
            political_analytics_path = f'{general_folder}/Outputs/PoliticianAnalysis/MediaAnalytics_{politician_name.replace(" ", "_")}.md'
            with open(political_analytics_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        except Exception as e:
            logging.error(f"Error saving political media analytics report: {str(e)}")
        
        logging.info("Political media analytics report generation completed")
        return markdown_content
    
    except Exception as e:
        logging.error(f"Error in generate_politician_analytics_output: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error generating political media analytics: {str(e)}"


def generate_political_sentiment_trend(df, politician_name):
    """Generate a line chart showing the sentiment trend over time."""
    # Ensure the date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Determine sentiment column name
    sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment score'
    
    # Group by month and calculate average sentiment
    monthly_sentiment = df.set_index('date').resample('M')[sentiment_col].mean().reset_index()
    
    # Create the sentiment trend chart
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sentiment['date'], monthly_sentiment[sentiment_col], marker='o', color='#1f77b4', linewidth=2)
    
    # Add a horizontal line at y=0 to indicate neutral sentiment
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Formatting
    plt.title(f'Sentiment Trend for {politician_name} over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Average Sentiment Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add annotations for major peaks and dips
    if len(monthly_sentiment) > 3:
        # Find the top 2 peaks and bottom 2 dips
        peaks = monthly_sentiment.nlargest(2, sentiment_col)
        dips = monthly_sentiment.nsmallest(2, sentiment_col)
        
        for _, row in peaks.iterrows():
            plt.annotate(f'{row[sentiment_col]:.2f}', 
                         xy=(row['date'], row[sentiment_col]),
                         xytext=(10, 10),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        for _, row in dips.iterrows():
            plt.annotate(f'{row[sentiment_col]:.2f}', 
                         xy=(row['date'], row[sentiment_col]),
                         xytext=(10, -20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Save the plot as base64
    image_base64 = save_plot_base64()
    plt.close()
    
    return image_base64


def generate_political_sentiment_analysis(df, politician_name):
    """
    Generate a comprehensive analysis of sentiment trends for a politician.
    
    Args:
        df (pd.DataFrame): DataFrame with the articles data
        politician_name (str): Name of the politician
        
    Returns:
        str: Markdown formatted sentiment analysis
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Determine sentiment column name
    sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment score'
    
    # Calculate monthly sentiment
    monthly_sentiment = df.set_index('date').resample('M')[sentiment_col].mean().reset_index()
    
    # Find peaks and dips in sentiment
    if len(monthly_sentiment) > 3:
        peaks = monthly_sentiment.nlargest(2, sentiment_col)
        dips = monthly_sentiment.nsmallest(2, sentiment_col)
    else:
        peaks = monthly_sentiment.head(1)
        dips = monthly_sentiment.tail(1)
    
    # Generate analysis for peak periods
    peak_analyses = []
    for _, row in peaks.iterrows():
        month_start = row['date']
        # Get articles from that month +/- 2 weeks
        month_articles = df[(df['date'] >= month_start - pd.Timedelta(days=14)) & 
                          (df['date'] <= month_start + pd.Timedelta(days=14))]
        
        # Prepare article data for analysis
        articles_data = []
        for _, article in month_articles.iterrows():
            # Get a preview of the content (first 300 chars)
            content_preview = article['content'][:300] + "..." if len(article['content']) > 300 else article['content']
            articles_data.append({
                "title": article.get('title', 'No title'),
                "date": article['date'].strftime('%Y-%m-%d'),
                "sentiment": article[sentiment_col],
                "content_preview": content_preview,
                "media_outlet": article.get('media_outlet', 'Unknown')
            })
        
        # Get analysis from ChatGPT
        chatbot = ChatGPT(
            system_prompt=f"You are a political media analyst specializing in sentiment analysis.",
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=800
        )
        
        prompt = f"""
Analyze these articles from {month_start.strftime('%B %Y')} about {politician_name} to explain why this period shows a peak in positive sentiment.
Focus on identifying the key events, statements, or developments that contributed to this positive coverage.

Articles data:
{json.dumps(articles_data, indent=2)}

Provide a concise 1-2 paragraph analysis explaining the factors behind this positive sentiment peak. Reference specific media outlets and events.
"""
        
        try:
            peak_analysis = chatbot.ask(prompt)
            peak_analyses.append({
                "date": month_start.strftime('%B %Y'),
                "sentiment": row[sentiment_col],
                "analysis": peak_analysis
            })
        except Exception as e:
            logging.error(f"Error generating peak analysis: {str(e)}")
            peak_analyses.append({
                "date": month_start.strftime('%B %Y'),
                "sentiment": row[sentiment_col],
                "analysis": f"Error analyzing this period: {str(e)}"
            })
    
    # Generate analysis for dip periods
    dip_analyses = []
    for _, row in dips.iterrows():
        month_start = row['date']
        # Get articles from that month +/- 2 weeks
        month_articles = df[(df['date'] >= month_start - pd.Timedelta(days=14)) & 
                          (df['date'] <= month_start + pd.Timedelta(days=14))]
        
        # Prepare article data for analysis
        articles_data = []
        for _, article in month_articles.iterrows():
            # Get a preview of the content (first 300 chars)
            content_preview = article['content'][:300] + "..." if len(article['content']) > 300 else article['content']
            articles_data.append({
                "title": article.get('title', 'No title'),
                "date": article['date'].strftime('%Y-%m-%d'),
                "sentiment": article[sentiment_col],
                "content_preview": content_preview,
                "media_outlet": article.get('media_outlet', 'Unknown')
            })
        
        # Get analysis from ChatGPT
        chatbot = ChatGPT(
            system_prompt=f"You are a political media analyst specializing in sentiment analysis.",
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=800
        )
        
        prompt = f"""
Analyze these articles from {month_start.strftime('%B %Y')} about {politician_name} to explain why this period shows a dip in sentiment (more negative coverage).
Focus on identifying the key events, controversies, or developments that contributed to this negative coverage.

Articles data:
{json.dumps(articles_data, indent=2)}

Provide a concise 1-2 paragraph analysis explaining the factors behind this negative sentiment dip. Reference specific media outlets and events.
"""
        
        try:
            dip_analysis = chatbot.ask(prompt)
            dip_analyses.append({
                "date": month_start.strftime('%B %Y'),
                "sentiment": row[sentiment_col],
                "analysis": dip_analysis
            })
        except Exception as e:
            logging.error(f"Error generating dip analysis: {str(e)}")
            dip_analyses.append({
                "date": month_start.strftime('%B %Y'),
                "sentiment": row[sentiment_col],
                "analysis": f"Error analyzing this period: {str(e)}"
            })
    
    # Generate overall trend analysis
    overall_chatbot = ChatGPT(
        system_prompt=f"You are a political media analyst specializing in sentiment trends.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1000
    )
    
    trend_data = monthly_sentiment.to_dict('records')
    
    overall_prompt = f"""
Analyze the sentiment trend for {politician_name} based on this monthly sentiment data:
{json.dumps(trend_data, indent=2)}

Also consider these analyses of peak sentiment periods:
{json.dumps([p['analysis'] for p in peak_analyses], indent=2)}

And these analyses of dip sentiment periods:
{json.dumps([d['analysis'] for d in dip_analyses], indent=2)}

Provide a comprehensive 2-3 paragraph analysis explaining the overall sentiment trajectory for {politician_name} over time. Identify patterns, trends, and potential factors influencing the media's perception. Be objective and balanced in your assessment.
"""
    
    try:
        overall_trend_analysis = overall_chatbot.ask(overall_prompt)
    except Exception as e:
        logging.error(f"Error generating overall trend analysis: {str(e)}")
        overall_trend_analysis = f"Error generating overall trend analysis: {str(e)}"
    
    # Build the markdown content
    markdown_content = "\n### Sentiment Evolution Analysis\n\n"
    
    # Add peak periods section
    markdown_content += "#### Peak Sentiment Periods\n"
    for peak in peak_analyses:
        markdown_content += f"**{peak['date']} (Sentiment: {peak['sentiment']:.2f})**\n\n{peak['analysis']}\n\n"
    
    # Add dip periods section
    markdown_content += "#### Dip Sentiment Periods\n"
    for dip in dip_analyses:
        markdown_content += f"**{dip['date']} (Sentiment: {dip['sentiment']:.2f})**\n\n{dip['analysis']}\n\n"
    
    # Add overall trend analysis
    markdown_content += "#### Overall Sentiment Trajectory\n\n"
    markdown_content += overall_trend_analysis
    
    return markdown_content


def generate_narrative_analysis(narrative_stats, politician_name):
    """
    Generate analysis of the narrative categories used to describe the politician.
    
    Args:
        narrative_stats (List): Statistics about narratives
        politician_name (str): Name of the politician
        
    Returns:
        str: Markdown formatted narrative analysis
    """
    if not narrative_stats:
        return "No narrative data available for analysis."
    
    # Use ChatGPT for narrative analysis
    chatbot = ChatGPT(
        system_prompt=f"You are a political media analyst specializing in narrative analysis.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1200
    )
    
    prompt = f"""
Analyze how {politician_name} is portrayed in the media based on these narrative categories:
{json.dumps(narrative_stats, indent=2)}

For each narrative, consider:
1. The frequency (number of articles)
2. The sentiment (positive, neutral, negative percentages and average sentiment)
3. The implications for {politician_name}'s public image

Provide a comprehensive analysis that:
1. Identifies the dominant narratives and their tone
2. Explains how these narratives shape public perception
3. Notes any patterns or contradictions in the coverage
4. Considers how these narratives might affect {politician_name}'s political standing

Format your response as a well-structured analysis of 3-4 paragraphs. Be balanced and objective in your assessment.
"""
    
    try:
        narrative_analysis = chatbot.ask(prompt)
        return narrative_analysis
    except Exception as e:
        logging.error(f"Error generating narrative analysis: {str(e)}")
        return f"Error generating narrative analysis: {str(e)}"


def generate_policy_analysis(df, politician_name):
    """
    Generate analysis of the politician's policy focus areas based on media coverage.
    
    Args:
        df (pd.DataFrame): DataFrame with the articles data
        politician_name (str): Name of the politician
        
    Returns:
        str: Markdown formatted policy analysis
    """
    # Check if we have policy categories
    if 'policy_category' not in df.columns:
        return "No policy data available for analysis."
    
    # Calculate statistics for each policy area
    policy_stats = []
    sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment score'
    
    for policy in df['policy_category'].unique():
        if policy == "No specific policy":
            continue
            
        policy_df = df[df['policy_category'] == policy]
        policy_stats.append({
            "policy": policy,
            "articles": len(policy_df),
            "avg_sentiment": policy_df[sentiment_col].mean(),
            "pos_percent": (policy_df['tone'] == 'Positive').mean() * 100,
            "neu_percent": (policy_df['tone'] == 'Neutral').mean() * 100,
            "neg_percent": (policy_df['tone'] == 'Negative').mean() * 100
        })
    
    # Sort by number of articles
    policy_stats.sort(key=lambda x: x['articles'], reverse=True)
    
    # Use ChatGPT for policy analysis
    chatbot = ChatGPT(
        system_prompt=f"You are a political media analyst specializing in policy analysis.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1200
    )
    
    prompt = f"""
Analyze the policy focus areas for {politician_name} based on media coverage statistics:
{json.dumps(policy_stats, indent=2)}

Provide a comprehensive analysis that:
1. Identifies the politician's primary policy focus areas according to media coverage
2. Analyzes how these policy areas are generally portrayed (sentiment analysis)
3. Discusses the implications of this policy focus distribution for {politician_name}'s political identity
4. Notes any surprising patterns or gaps in the policy coverage

Format your response as a well-structured analysis of 3-4 paragraphs. Be balanced and objective in your assessment.
"""
    
    try:
        policy_analysis = chatbot.ask(prompt)
        return policy_analysis
    except Exception as e:
        logging.error(f"Error generating policy analysis: {str(e)}")
        return f"Error generating policy analysis: {str(e)}"


def generate_stakeholder_analysis(df, politician_name, general_folder):
    """
    Generate analysis of the politician's relationships with stakeholders.
    
    Args:
        df (pd.DataFrame): DataFrame with the articles data
        politician_name (str): Name of the politician
        general_folder (str): Base path for output files
        
    Returns:
        str: Markdown formatted stakeholder analysis
    """
    # Try to load stakeholder data
    stakeholder_path = os.path.join(
        general_folder, "Outputs", "PoliticianAnalysis", "Stakeholders",
        f"StakeholderAnalysis_{politician_name.replace(' ', '_')}.json"
    )
    
    if os.path.exists(stakeholder_path):
        with open(stakeholder_path, 'r', encoding='utf-8') as f:
            stakeholder_data = json.load(f)
            
        # Prepare data for analysis
        stakeholders = stakeholder_data.get("stakeholders", [])
        categories = stakeholder_data.get("categories", [])
        
        # Count stakeholders by category
        category_counts = Counter([s['category'] for s in stakeholders])
        
        # Prepare category data
        category_data = []
        for category in categories:
            category_name = category['category']
            category_data.append({
                "category": category_name,
                "description": category.get('description', ''),
                "count": category_counts.get(category_name, 0),
                "stakeholders": [s for s in stakeholders if s['category'] == category_name][:5]  # Top 5 per category
            })
        
        # Use ChatGPT for stakeholder analysis
        chatbot = ChatGPT(
            system_prompt=f"You are a political network analyst specializing in stakeholder relationships.",
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1500
        )
        
        prompt = f"""
Analyze the stakeholder network for {politician_name} based on these relationship categories:
{json.dumps(category_data, indent=2)}

Provide a comprehensive analysis that:
1. Identifies the key relationship groups and their significance
2. Analyzes the most important individual stakeholders and their relationships with {politician_name}
3. Discusses the implications of this stakeholder network for {politician_name}'s political influence and operations
4. Notes any patterns or strategic insights about {politician_name}'s political relationships

Format your response as a well-structured analysis of 3-4 paragraphs. Be balanced and objective in your assessment.
"""
        
        try:
            stakeholder_analysis = chatbot.ask(prompt)
            return stakeholder_analysis
        except Exception as e:
            logging.error(f"Error generating stakeholder analysis: {str(e)}")
            return f"Error generating stakeholder analysis: {str(e)}"
    
    # If no stakeholder data found, create a simplified analysis from people entities
    else:
        # Extract people entities if available
        people_data = []
        for _, row in df.iterrows():
            if 'people' in row and isinstance(row['people'], list):
                for person in row['people']:
                    people_data.append({
                        "name": person.get('name', ''),
                        "role": person.get('role', ''),
                        "context": person.get('context', '')
                    })
        
        if not people_data:
            return "No stakeholder data available for analysis."
        
        # Count mentions for each person
        person_counts = Counter([p['name'] for p in people_data if p['name']])
        top_people = person_counts.most_common(10)  # Top 10 most mentioned people
        
        # Prepare data for analysis
        top_people_data = []
        for person, count in top_people:
            person_info = [p for p in people_data if p['name'] == person][0]
            top_people_data.append({
                "name": person,
                "role": person_info['role'],
                "context": person_info['context'],
                "mentions": count
            })
        
        # Use ChatGPT for simplified stakeholder analysis
        chatbot = ChatGPT(
            system_prompt=f"You are a political analyst specializing in relationship analysis.",
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1000
        )
        
        prompt = f"""
Analyze the key people mentioned in relation to {politician_name} in media coverage:
{json.dumps(top_people_data, indent=2)}

Provide a concise analysis that:
1. Identifies the most significant individuals in {politician_name}'s media coverage
2. Discusses their roles and potential relationships with {politician_name}
3. Considers the implications of these relationships for {politician_name}'s political network

Format your response as a well-structured analysis of 2-3 paragraphs. Be balanced and objective.
"""
        
        try:
            people_analysis = chatbot.ask(prompt)
            return people_analysis
        except Exception as e:
            logging.error(f"Error generating people analysis: {str(e)}")
            return f"Error generating people analysis: {str(e)}"


def generate_media_outlet_analysis(media_outlet_stats, politician_name):
    """
    Generate analysis of media outlet coverage patterns.
    
    Args:
        media_outlet_stats (List): Statistics about media outlets
        politician_name (str): Name of the politician
        
    Returns:
        str: Markdown formatted media outlet analysis
    """
    if not media_outlet_stats:
        return "No media outlet data available for analysis."
    
    # Use ChatGPT for media outlet analysis
    chatbot = ChatGPT(
        system_prompt=f"You are a media analyst specializing in political coverage patterns.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1200
    )
    
    prompt = f"""
Analyze how different media outlets cover {politician_name} based on these statistics:
{json.dumps(media_outlet_stats, indent=2)}

Provide a comprehensive analysis that:
1. Identifies which outlets give the most coverage to {politician_name}
2. Analyzes sentiment patterns across outlets (which are most positive/negative)
3. Discusses potential biases or patterns in the coverage
4. Considers the implications of these coverage patterns for {politician_name}'s public image

Format your response as a well-structured analysis of 3-4 paragraphs. Be balanced and objective in your assessment.
"""
    
    try:
        media_analysis = chatbot.ask(prompt)
        return media_analysis
    except Exception as e:
        logging.error(f"Error generating media outlet analysis: {str(e)}")
        return f"Error generating media outlet analysis: {str(e)}"


def generate_reputation_analysis(df, politician_name):
    """
    Generate analysis of the politician's reputation based on media coverage.
    
    Args:
        df (pd.DataFrame): DataFrame with the articles data
        politician_name (str): Name of the politician
        
    Returns:
        str: Markdown formatted reputation analysis
    """
    # Determine sentiment column name
    sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment score'
    
    # Calculate outlet sentiment statistics
    outlet_stats = []
    for outlet in df['media_outlet'].unique():
        outlet_df = df[df['media_outlet'] == outlet]
        
        # Skip outlets with very few articles
        if len(outlet_df) < 3:
            continue
            
        outlet_stats.append({
            "outlet": outlet,
            "articles": len(outlet_df),
            "avg_sentiment": outlet_df[sentiment_col].mean(),
            "pos_percent": (outlet_df['tone'] == 'Positive').mean() * 100,
            "neu_percent": (outlet_df['tone'] == 'Neutral').mean() * 100,
            "neg_percent": (outlet_df['tone'] == 'Negative').mean() * 100
        })


def generate_narrative_distribution_chart(df):
    """Generate a pie chart showing the distribution of narrative categories."""
    if 'narrative_category' not in df.columns:
        logging.warning("No narrative_category column found. Skipping narrative distribution chart.")
        return None
    
    # Count narratives
    narrative_counts = df['narrative_category'].value_counts()
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        narrative_counts, 
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontsize': 12}
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title('Distribution of Narrative Categories', fontsize=16, pad=20)
    
    # Add a legend with percentage and count
    legend_labels = [f"{label} ({count})" for label, count in zip(narrative_counts.index, narrative_counts)]
    plt.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot as base64
    image_base64 = save_plot_base64()
    plt.close()
    
    return image_base64


def generate_narrative_sentiment_chart(df):
    """Generate a horizontal bar chart showing sentiment by narrative category."""
    if 'narrative_category' not in df.columns:
        logging.warning("No narrative_category column found. Skipping narrative sentiment chart.")
        return None
    
    # Determine sentiment column name
    sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment score'
    
    # Calculate average sentiment per narrative
    narrative_sentiment = df.groupby('narrative_category')[sentiment_col].mean().sort_values()
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 8))
    bars = plt.barh(narrative_sentiment.index, narrative_sentiment, color='#8c96c6')
    
    # Add a vertical line at x=0 to indicate neutral sentiment
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add value labels to each bar
    for i, v in enumerate(narrative_sentiment):
        plt.text(v + 0.1 if v >= 0 else v - 0.5, i, f'{v:.2f}', va='center')
    
    # Formatting
    plt.title('Average Sentiment by Narrative Category', fontsize=16)
    plt.xlabel('Average Sentiment Score', fontsize=14)
    plt.ylabel('Narrative Category', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save the plot as base64
    image_base64 = save_plot_base64()
    plt.close()
    
    return image_base64


def generate_policy_focus_chart(df, politician_name):
    """
    Generate a visualization of policy focus based on extracted entities and content analysis.
    Uses LLM to extract policy areas if they're not already in the data.
    """
    # Check if policy data is available
    if 'policy_category' not in df.columns:
        # Extract policy categories using LLM
        logging.info("No policy_category column found. Attempting to extract using LLM.")
        try:
            policy_categories = extract_policy_categories(df, politician_name)
            df['policy_category'] = policy_categories
        except Exception as e:
            logging.error(f"Failed to extract policy categories: {str(e)}")
            return None
    
    # Count policy areas
    policy_counts = df['policy_category'].value_counts()
    
    # Create the chart
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(policy_counts.index, policy_counts, color='#4682B4')
    
    # Add count labels
    for i, v in enumerate(policy_counts):
        plt.text(v + 0.3, i, str(v), va='center')
    
    # Formatting
    plt.title(f'Policy Focus Areas for {politician_name}', fontsize=16)
    plt.xlabel('Number of Articles', fontsize=14)
    plt.ylabel('Policy Area', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save the plot as base64
    image_base64 = save_plot_base64()
    plt.close()
    
    return image_base64


def extract_policy_categories(df, politician_name):
    """
    Extract policy categories from article content using LLM.
    This is a fallback for when policy_category is not already available.
    """
    policy_categories = []
    
    # Define the policy categories (customize as needed)
    standard_categories = [
        "Economy & Finance", 
        "Foreign Policy & International Relations",
        "Environment & Climate Change",
        "Healthcare",
        "Education",
        "National Security & Defense",
        "Immigration",
        "Social Justice & Human Rights",
        "Infrastructure & Transportation",
        "Technology & Digital Policy"
    ]
    
    # Create a prompt for the ChatGPT model
    category_prompt = f"""
You are analyzing an article about {politician_name}. Based on the content, which of the following policy areas is MOST prominently discussed in relation to {politician_name}?

{', '.join(standard_categories)}

Select ONLY ONE category that best represents the primary policy focus of the article. If no policy area is discussed, respond with "No specific policy".
Output just the category name and nothing else.
"""
    
    # Process each article
    for i, row in df.iterrows():
        try:
            content = row['content']
            chatbot = ChatGPT(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=50
            )
            response = chatbot.ask(category_prompt + f"\n\nArticle content: {content}")
            
            # Clean the response
            category = response.strip()
            if category not in standard_categories and category != "No specific policy":
                # Try to match to closest standard category
                best_match = max(standard_categories, key=lambda x: text_similarity(category, x))
                category = best_match
            
            policy_categories.append(category)
            
        except Exception as e:
            logging.error(f"Error extracting policy category for article {i}: {str(e)}")
            policy_categories.append("No specific policy")
    
    return policy_categories


def text_similarity(text1, text2):
    """Simple text similarity function."""
    # Convert to lowercase and split into words
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union) if union else 0


def generate_stakeholder_network_chart(df, politician_name, general_folder):
    """
    Generate a network visualization of the politician's stakeholder relationships.
    Uses stakeholder data if available, otherwise extracts from articles.
    """
    # Try to load stakeholder data first
    stakeholder_path = os.path.join(
        general_folder, "Outputs", "PoliticianAnalysis", "Stakeholders",
        f"StakeholderAnalysis_{politician_name.replace(' ', '_')}.json"
    )
    
    if os.path.exists(stakeholder_path):
        try:
            with open(stakeholder_path, 'r', encoding='utf-8') as f:
                stakeholder_data = json.load(f)
            
            # Create a network graph
            G = create_stakeholder_network_graph(stakeholder_data, politician_name)
            
            # Visualize the network
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # Draw nodes
            node_sizes = []
            node_colors = []
            for node in G.nodes():
                if node == politician_name:
                    node_sizes.append(1000)  # Larger size for the politician
                    node_colors.append('red')
                else:
                    node_sizes.append(500)  # Default size for stakeholders
                    relationship = G.nodes[node].get('relationship', 'neutral')
                    if relationship == 'ally':
                        node_colors.append('green')
                    elif relationship == 'opponent':
                        node_colors.append('orange')
                    else:
                        node_colors.append('blue')
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
            
            # Draw edges
            edge_colors = []
            for u, v, data in G.edges(data=True):
                edge_type = data.get('type', 'neutral')
                if edge_type == 'positive':
                    edge_colors.append('green')
                elif edge_type == 'negative':
                    edge_colors.append('red')
                else:
                    edge_colors.append('gray')
            
            nx.draw_networkx_edges(G, pos, width=1.5, edge_color=edge_colors, alpha=0.7)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            
            plt.title(f"Stakeholder Network for {politician_name}", fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            
            # Save the plot as base64
            image_base64 = save_plot_base64()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logging.error(f"Error generating stakeholder network from data: {str(e)}")
    
    # If we reach here, either the file doesn't exist or we encountered an error
    logging.warning("No stakeholder data found or error occurred. Generating simplified network.")
    
    # Create a simplified network with just people entities from articles
    try:
        # Extract people if available
        people_entities = []
        for _, row in df.iterrows():
            if 'people' in row and isinstance(row['people'], list):
                for person in row['people']:
                    name = person.get('name', '')
                    role = person.get('role', '')
                    if name and name != politician_name:
                        people_entities.append({'name': name, 'role': role})
        
        # Count mentions
        entity_counts = Counter([entity['name'] for entity in people_entities])
        top_entities = entity_counts.most_common(10)  # Top 10 mentioned people
        
        # Create a simple network graph
        G = nx.Graph()
        G.add_node(politician_name, type='politician')
        
        for entity, count in top_entities:
            G.add_node(entity, type='person')
            G.add_edge(politician_name, entity, weight=count)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=[politician_name], 
                              node_color='red',
                              node_size=800,
                              alpha=0.8)
        
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=[n for n in G.nodes() if n != politician_name], 
                              node_color='blue',
                              node_size=500,
                              alpha=0.8)
        
        # Draw edges with varying thickness based on mention count
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            width = np.log1p(weight) * 1.5  # Logarithmic scaling for better visualization
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title(f"People Most Frequently Mentioned with {politician_name}", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot as base64
        image_base64 = save_plot_base64()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        logging.error(f"Error generating simplified network: {str(e)}")
        return None


def create_stakeholder_network_graph(stakeholder_data, politician_name):
    """Create a network graph from stakeholder data."""
    G = nx.Graph()
    
    # Add the politician as the central node
    G.add_node(politician_name, type='politician')
    
    # Get all stakeholders
    stakeholders = stakeholder_data.get("stakeholders", [])
    
    # Add stakeholder nodes and edges
    for stakeholder in stakeholders:
        name = stakeholder.get('name', '')
        if not name or name == politician_name:
            continue
        
        stakeholder_type = stakeholder.get('type', 'unknown')
        role = stakeholder.get('role', '')
        relationship_type = stakeholder.get('relationship_type', 'neutral')
        category = stakeholder.get('category', '')
        mentions = stakeholder.get('mentions', 1)
        
        # Simplify relationship type to ally/opponent/neutral
        if any(term in relationship_type.lower() for term in ['ally', 'support', 'collaborator', 'partner']):
            simple_relationship = 'ally'
            edge_type = 'positive'
        elif any(term in relationship_type.lower() for term in ['opponent', 'critic', 'adversary', 'opposition']):
            simple_relationship = 'opponent'
            edge_type = 'negative'
        else:
            simple_relationship = 'neutral'
            edge_type = 'neutral'
        
        # Add node with attributes
        G.add_node(name, 
                  type=stakeholder_type, 
                  role=role, 
                  relationship=simple_relationship,
                  category=category)
        
        # Add edge with weight based on mentions
        G.add_edge(politician_name, name, 
                  weight=mentions, 
                  type=edge_type)
    
    return G


def generate_political_reputation_chart(df, politician_name):
    """
    Generate a multi-dimensional chart showing the politician's reputation across
    different media outlets and narrative dimensions.
    """
    # Determine sentiment column name
    sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment score'
    
    # For each media outlet, calculate sentiment stats
    outlet_sentiments = df.groupby('media_outlet')[sentiment_col].agg(['mean', 'count']).reset_index()
    outlet_sentiments = outlet_sentiments.sort_values('count', ascending=False).head(10)
    
    # Calculate the tone percentages for each outlet
    outlet_tones = {}
    for outlet in outlet_sentiments['media_outlet']:
        outlet_df = df[df['media_outlet'] == outlet]
        pos = (outlet_df['tone'] == 'Positive').mean() * 100
        neu = (outlet_df['tone'] == 'Neutral').mean() * 100
        neg = (outlet_df['tone'] == 'Negative').mean() * 100
        outlet_tones[outlet] = {'Positive': pos, 'Neutral': neu, 'Negative': neg}
    
    # Create the reputation chart
    plt.figure(figsize=(14, 8))
    
    # Use a colormap
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(outlet_sentiments)))
    
    bars = plt.bar(outlet_sentiments['media_outlet'], outlet_sentiments['mean'], color=colors)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add labels with outlet tone breakdown
    for i, bar in enumerate(bars):
        outlet = outlet_sentiments.iloc[i]['media_outlet']
        count = outlet_sentiments.iloc[i]['count']
        tone_breakdown = outlet_tones[outlet]
        
        # Format the tooltip text
        tooltip = f"{outlet} ({count} articles)\n" + \
                 f"Pos: {tone_breakdown['Positive']:.1f}%\n" + \
                 f"Neu: {tone_breakdown['Neutral']:.1f}%\n" + \
                 f"Neg: {tone_breakdown['Negative']:.1f}%"
        
        # Position the annotation
        y_pos = outlet_sentiments.iloc[i]['mean']
        plt.annotate(tooltip, 
                   xy=(i, y_pos),
                   xytext=(0, 10 if y_pos >= 0 else -50),
                   textcoords="offset points",
                   ha='center',
                   va='bottom' if y_pos >= 0 else 'top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                   fontsize=8)
    
    # Formatting
    plt.title(f'Reputation Analysis of {politician_name} by Media Outlets', fontsize=16)
    plt.xlabel('Media Outlet', fontsize=14)
    plt.ylabel('Average Sentiment Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot as base64
    image_base64 = save_plot_base64()
    plt.close()
    
    return image_base64


def generate_politician_markdown_report(
    politician_name, political_party, region, total_articles, date_range, 
    avg_sentiment, median_sentiment, media_outlet_pie_chart,
    media_outlet_tone_chart, overall_sentiment_trend, narrative_distribution_chart,
    narrative_sentiment_chart, policy_focus_chart, stakeholder_network_chart,
    political_reputation_chart, media_outlet_stats, narrative_stats,
    df, general_folder, language='English'):
    """
    Generate a comprehensive markdown report for political media analytics.
    
    Args:
        politician_name (str): Name of the politician being analyzed
        political_party (str): Political party of the politician
        region (str): Geographic region
        total_articles (int): Total number of articles analyzed
        date_range (str): Range of dates covered by the articles
        avg_sentiment (float): Average sentiment score
        median_sentiment (float): Median sentiment score
        media_outlet_pie_chart (str): Base64 encoded image of media outlet distribution
        media_outlet_tone_chart (str): Base64 encoded image of media outlet tone analysis
        overall_sentiment_trend (str): Base64 encoded image of sentiment trend over time
        narrative_distribution_chart (str): Base64 encoded image of narrative distribution
        narrative_sentiment_chart (str): Base64 encoded image of narrative sentiment
        policy_focus_chart (str): Base64 encoded image of policy focus analysis
        stakeholder_network_chart (str): Base64 encoded image of stakeholder network
        political_reputation_chart (str): Base64 encoded image of political reputation
        media_outlet_stats (List): Statistics about media outlets
        narrative_stats (List): Statistics about narratives
        df (pd.DataFrame): DataFrame containing the articles data
        general_folder (str): Base path for output files
        language (str): Output language
        
    Returns:
        str: Markdown content for the political media analytics report
    """
    # Define translatable strings
    pie_chart_text = "The pie chart below shows the distribution of articles across different media outlets."
    political_identity = f"{politician_name} is a politician from {region}"
    if political_party:
        political_identity += f", affiliated with the {political_party}."
    else:
        political_identity += "."
    
    # Translate if needed
    if language.lower() != "english":
        pie_chart_text = translate_content(pie_chart_text, "English", language).replace("#", "")
        political_identity = translate_content(political_identity, "English", language).replace("#", "")
    
    # Start building the markdown content
    markdown_content = f"""
# {politician_name} - Political Media Analytics Report

## Introduction
{political_identity} This report analyzes the media coverage of {politician_name} based on {total_articles} articles.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Media Coverage Distribution](#media-coverage-distribution)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Narrative Analysis](#narrative-analysis)
6. [Policy Focus Analysis](#policy-focus-analysis)
7. [Stakeholder Network Analysis](#stakeholder-network-analysis)
8. [Media Outlet Analysis](#media-outlet-analysis)
9. [Political Reputation Analysis](#political-reputation-analysis)

## Data Overview
- **Politician Name**: {politician_name}
- **Political Party**: {political_party if political_party else "Not specified"}
- **Region**: {region}
- **Total number of articles analyzed**: {total_articles}
- **Date range**: {date_range}
- **Average sentiment score**: {avg_sentiment:.2f}
- **Median sentiment score**: {median_sentiment:.2f}
"""

    # Add media coverage distribution section
    markdown_content += f"""
## Media Coverage Distribution
{pie_chart_text}

![Distribution of Articles by Media Outlet](data:image/png;base64,{media_outlet_pie_chart})

### Articles per Media Outlet (By Tone)
![Articles per Media Outlet (By Tone)](data:image/png;base64,{media_outlet_tone_chart})
"""

    # Add publication timeline section
    if 'date' in df.columns:
        publication_timeline_section = generate_publication_timeline_section(df, politician_name, language)
        markdown_content += publication_timeline_section

    # Add sentiment analysis section
    markdown_content += f"""
## Sentiment Analysis
![Overall Sentiment Trend](data:image/png;base64,{overall_sentiment_trend})
"""
    
    # Add sentiment evolution analysis
    sentiment_analysis_content = generate_political_sentiment_analysis(df, politician_name)
    if language.lower() != "english":
        sentiment_analysis_content = translate_content(sentiment_analysis_content, "English", language)
    markdown_content += sentiment_analysis_content

    # Add narrative analysis section if available
    if narrative_distribution_chart and narrative_sentiment_chart:
        markdown_content += f"""
## Narrative Analysis
### Distribution of Narrative Categories
![Distribution of Narrative Categories](data:image/png;base64,{narrative_distribution_chart})

### Sentiment by Narrative Category
![Sentiment by Narrative Category](data:image/png;base64,{narrative_sentiment_chart})
"""

        # Add table of narrative statistics
        if narrative_stats:
            markdown_content += """
### Narrative Statistics
| Narrative | Articles | Avg. Sentiment | Positive % | Neutral % | Negative % |
|-----------|----------|---------------|-----------|-----------|-----------|
"""
            for stat in narrative_stats:
                markdown_content += f"| {stat['narrative']} | {stat['articles']} | {stat['avg_sentiment']:.2f} | {stat['pos_percent']:.1f}% | {stat['neu_percent']:.1f}% | {stat['neg_percent']:.1f}% |\n"

        # Add narrative analysis from ChatGPT
        narrative_analysis = generate_narrative_analysis(narrative_stats, politician_name)
        if language.lower() != "english":
            narrative_analysis = translate_content(narrative_analysis, "English", language)
        markdown_content += f"\n### Narrative Analysis\n{narrative_analysis}\n"

    # Add policy focus analysis section if available
    if policy_focus_chart:
        markdown_content += f"""
## Policy Focus Analysis
![Policy Focus Areas](data:image/png;base64,{policy_focus_chart})
"""
        # Add policy analysis from ChatGPT
        policy_analysis = generate_policy_analysis(df, politician_name)
        if language.lower() != "english":
            policy_analysis = translate_content(policy_analysis, "English", language)
        markdown_content += f"\n### Policy Analysis\n{policy_analysis}\n"

    # Add stakeholder network analysis section if available
    if stakeholder_network_chart:
        markdown_content += f"""
## Stakeholder Network Analysis
![Stakeholder Network](data:image/png;base64,{stakeholder_network_chart})
"""
        # Add stakeholder analysis from ChatGPT
        stakeholder_analysis = generate_stakeholder_analysis(df, politician_name, general_folder)
        if language.lower() != "english":
            stakeholder_analysis = translate_content(stakeholder_analysis, "English", language)
        markdown_content += f"\n### Stakeholder Analysis\n{stakeholder_analysis}\n"

    # Add media outlet analysis section
    markdown_content += f"""
## Media Outlet Analysis
### Media Outlet Statistics
| Media Outlet | Number of Articles | Average Sentiment | Median Sentiment |
|--------------|---------------------|-------------------|-------------------|
"""
    
    for stat in media_outlet_stats:
        markdown_content += f"| {stat['outlet']} | {stat['articles']} | {stat['avg_sentiment']:.2f} | {stat['median_sentiment']:.2f} |\n"

    # Add media analysis from ChatGPT
    media_analysis = generate_media_outlet_analysis(media_outlet_stats, politician_name)
    if language.lower() != "english":
        media_analysis = translate_content(media_analysis, "English", language)
    markdown_content += f"\n### Media Outlet Coverage Analysis\n{media_analysis}\n"

    # Add political reputation analysis section if available
    if political_reputation_chart:
        markdown_content += f"""
## Political Reputation Analysis
![Political Reputation by Media Outlet](data:image/png;base64,{political_reputation_chart})
"""
        # Add reputation analysis from ChatGPT
        reputation_analysis = generate_reputation_analysis(df, politician_name)
        if language.lower() != "english":
            reputation_analysis = translate_content(reputation_analysis, "English", language)
        markdown_content += f"\n### Reputation Analysis\n{reputation_analysis}\n"

#    # Add overall analysis summary
#    markdown_content += f"""
### Overall Analysis Summary
#"""
#    overall_analysis = generate_overall_analysis_summary(
#        df, politician_name, political_party, region, 
#        total_articles, avg_sentiment, narrative_stats if narrative_stats else []
#    )
#    if language.lower() != "english":
#        overall_analysis = translate_content(overall_analysis, "English", language)
#    markdown_content += overall_analysis

    return markdown_content