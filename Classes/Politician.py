import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import logging
import traceback
from pathlib import Path

from Classes.SimplifiedChatbots import ChatGPT, BigSummarizerGPT
from Classes.DocumentProcessor import DocumentProcessor, CitationProcessor
from Utils.Helpers import *

class PoliticianAnalyzer:
    """
    A class for analyzing media coverage of politicians.
    Implements a hybrid approach that leverages core components
    while adding politician-specific analysis.
    """
    
    def __init__(self, 
                 politician_name: str, 
                 articles: List[Dict], 
                 general_folder: str,
                 region: str = None,
                 political_party: str = None,
                 language: str = "English",
                 force_reprocess: bool = False):
        """
        Initialize the politician analyzer.
        
        Args:
            politician_name (str): Name of the politician being analyzed
            articles (List[Dict]): List of preprocessed articles
            general_folder (str): Base path for output files
            region (str, optional): Geographic region of politician
            political_party (str, optional): Political party affiliation
            language (str, optional): Output language for analysis
            force_reprocess (bool, optional): Whether to force reprocessing
        """
        self.politician_name = politician_name
        self.articles = articles
        self.general_folder = general_folder
        self.region = region
        self.political_party = political_party
        self.language = language
        self.force_reprocess = force_reprocess
        
        # Set up directories
        self.setup_directories()
        
        # Initialize processing status
        self.has_basic_sentiment = False
        self.has_narratives = False
        self.has_policy_positions = False
        self.has_stakeholders = False
        self.has_entities = False
        
        # Check existing processing status
        self.check_processing_status()
    
    def setup_directories(self):
        """Create necessary directory structure for outputs."""
        # Main output directories
        os.makedirs(os.path.join(self.general_folder, "Outputs", "CompiledOutputs"), exist_ok=True)
        os.makedirs(os.path.join(self.general_folder, "Outputs", "PoliticianAnalysis"), exist_ok=True)
        os.makedirs(os.path.join(self.general_folder, "Outputs", "Visualizations"), exist_ok=True)
        
        # Specialized directories
        os.makedirs(os.path.join(self.general_folder, "Outputs", "PoliticianAnalysis", "PolicyPositions"), exist_ok=True)
        os.makedirs(os.path.join(self.general_folder, "Outputs", "PoliticianAnalysis", "Narratives"), exist_ok=True)
        os.makedirs(os.path.join(self.general_folder, "Outputs", "PoliticianAnalysis", "Stakeholders"), exist_ok=True)
        os.makedirs(os.path.join(self.general_folder, "Outputs", "PoliticianAnalysis", "Timeline"), exist_ok=True)
    
    def check_processing_status(self):
        """Check for existing processed data to avoid redundant processing."""
        politician_data_path = os.path.join(
            self.general_folder, "Outputs", "CompiledOutputs", 
            f"PoliticianAnalysis_{self.politician_name.replace(' ', '_')}.json"
        )
        
        if os.path.exists(politician_data_path) and not self.force_reprocess:
            try:
                with open(politician_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.has_basic_sentiment = data.get('has_basic_sentiment', False)
                self.has_narratives = data.get('has_narratives', False)
                self.has_policy_positions = data.get('has_policy_positions', False)
                self.has_stakeholders = data.get('has_stakeholders', False)
                self.has_entities = data.get('has_entities', False)
                
                # If we have processed data, update our articles with it
                if 'articles' in data:
                    self.articles = data['articles']
                    logging.info(f"Loaded {len(self.articles)} preprocessed articles")
            except Exception as e:
                logging.error(f"Error loading preprocessing status: {str(e)}")
                logging.error(traceback.format_exc())
    
    def save_processing_status(self):
        """Save current processing status and articles to avoid reprocessing."""
        politician_data_path = os.path.join(
            self.general_folder, "Outputs", "CompiledOutputs", 
            f"PoliticianAnalysis_{self.politician_name.replace(' ', '_')}.json"
        )
        
        try:
            data = {
                'politician_name': self.politician_name,
                'has_basic_sentiment': self.has_basic_sentiment,
                'has_narratives': self.has_narratives,
                'has_policy_positions': self.has_policy_positions,
                'has_stakeholders': self.has_stakeholders,
                'has_entities': self.has_entities,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'articles': self.articles
            }
            
            with open(politician_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logging.info(f"Saved processing status to {politician_data_path}")
        except Exception as e:
            logging.error(f"Error saving processing status: {str(e)}")
            logging.error(traceback.format_exc())
    
    def preprocess_articles(self):
        """
        Preprocess articles for analysis, including:
        - Extracting basic sentiment
        - Adding timestamps for chronological analysis
        - Deduplicating articles
        """
        logging.info(f"Preprocessing {len(self.articles)} articles")
        
        if self.has_basic_sentiment and not self.force_reprocess:
            logging.info("Using existing sentiment analysis")
            return self.articles
        
        system_prompt = f"""You are a helpful assistant. Your role is to assess the overall tone of a news article, specifically focusing on the article's content about {self.politician_name}."""
        
        for article in self.articles:
            # Add timestamp for chronological analysis
            if 'date' in article:
                try:
                    date_object = datetime.strptime(article['date'], '%B %d, %Y')
                    article['timestamp'] = date_object.timestamp()
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse date: {article.get('date')}")
                    article['timestamp'] = 0
            else:
                article['timestamp'] = 0
            
            # Add sentiment analysis if not already present
            if 'tone' not in article or not article['tone'] or self.force_reprocess:
                chatbot = ChatGPT(
                    system_prompt=system_prompt,
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=50,
                )
                question = f"""
Please assess the tone of the news article with specific regard to: {self.politician_name}. Focus on how the article portrays {self.politician_name} in terms of actions, decisions, statements, and character. The tone should be categorized as one of the following:

Positive: The article reflects well on {self.politician_name}, highlighting favorable aspects.
Neutral: The article presents information about {self.politician_name} in a balanced, objective manner without any strong positive or negative bias.
Negative: The article is critical of {self.politician_name}, highlighting challenges, controversies, failures, or unfavorable aspects.

Provide the final tone assessment as "Positive," "Neutral," or "Negative."

Here is the article content: {article['content']}

Your output should solely be one of these three words based on your assessment: "Positive", "Neutral", or "Negative". Nothing else
                """
                response = chatbot.ask(question)
                print(response)
                article['tone'] = response.strip()
            
            # Add sentiment score if not already present
            if 'sentiment_score' not in article or article['sentiment_score'] is None or self.force_reprocess:
                chatbot = ChatGPT(
                    system_prompt=system_prompt,
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=50,
                )
                question = f"""
Perform a detailed sentiment analysis of the article provided below, focusing exclusively on how it describes {self.politician_name}. Your task is to evaluate the overall sentiment expressed in the article regarding {self.politician_name} by carefully considering the tone, language, context, and any descriptive cues related to {self.politician_name}.

Guidelines for Scoring:

-5: The article is extremely critical and conveys a highly negative sentiment toward {self.politician_name}.
-4: The article offers notable criticism, highlighting significant negative aspects of {self.politician_name}.
-3: The sentiment is moderately negative, with clear indications of disapproval of {self.politician_name}.
-2: Some negative remarks about {self.politician_name} are present, though they are not predominant.
-1: The article shows mild negativity or slight disapproval of {self.politician_name}.
0: The portrayal of {self.politician_name} is neutral with no significant positive or negative sentiment.
1: The article exhibits slight positive sentiment or mild approval of {self.politician_name}.
2: The tone is moderately positive, suggesting a favorable view of {self.politician_name}.
3: The article is largely favorable, displaying clear positive sentiment about {self.politician_name}.
4: The content strongly praises {self.politician_name}.
5: The article is exceptionally complimentary, demonstrating an extremely positive sentiment about {self.politician_name}.

Instructions:
Analyze the language, tone, and context used in the article with respect to {self.politician_name}.
Based solely on the observations, assign one sentiment score from the list: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].
Your final output must be exactly one of these values and nothing else.
Article Content: {article['content']}

Only output one of these values: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], and nothing else.
                """
                response = chatbot.ask(question)
                print(response)
                try:
                    sentiment_score = int(response.strip())
                    if sentiment_score < -5 or sentiment_score > 5:
                        raise ValueError("Score out of range")
                    article['sentiment_score'] = sentiment_score
                except (ValueError, TypeError):
                    logging.warning(f"Invalid sentiment score: {response}. Using 0 as default.")
                    article['sentiment_score'] = 0
        
        # Sort articles chronologically
        self.articles = sorted(self.articles, key=lambda x: x.get('timestamp', 0))
        
        self.has_basic_sentiment = True
        self.save_processing_status()
        
        return self.articles
    
    def analyze_narratives(self):
        """
        Identify recurring narratives and frames used to describe the politician.
        Returns dictionary of narrative categories and their descriptions.
        """
        logging.info("Beginning narrative analysis")
        
        if self.has_narratives and not self.force_reprocess:
            logging.info("Using existing narrative analysis")
            return self._get_narratives_from_articles()
        
        # First generate one-sentence descriptions for each article
        system_prompt = f"""You are a political media analyst. Your role is to describe in one single sentence what a given news media article says about {self.politician_name}."""
        compiled_sentences = ""
        
        for article in self.articles:
            if 'one_sentence_description' not in article or self.force_reprocess:
                chatbot = ChatGPT(
                    system_prompt=system_prompt,
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=350,
                )
                question = f"""
Please write a single sentence summarizing this article's portrayal of {self.politician_name}. Focus on how the politician is framed, what actions or positions are highlighted, and the overall narrative about them.

Article: {article['content']}
                """
                response = chatbot.ask(question)
                print(response)
                article['one_sentence_description'] = response
            
            compiled_sentences += article['one_sentence_description'] + "\n"
        
        # Identify key narrative frames
        system_prompt = f"""You are a political communication expert. Your role is to identify the primary recurring narratives and frames used to portray {self.politician_name} across multiple media articles."""
        
        chatbot = ChatGPT(
            system_prompt=system_prompt,
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1500,
        )
        
        question = f"""
Based on these article summaries, identify the main recurring narratives or frames used to portray {self.politician_name} in the media.
Create 5-8 distinct narrative categories that represent specific framing patterns, recurring stories, or thematic approaches to covering this politician.

Article summaries:
{compiled_sentences}

For each narrative category:
1. Give it a specific name that describes the frame or narrative approach
2. Provide a detailed description of what this narrative entails, how it portrays the politician
3. Focus on identifying narrative patterns rather than just policy areas

Format your response exactly as follows:

CATEGORY: [Narrative/Frame Name]
DESCRIPTION: [Detailed explanation of the narrative/frame and how it portrays the politician]

Ensure categories are distinct and capture different aspects of media portrayal.
"""
        
        categories_response = chatbot.ask(question)
        print(categories_response)
        
        # Parse the narrative categories
        categories_data = []
        current_category = None
        current_description = None
        
        for line in categories_response.split('\n'):
            if line.startswith('CATEGORY:'):
                if current_category is not None:
                    categories_data.append({
                        'category': current_category.strip(),
                        'description': current_description.strip() if current_description else '',
                        'articles': []
                    })
                current_category = line.replace('CATEGORY:', '').strip()
                current_description = None
            elif line.startswith('DESCRIPTION:'):
                current_description = line.replace('DESCRIPTION:', '').strip()
        
        # Add the last category
        if current_category is not None:
            categories_data.append({
                'category': current_category,
                'description': current_description if current_description else '',
                'articles': []
            })
        
        # Categorize each article into a narrative category
        for article in self.articles:
            classification_prompt = ""
            for category in categories_data:
                classification_prompt += f"\n{category['category']}: {category['description']}"
            
            chatbot = ChatGPT(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=200
            )
            
            classification = chatbot.ask(
                f"""
Given these narrative categories for {self.politician_name}:
{classification_prompt}

Classify this article into one of these categories. Choose the most appropriate narrative frame based on the descriptions provided.

Article content:
{article['content']}

Only output the exact category name that best matches this article. Your output should be just the category name and nothing else.
                """
            )
            print(classification)
            
            # Find the matching category
            article['narrative_category'] = None
            for category in categories_data:
                if classification.strip() == category['category'].strip():
                    article['narrative_category'] = category['category']
                    category['articles'].append(article)
                    break
            
            # If no match was found, use the first category as a fallback
            if article['narrative_category'] is None and categories_data:
                article['narrative_category'] = categories_data[0]['category']
                categories_data[0]['articles'].append(article)
        
        # Save narrative frames as a separate file
        narrative_frames_path = os.path.join(
            self.general_folder, "Outputs", "PoliticianAnalysis", "Narratives",
            f"NarrativeFrames_{self.politician_name.replace(' ', '_')}.json"
        )
        
        with open(narrative_frames_path, 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, indent=2)
        
        self.has_narratives = True
        self.save_processing_status()
        
        return categories_data
    
    def _get_narratives_from_articles(self):
        """Helper method to reconstruct narrative categories from article data."""
        narrative_frames_path = os.path.join(
            self.general_folder, "Outputs", "PoliticianAnalysis", "Narratives",
            f"NarrativeFrames_{self.politician_name.replace(' ', '_')}.json"
        )
        
        if os.path.exists(narrative_frames_path):
            with open(narrative_frames_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # If file doesn't exist but articles have narrative categories,
        # reconstruct from article data
        categories = {}
        for article in self.articles:
            if 'narrative_category' in article:
                category = article['narrative_category']
                if category not in categories:
                    categories[category] = {
                        'category': category,
                        'description': '',  # We don't have descriptions
                        'articles': []
                    }
                categories[category]['articles'].append(article)
        
        return list(categories.values())
    
    def analyze_policy_positions(self, force_reprocess=False):
        """
        Extract and analyze how the politician's policy positions are portrayed using BigSummarizerGPT
        to process large volumes of text, with an additional clustering step to consolidate similar policy areas.
        Returns dictionary of policy areas and associated data.
        """
        logging.info("Beginning policy position analysis")

        # Use either the instance-level or method-level force_reprocess
        local_force_reprocess = force_reprocess or self.force_reprocess

        if self.has_policy_positions and not local_force_reprocess:
            logging.info("Using existing policy position analysis")
            policy_positions_path = os.path.join(
                self.general_folder, "Outputs", "PoliticianAnalysis", "PolicyPositions",
                f"PolicyPositions_{self.politician_name.replace(' ', '_')}.json"
            )

            if os.path.exists(policy_positions_path):
                with open(policy_positions_path, 'r', encoding='utf-8') as f:
                    policy_clusters = json.load(f)
                    # Generate the MD output file even if we're using cached data
                    self._generate_policy_clusters_md(policy_clusters)
                    return policy_clusters
            else:
                logging.warning("Policy positions marked as processed but file not found")

        # Compile all article contents for analysis
        temp_compiled_path = os.path.join(
            self.general_folder, "Outputs", "CompiledOutputs", 
            f"CompiledArticles_{self.politician_name.replace(' ', '_')}.md"
        )

        compiled_content = ""
        for article in self.articles:
            metadata = f"""
    Title: {article.get('title', 'Untitled')}
    Media outlet: {article.get('media_outlet', 'Unknown')}
    Date: {article.get('date', 'Unknown')}
    """
            compiled_content += "\n\n" + "---" + metadata + article.get('content', '') + "\n\n" + metadata + "---\n\n"

        # Save compiled content for BigSummarizerGPT
        with open(temp_compiled_path, "w", encoding='utf-8') as f:
            f.write(compiled_content)

        # STEP 1: Identify initial policy areas using BigSummarizerGPT
        system_prompt = f"""You are a political policy analyst. Your role is to identify the main policy areas discussed in relation to {self.politician_name} across multiple media articles."""

        bigbot = BigSummarizerGPT(
            system_prompt=system_prompt,
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1500
        )

        policy_areas_prompt = f"""
Based on the media coverage, identify the main policy areas where {self.politician_name}'s positions, statements, or actions are discussed.
Create a comprehensive list of distinct policy areas that represent the key domains of political discussion.
Be specific and provide details - we will cluster these together in a later step.

For each policy area:
1. Give it a clear, specific name (e.g., "Carbon Pricing Policy" rather than just "Climate Policy")
2. Provide a brief description of what this policy area encompasses
3. Focus on areas where the politician's positions or actions are specifically mentioned

Format your response exactly as follows:

POLICY_AREA: [Policy Area Name]
DESCRIPTION: [Brief description of what this policy area covers]

Aim to identify all distinct policy areas without concern for the total number at this stage.
    """

        policy_areas_response = bigbot.ask(policy_areas_prompt, temp_compiled_path)
        print("Initial policy areas identified:")
        print(policy_areas_response)

        # Parse initial policy areas
        initial_policy_areas = []
        current_area = None
        current_description = None

        for line in policy_areas_response.split('\n'):
            if line.startswith('POLICY_AREA:'):
                if current_area is not None:
                    initial_policy_areas.append({
                        'policy_area': current_area.strip(),
                        'description': current_description.strip() if current_description else ''
                    })
                current_area = line.replace('POLICY_AREA:', '').strip()
                current_description = None
            elif line.startswith('DESCRIPTION:'):
                current_description = line.replace('DESCRIPTION:', '').strip()

        # Add the last policy area
        if current_area is not None:
            initial_policy_areas.append({
                'policy_area': current_area.strip(),
                'description': current_description.strip() if current_description else ''
            })

        print(f"Parsed {len(initial_policy_areas)} initial policy areas")

        # STEP 2: Cluster policy areas into broader categories (6-9 clusters)
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2000
        )

        clustering_prompt = f"""
I've identified {len(initial_policy_areas)} specific policy areas related to {self.politician_name}. 
I need to cluster these into 6-9 broader policy categories that are still descriptive and meaningful.

For example, specific areas like "Carbon Pricing", "Biodiversity Restoration", and "Renewable Energy Subsidies" 
might be clustered into a broader "Environment & Climate Policy" category.

Here are the specific policy areas:

{json.dumps([{'name': area['policy_area'], 'description': area['description']} for area in initial_policy_areas], indent=2)}

Please create 6-9 broader policy clusters by grouping these specific areas.
For each cluster:
1. Provide a descriptive name that captures the theme
2. Make an exhaustive and extensive list of which specific policy areas belong to this cluster (BUT AVOID DUPLICATING SYNONYM POLICIES)
3. Write a comprehensive description of the cluster that incorporates elements from its component areas

Format your response as follows:

CLUSTER: [Cluster Name]
COMPONENTS: [List of specific policy areas that belong to this cluster, without duplicating synonym policies]
DESCRIPTION: [Comprehensive description of this policy cluster]

Ensure that every specific policy area is assigned to exactly one cluster, and that the clusters are balanced and meaningful.
    """

        clustering_response = chatbot.ask(clustering_prompt)
        print("Policy clustering result:")
        print(clustering_response)

        # Parse the clustered policy areas
        policy_clusters = []
        current_cluster = None
        current_components = None
        current_description = None

        for line in clustering_response.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Updated parsing for "CLUSTER:" format
            if "CLUSTER:" in line:
                if current_cluster is not None:
                    policy_clusters.append({
                        'policy_area': current_cluster.strip(),
                        'components': current_components if current_components else [],
                        'description': current_description.strip() if current_description else '',
                        'positions': [],
                        'sentiment': {
                            'average': 0,
                            'positive': 0,
                            'neutral': 0,
                            'negative': 0
                        }
                    })
                # Extract cluster name, handling both "CLUSTER: X" and "**CLUSTER:** X" formats
                current_cluster = line.split("CLUSTER:")[1].replace('*', '').strip()
                current_components = []
                current_description = None

            # Handle components section - note the response is using bullet points
            elif "COMPONENTS:" in line or line.strip().startswith('- '):
                if "COMPONENTS:" in line:
                    # Skip the header line
                    continue
                elif line.strip().startswith('- '):
                    # This is a component item - add to components list
                    component = line.strip().replace('- ', '').strip()
                    if current_components is not None:
                        current_components.append(component)

            # Handle description section
            elif "DESCRIPTION:" in line:
                # Start capturing description
                current_description = line.replace("DESCRIPTION:", "").strip().replace('*', '')
            elif current_description is not None:
                # Continue appending to description
                current_description += " " + line.strip()

        # Add the last cluster if it exists
        if current_cluster is not None:
            policy_clusters.append({
                'policy_area': current_cluster.strip(),
                'components': current_components if current_components else [],
                'description': current_description.strip() if current_description else '',
                'positions': [],
                'sentiment': {
                    'average': 0,
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0
                }
            })

        print(f"Created {len(policy_clusters)} policy clusters")

        # STEP 3: Extract positions for each policy cluster using BigSummarizerGPT
        for policy_cluster in policy_clusters:
            # Create a comma-separated list of all component policy areas
            components_str = ", ".join(policy_cluster['components'])

            bigbot = BigSummarizerGPT(
                system_prompt=system_prompt,
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=1500
            )

            positions_prompt = f"""
Extract comprehensive information about {self.politician_name}'s positions, statements, or actions related to {policy_cluster['policy_area']}.
Among others, This cluster includes the following specific policy areas: {components_str}

For each relevant mention across all articles, provide:
1. A direct quote or paraphrase of the position/statement (if available)
2. The context in which it was mentioned
3. The date of the article
4. The media outlet

Format each position as a bullet point starting with a dash (-).
Each bullet point should be self-contained and include all the information above.
For each bullet point, include references to the specific articles where each position was mentioned. Reference the media outlet, author and date in [Media outlet, Austhor, date].
If no information is found for this policy area, output: "- No specific positions found on {policy_cluster['policy_area']}."
    """

            positions_response = bigbot.ask(positions_prompt, temp_compiled_path)
            print(f"Extracted positions for {policy_cluster['policy_area']}:")
            print(positions_response)

            # Parse bullet points
            bullet_points = re.findall(r'(?:^|\n)- .+?(?=\n-|\n\n|$)', positions_response, re.DOTALL)
            bullet_points = [point.strip() for point in bullet_points]

            # Add to policy cluster data
            policy_cluster['positions'] = bullet_points
            policy_cluster['position_count'] = len(bullet_points)

            # STEP 4: Generate a concise analysis of each policy cluster
            analysis_prompt = f"""
Based on the extracted positions, create a concise, well-structured analysis of {self.politician_name}'s stance on {policy_cluster['policy_area']}.

Policy cluster: {policy_cluster['policy_area']}
Component policy areas: {components_str}
Cluster description: {policy_cluster['description']}

Extracted positions and statements:
{positions_response}

Please provide a comprehensive analysis that:
1. Summarizes {self.politician_name}'s overall position or approach to this policy area
2. Identifies key themes, consistencies, or contradictions in their stance
3. Notes how their position has evolved over time (if applicable)
4. References specific statements or actions, citing the media sources
5. Analyzes how media outlets frame their position

Format your analysis as a concise, well-structured summary of 3-4 paragraphs.
Focus on providing a specific positions and quotes which clearly illustrate {self.politician_name}'s overall position on {policy_cluster['policy_area']}.
Adopt a neutral writing style, your output should not be overly favourable or defavourable to {self.politician_name}. It should be objective.
Include specific references to media sources in [brackets] to support your points.
    """

            analysis_chatbot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1900
            )

            policy_analysis = analysis_chatbot.ask(analysis_prompt)
            print(f"Analysis for {policy_cluster['policy_area']}:")
            print(policy_analysis)
            policy_cluster['analysis'] = policy_analysis

        # Save policy positions
        policy_positions_path = os.path.join(
            self.general_folder, "Outputs", "PoliticianAnalysis", "PolicyPositions",
            f"PolicyPositions_{self.politician_name.replace(' ', '_')}.json"
        )

        with open(policy_positions_path, 'w', encoding='utf-8') as f:
            json.dump(policy_clusters, f, indent=2)

        self.has_policy_positions = True
        self.save_processing_status()

        # Generate the nicely formatted markdown output file
        self._generate_policy_clusters_md(policy_clusters)

        return policy_clusters

    def _generate_policy_clusters_md(self, policy_clusters):
        """
        Generate a nicely formatted markdown file for the policy analysis results.

        Args:
            policy_clusters (List[Dict]): The policy clusters data
        """
        logging.info(f"Generating policy clusters markdown file for {self.politician_name}")

        # Create the output path
        output_dir = os.path.join(self.general_folder, "Outputs", "PoliticianAnalysis", "PolicyPositions")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"PolicyClusters_{self.politician_name.replace(' ', '_')}.md"
        )

        # Get the date of analysis
        current_date = datetime.now().strftime("%B %d, %Y")

        # Count total positions across all clusters
        total_positions = sum(len(cluster.get('positions', [])) for cluster in policy_clusters)

        # Generate chart first to ensure it exists for the document
        try:
            chart_path = self._generate_policy_distribution_chart(policy_clusters)
            chart_relative_path = os.path.relpath(
                chart_path, 
                os.path.dirname(output_path)
            )
        except Exception as e:
            logging.warning(f"Could not generate policy distribution chart: {str(e)}")
            chart_relative_path = f"../../../Outputs/Visualizations/PolicyDistribution_{self.politician_name.replace(' ', '_')}.png"

        # Generate introduction using AI
        try:
            introduction = self._generate_introduction(policy_clusters)
        except Exception as e:
            logging.warning(f"Could not generate introduction: {str(e)}")
            introduction = f"This document presents an analysis of {self.politician_name}'s policy positions based on media coverage, organized into {len(policy_clusters)} key policy clusters. The analysis synthesizes information from {total_positions} distinct policy statements or actions reported in media sources."

        # Start building the markdown content
        md_content = f"""# Policy Position Analysis: {self.politician_name}

## Overview

{introduction}

## Policy Mentions Distribution

The following chart shows the distribution of {self.politician_name}'s policy mentions across major policy areas:

![Policy Mentions Distribution]({chart_relative_path})

<div style="page-break-after: always;"></div>

## Table of Contents

"""

        # Add table of contents
        for i, cluster in enumerate(policy_clusters, 1):
            md_content += f"{i}. [{cluster['policy_area']}](#{cluster['policy_area'].lower().replace(' ', '-')})\n"

        md_content += "\n---\n\n"

        # Add detailed cluster analyses
        for i, cluster in enumerate(policy_clusters, 1):
            position_count = len(cluster.get('positions', []))
            component_count = len(cluster.get('components', []))

            # Create section anchor
            anchor = cluster['policy_area'].lower().replace(' ', '-')
            new_section = f"""<a id="{anchor}"></a>
## {i}. {cluster['policy_area']}

### Overview
{cluster['description']}

**Media Coverage**: {position_count} related mentions of positions/statements identified

### Positions and Statements
{cluster['analysis']}

"""
            new_section = new_section.replace('---', '')
            md_content += "\n"
            md_content += new_section
            md_content += "\n"

        # Generate conclusion using AI
        try:
            conclusion = self._generate_conclusion(policy_clusters)
        except Exception as e:
            logging.warning(f"Could not generate conclusion: {str(e)}")
            conclusion = f"This analysis presents a comprehensive overview of {self.politician_name}'s policy positions across {len(policy_clusters)} major policy areas. The data is derived from media coverage and provides insights into how {self.politician_name}'s political stance is portrayed in public discourse."

        # Add conclusion section
        md_content += f"""## Conclusion

{conclusion}
"""

        try:
            # Import the processor if not already imported
            from Classes.DocumentProcessor import CitationProcessor
            processor = CitationProcessor()
            md_content = processor.process_citations_in_markdown(md_content)
            logging.info("Successfully processed citations to superscript format")
        except Exception as e:
            logging.error(f"Error processing citations: {str(e)}")
            # Continue with unprocessed content if citation processing fails

        # Write the markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logging.info(f"Policy clusters markdown file generated at {output_path}")
        return output_path

    def _generate_introduction(self, policy_clusters):
        """Generate an AI-powered introduction for the policy analysis."""
        try:
            chatbot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1000
            )

            # Prepare prompt with summarized policy data
            policy_summary = "\n".join([
                f"- {cluster['policy_area']}: {len(cluster.get('positions', []))} mentions, covering: {', '.join(cluster.get('components', []))[:100]}..."
                for cluster in policy_clusters
            ])

            prompt = f"""
Generate a concise introduction (2-3 paragraphs) for a policy position analysis document about {self.politician_name}.
Include:
1. A brief contextual overview of {self.politician_name}'s role
2. The significance of analyzing their policy positions
3. A summary of the key policy areas covered in the analysis
4. The methodology (analysis of media coverage)

Policy clusters analyzed:
{policy_summary}

The introduction should be informative, neutral, and professional in tone.
    """

            introduction = chatbot.ask(prompt)
            return introduction

        except Exception as e:
            logging.error(f"Error generating introduction: {str(e)}")
            raise

    def _generate_conclusion(self, policy_clusters):
        """Generate an AI-powered conclusion for the policy analysis."""
        try:
            chatbot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1000
            )

            # Get top 3 policy areas by position count
            sorted_clusters = sorted(
                policy_clusters, 
                key=lambda x: len(x.get('positions', [])), 
                reverse=True
            )
            top_clusters = sorted_clusters[:min(3, len(sorted_clusters))]

            top_summary = "\n".join([
                f"- {cluster['policy_area']}: {len(cluster.get('positions', []))} mentions"
                for cluster in top_clusters
            ])

            prompt = f"""
Generate a concise conclusion (2-3 paragraphs) for a policy position analysis document about {self.politician_name}.
Include:
1. A summary of the most significant policy areas based on media coverage
2. Key patterns or themes observed across policy areas
3. The broader implications for understanding {self.politician_name}'s political stance
4. The value of this analysis for stakeholders and the public

Top policy areas by media mentions:
{top_summary}

The conclusion should be informative, neutral, and professional in tone.
    """

            conclusion = chatbot.ask(prompt)
            return conclusion

        except Exception as e:
            logging.error(f"Error generating conclusion: {str(e)}")
            raise
    
    def _generate_policy_distribution_chart(self, policy_clusters):
        """
        Generate a visualization of policy distribution across clusters.

        Args:
            policy_clusters (List[Dict]): The policy clusters data
        """
        viz_dir = os.path.join(self.general_folder, "Outputs", "Visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Extract data for visualization
        labels = [cluster['policy_area'] for cluster in policy_clusters]
        sizes = [len(cluster.get('positions', [])) for cluster in policy_clusters]

        # Only proceed if we have data
        if sum(sizes) == 0:
            logging.warning("No policy positions to visualize")
            return

        # Convert to pandas Series for compatibility with create_professional_pie
        from pandas import Series
        data = Series(sizes, index=labels)

        # Create a professional pie chart using the helper function
        try:
            from Utils.Helpers import create_professional_pie, create_custom_colormap, get_text_color
            fig = create_professional_pie(
                data=data,
                title=f"{self.politician_name}'s Policy Mentions Distribution",
                figsize=(15, 10)
            )
        except ImportError:
            # Fallback to standard matplotlib pie chart if helper functions not available
            fig = plt.figure(figsize=(15, 10))
            plt.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, 
                    shadow=False, explode=[0.05] * len(sizes))
            plt.axis('equal')
            plt.title(f"{self.politician_name}'s Policy Mentions Distribution", fontsize=16, pad=20)

            # Add a legend with percentages
            percentages = [100 * size / sum(sizes) for size in sizes]
            legend_labels = [f"{label} ({size} mentions, {percentage:.1f}%)" 
                             for label, size, percentage in zip(labels, sizes, percentages)]

            plt.legend(legend_labels, loc="best", bbox_to_anchor=(1, 0.5), fontsize=10)
            plt.tight_layout()

        # Save the chart
        chart_path = os.path.join(viz_dir, f"PolicyDistribution_{self.politician_name.replace(' ', '_')}.png")
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.close()

        logging.info(f"Policy distribution chart saved to {chart_path}")
        return chart_path
    
    def analyze_stakeholders(self):
        """
        Analyze the politician's relationships with stakeholders mentioned in the articles.
        Identifies people and organizations relevant to the politician, categorizes them,
        and generates comprehensive stakeholder analysis.

        Returns:
            dict: Structured stakeholder analysis with categorization and relationship mapping
        """
        logging.info(f"Beginning stakeholder analysis for {self.politician_name}")

        # Check if we already have processed stakeholder data
        stakeholder_path = os.path.join(
            self.general_folder, "Outputs", "PoliticianAnalysis", "Stakeholders",
            f"StakeholderAnalysis_{self.politician_name.replace(' ', '_')}.json"
        )

        if os.path.exists(stakeholder_path) and not self.force_reprocess:
            logging.info("Loading existing stakeholder analysis")
            with open(stakeholder_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Step 1: Extract stakeholders from each article
        all_stakeholders = []

        for i, article in enumerate(self.articles):
            try:
                logging.info(f"Extracting stakeholders from article {i+1}/{len(self.articles)}")
                article_content = article.get('content', '')
                article_metadata = f"Title: {article.get('title', 'Unknown')}\nDate: {article.get('date', 'Unknown')}\nMedia: {article.get('media_outlet', 'Unknown')}"

                chatbot = ChatGPT(
                    system_prompt=f"You are a political analyst expert in identifying stakeholders relevant to politicians.",
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=1500
                )

                stakeholder_prompt = f"""
Analyze the following article and identify all stakeholders (people and organizations) that are directly relevant to {self.politician_name}.

A stakeholder is considered relevant if they:
1. Have direct interaction with {self.politician_name}
2. Influence {self.politician_name}'s political decisions or career
3. Are influenced by {self.politician_name}'s actions or policies
4. Share political interests or opposition with {self.politician_name}
5. The stakeholder must be a person or an organization, not a general group or demographic. Exclude mentions of groups (e.g. women, children, employees, businesses, etc.)
6. Exclude {self.politician_name} from the list.

For each identified stakeholder, provide:
- Name (full, official name of the person or organization)
- Type (either "person" or "organization")
- Description (brief explanation of who this stakeholder is)

Format your response as a JSON list of objects:
[
    {{
    "name": "stakeholder name",
    "type": "person/organization",
    "description": "brief description"
    }},
    ...
]

Article Metadata:
{article_metadata}

Article Content:
{article_content}

If no stakeholders relevant to {self.politician_name} are found in this article, return an empty list: []
"""

                response = chatbot.ask(stakeholder_prompt)
                print(response)

                # Process the response to extract stakeholders
                try:
                    # Clean the response to ensure valid JSON
                    json_start = response.find('[')
                    json_end = response.rfind(']') + 1

                    if json_start != -1 and json_end != -1:
                        cleaned_json = response[json_start:json_end]
                        article_stakeholders = json.loads(cleaned_json)

                        # Add article reference to each stakeholder
                        for stakeholder in article_stakeholders:
                            stakeholder['article_ref'] = {
                                'title': article.get('title', 'Unknown'),
                                'date': article.get('date', 'Unknown'),
                                'media_outlet': article.get('media_outlet', 'Unknown')
                            }
                            stakeholder['content'] = article.get('content', '')

                        all_stakeholders.extend(article_stakeholders)
                        logging.info(f"Extracted {len(article_stakeholders)} stakeholders from article {i+1}")
                    else:
                        logging.warning(f"No valid JSON found in response for article {i+1}")
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse stakeholder JSON for article {i+1}")
                    continue
                
            except Exception as e:
                logging.error(f"Error extracting stakeholders from article {i+1}: {str(e)}")
                continue
            
        # Deduplicate stakeholders by name/type
        unique_stakeholders = {}
        for stakeholder in all_stakeholders:
            key = f"{stakeholder['name']}|{stakeholder['type']}"
            if key not in unique_stakeholders:
                unique_stakeholders[key] = {
                    'name': stakeholder['name'],
                    'type': stakeholder['type'],
                    'description': stakeholder['description'],
                    'article_refs': [stakeholder['article_ref']],
                    'contents': [stakeholder['content']]
                }
            else:
                unique_stakeholders[key]['article_refs'].append(stakeholder['article_ref'])
                unique_stakeholders[key]['contents'].append(stakeholder['content'])

        stakeholders_list = list(unique_stakeholders.values())
        print(stakeholders_list)
        logging.info(f"Identified {len(stakeholders_list)} unique stakeholders after deduplication")

        # Step 2: Enhance stakeholder information with relationship data
        enhanced_stakeholders = []

        for i, stakeholder in enumerate(stakeholders_list):
            try:
                logging.info(f"Enhancing stakeholder data for {stakeholder['name']} ({i+1}/{len(stakeholders_list)})")

                # Join article contents for context, but limit size
                combined_content = "\n\n".join(stakeholder['contents'])
                if len(combined_content) > 10000:  # Limit to ~10k characters
                    combined_content = combined_content[:10000] + "..."

                chatbot = ChatGPT(
                    system_prompt=f"You are a political analyst specializing in stakeholder relationships.",
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=800
                )

                relationship_prompt = f"""
Analyze the relationship between {self.politician_name} and the stakeholder {stakeholder['name']} ({stakeholder['type']}) based on the following article content.

Stakeholder description: {stakeholder['description']}

Extract the following information:
1. Role/Position: The stakeholder's professional role/position (specific title/role) or organizational type.
2. Relationship Type: How the stakeholder relates to {self.politician_name} (e.g., ally, opponent, mentor, colleague, constituent)
3. Influence Description: How this stakeholder influences or is influenced by {self.politician_name} in the political landscape

Format your response as a JSON object:
{{
    "role": "stakeholder's role or position",
    "relationship_type": "type of relationship with the politician",
    "influence_description": "description of political influence/relevance"
}}

Article content:
{combined_content}
"""

                response = chatbot.ask(relationship_prompt)
                print(response)

                # Process the response to extract relationship info
                try:
                    # Clean the response to ensure valid JSON
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1

                    if json_start != -1 and json_end != -1:
                        cleaned_json = response[json_start:json_end]
                        relationship_data = json.loads(cleaned_json)

                        # Merge relationship data with stakeholder info
                        enhanced_stakeholder = {
                            **stakeholder,
                            'role': relationship_data.get('role', 'Unknown'),
                            'relationship_type': relationship_data.get('relationship_type', 'Unknown'),
                            'influence_description': relationship_data.get('influence_description', ''),
                            'mentions': len(stakeholder['article_refs'])
                        }

                        # Remove verbose content data to keep output manageable
                        enhanced_stakeholder.pop('contents', None)

                        enhanced_stakeholders.append(enhanced_stakeholder)

                    else:
                        logging.warning(f"No valid JSON found in relationship response for {stakeholder['name']}")
                        # Include stakeholder with default relationship values
                        stakeholder['role'] = 'Unknown'
                        stakeholder['relationship_type'] = 'Unknown'
                        stakeholder['influence_description'] = ''
                        stakeholder['mentions'] = len(stakeholder['article_refs'])
                        stakeholder.pop('contents', None)
                        enhanced_stakeholders.append(stakeholder)

                except json.JSONDecodeError:
                    logging.error(f"Failed to parse relationship JSON for {stakeholder['name']}")
                    # Include stakeholder with default relationship values
                    stakeholder['role'] = 'Unknown'
                    stakeholder['relationship_type'] = 'Unknown'
                    stakeholder['influence_description'] = ''
                    stakeholder['mentions'] = len(stakeholder['article_refs'])
                    stakeholder.pop('contents', None)
                    enhanced_stakeholders.append(stakeholder)

            except Exception as e:
                logging.error(f"Error enhancing stakeholder data for {stakeholder['name']}: {str(e)}")
                # Include stakeholder with default relationship values
                stakeholder['role'] = 'Unknown'
                stakeholder['relationship_type'] = 'Unknown'
                stakeholder['influence_description'] = ''
                stakeholder['mentions'] = len(stakeholder['article_refs'])
                stakeholder.pop('contents', None)
                enhanced_stakeholders.append(stakeholder)

        # Step 3: Define stakeholder categories
        # Build a descriptive list of stakeholders for categorization
        stakeholder_descriptions = []
        for s in enhanced_stakeholders:
            desc = f"Name: {s['name']}, Type: {s['type']}, Role: {s['role']}, " \
                   f"Relationship: {s['relationship_type']}, Description: {s['description']}, " \
                   f"Influence: {s['influence_description']}"
            stakeholder_descriptions.append(desc)

        all_descriptions = "\n\n".join(stakeholder_descriptions)

        chatbot = ChatGPT(
            system_prompt=f"You are a political analyst specializing in stakeholder categorization and mapping.",
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1500
        )

        categories_prompt = f"""
Based on the following list of stakeholders related to {self.politician_name}, create 6-9 distinct categories to classify them.

Each category should group stakeholders based on their relationship to {self.politician_name}, their sector, influence type, or other relevant factors.

For each category you define, provide:
1. A clear, specific category name
2. A detailed description of the category and its significance to {self.politician_name}
3. The types of stakeholders that would belong in this category
4. The political implications of this stakeholder group for {self.politician_name}

Format your response as a JSON list:
[
    {{
    "category": "Category Name",
    "description": "Detailed category description",
    "stakeholder_types": "Types of stakeholders in this category",
    "political_implications": "Political implications of this stakeholder group"
    }},
    ...
]

Stakeholder List:
{all_descriptions}
"""

        categories_response = chatbot.ask(categories_prompt)
        print(categories_response)

        # Process the response to extract categories
        try:
            # Clean the response to ensure valid JSON
            json_start = categories_response.find('[')
            json_end = categories_response.rfind(']') + 1

            if json_start != -1 and json_end != -1:
                cleaned_json = categories_response[json_start:json_end]
                stakeholder_categories = json.loads(cleaned_json)
                logging.info(f"Created {len(stakeholder_categories)} stakeholder categories")
            else:
                logging.warning("No valid category JSON found in response")
                stakeholder_categories = [
                    {"category": "Political Associates", "description": "Direct political connections", "stakeholder_types": "Politicians, party members", "political_implications": "Direct influence on political agenda"},
                    {"category": "Government Entities", "description": "Government-related organizations", "stakeholder_types": "Departments, agencies", "political_implications": "Execution of policies"},
                    {"category": "Media Relations", "description": "Media entities and personnel", "stakeholder_types": "Journalists, news outlets", "political_implications": "Public image management"},
                    {"category": "Civil Society", "description": "Non-governmental organizations", "stakeholder_types": "NGOs, advocacy groups", "political_implications": "Public support mobilization"},
                    {"category": "Constituents", "description": "The public and voters", "stakeholder_types": "Voters, community members", "political_implications": "Electoral support"},
                    {"category": "Business Connections", "description": "Private sector entities", "stakeholder_types": "Companies, business leaders", "political_implications": "Economic policy influence"}
                ]
        except json.JSONDecodeError:
            logging.error("Failed to parse category JSON")
            stakeholder_categories = [
                {"category": "Political Associates", "description": "Direct political connections", "stakeholder_types": "Politicians, party members", "political_implications": "Direct influence on political agenda"},
                {"category": "Government Entities", "description": "Government-related organizations", "stakeholder_types": "Departments, agencies", "political_implications": "Execution of policies"},
                {"category": "Media Relations", "description": "Media entities and personnel", "stakeholder_types": "Journalists, news outlets", "political_implications": "Public image management"},
                {"category": "Civil Society", "description": "Non-governmental organizations", "stakeholder_types": "NGOs, advocacy groups", "political_implications": "Public support mobilization"},
                {"category": "Constituents", "description": "The public and voters", "stakeholder_types": "Voters, community members", "political_implications": "Electoral support"},
                {"category": "Business Connections", "description": "Private sector entities", "stakeholder_types": "Companies, business leaders", "political_implications": "Economic policy influence"}
            ]

        # Step 4: Classify each stakeholder into one of the categories
        categorized_stakeholders = []

        for s in enhanced_stakeholders:
            category_names = [cat["category"] for cat in stakeholder_categories]
            category_descriptions = "\n".join([f"{cat['category']}: {cat['description']}" for cat in stakeholder_categories])

            chatbot = ChatGPT(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=200
            )

            classify_prompt = f"""
Classify the following stakeholder into exactly one of these categories:
{category_descriptions}

Stakeholder:
- Name: {s['name']}
- Type: {s['type']}
- Role: {s['role']}
- Relationship with {self.politician_name}: {s['relationship_type']}
- Description: {s['description']}
- Influence: {s['influence_description']}

Respond with only the exact category name and nothing else. Choose from: {', '.join(category_names)}
    """

            try:
                category_response = chatbot.ask(classify_prompt).strip()
                print(category_response)

                # Find closest matching category
                if category_response in category_names:
                    matched_category = category_response
                else:
                    # Find closest match if exact match not found
                    matched_category = category_names[0]  # Default to first category
                    for cat_name in category_names:
                        if cat_name.lower() in category_response.lower():
                            matched_category = cat_name
                            break
                        
                # Add category to stakeholder
                s['category'] = matched_category
                categorized_stakeholders.append(s)

            except Exception as e:
                logging.error(f"Error classifying stakeholder {s['name']}: {str(e)}")
                s['category'] = "Uncategorized"
                categorized_stakeholders.append(s)

        # Step 5: Perform stakeholder mapping analysis for each category
        category_analyses = {}

        for category in stakeholder_categories:
            category_name = category["category"]
            stakeholders_in_category = [s for s in categorized_stakeholders if s['category'] == category_name]

            if not stakeholders_in_category:
                category_analyses[category_name] = {
                    "category_info": category,
                    "stakeholders": [],
                    "analysis": "No stakeholders identified in this category."
                }
                continue
            
            # Format stakeholder data for analysis
            stakeholder_data = []
            for s in stakeholders_in_category:
                stakeholder_data.append({
                    "name": s['name'],
                    "type": s['type'],
                    "role": s['role'],
                    "relationship_type": s['relationship_type'],
                    "mentions": s['mentions'],
                    "description": s['description'],
                    "influence_description": s['influence_description']
                })

            chatbot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=2000
            )

            analysis_prompt = f"""
Perform a comprehensive stakeholder mapping analysis for the following group of stakeholders related to {self.politician_name}.

These stakeholders belong to the category: "{category_name}"
Category description: {category["description"]}

Stakeholders in this category:
{json.dumps(stakeholder_data, indent=2)}

Your analysis should include:

1. Power-Interest Analysis:
    - Identify which stakeholders have high/low power and high/low interest regarding {self.politician_name}.
    - Explain the basis for these power and interest assessments.
    - Use these columns to format this part of the assessment: | Stakeholder | Power | Interest | Basis for Assessment |

2. Combined Network and Influence Analysis:
    - Provide an integrated analysis that merges network mapping with a detailed influence assessment.
    - Identify key network connectors and central influencers as well as any isolated or peripheral stakeholders, highlighting how these relationships could lead to alliances or conflicts.
    - Describe the communication flows and information exchange dynamics within the network.
    - Clearly pinpoint specific stakeholders capable of exerting direct influence on {self.politician_name}, detailing the nature of their influence (supportive, adversarial, or conditional) and explaining the factors and conditions that might shift their stance.
    - Format this section using a combination of narrative paragraphs and concise, descriptive bullet points that deliver rich information without excessive listing.

Format your response as a professionally written, comprehensive analysis with clear sections and insights. Start at the '###' header level as your output will be embedded in a markdown document. Use a mixture of bullet points and paragraphs to format your output. Do not use '---' symbols as delimiters.

Your output should only be the following two sections and nothing more: ###Power-Interest Analysis, ###Combined Network and Influence Analysis. No introduction or conclusion needed. Do not even include any concluding statement, stop straight after the last point in the Combined Network and Influence Analysis section.
"""

            analysis_response = chatbot.ask(analysis_prompt)
            print(analysis_response)

            category_analyses[category_name] = {
                "category_info": category,
                "stakeholders": stakeholders_in_category,
                "analysis": analysis_response
            }

        # Step 6: Create the final result structure
        result = {
            "stakeholders": categorized_stakeholders,
            "categories": stakeholder_categories,
            "category_analyses": category_analyses
        }

        # Save the results
        with open(stakeholder_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        # Generate markdown report
        self._generate_stakeholder_analysis_report(result)

        return result

    def _generate_stakeholder_analysis_report(self, stakeholder_data):
        """
        Generate a comprehensive markdown report of stakeholder analysis.
        """
        stakeholders = stakeholder_data["stakeholders"]
        categories = stakeholder_data["categories"]
        category_analyses = stakeholder_data["category_analyses"]

        # Setup output path
        output_path = os.path.join(
            self.general_folder, "Outputs", "PoliticianAnalysis", "Stakeholders",
            f"StakeholderAnalysisReport_{self.politician_name.replace(' ', '_')}.md"
        )

        # Generate visualization data for stakeholder categories
        category_counts = Counter([s['category'] for s in stakeholders])

        # Generate person vs organization counts
        type_counts = Counter([s['type'] for s in stakeholders])

        # Generate relationship type distribution
        relationship_counts = Counter([s['relationship_type'] for s in stakeholders])

        # Generate stakeholder visualization charts
        charts_folder = os.path.join(self.general_folder, "Outputs", "Visualizations")
        os.makedirs(charts_folder, exist_ok=True)

        # Create category distribution chart
        try:
            category_data = pd.Series(category_counts)
            category_chart_path = os.path.join(charts_folder, f"StakeholderCategories_{self.politician_name.replace(' ', '_')}.png")

            plt.figure(figsize=(10, 7))
            category_data.plot(kind='bar', color='skyblue')
            plt.title(f'Stakeholder Categories - {self.politician_name}')
            plt.ylabel('Number of Stakeholders')
            plt.xlabel('Category')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(category_chart_path)
            plt.close()

            category_chart_rel_path = os.path.relpath(
                category_chart_path, 
                os.path.dirname(output_path)
            )
        except Exception as e:
            logging.error(f"Error creating category chart: {str(e)}")
            category_chart_rel_path = None

        # Create type distribution chart (person vs organization)
        try:
            type_data = pd.Series(type_counts)
            type_chart_path = os.path.join(charts_folder, f"StakeholderTypes_{self.politician_name.replace(' ', '_')}.png")

            plt.figure(figsize=(8, 6))
            type_data.plot(kind='pie', autopct='%1.1f%%')
            plt.title(f'Stakeholder Types - {self.politician_name}')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(type_chart_path)
            plt.close()

            type_chart_rel_path = os.path.relpath(
                type_chart_path, 
                os.path.dirname(output_path)
            )
        except Exception as e:
            logging.error(f"Error creating type chart: {str(e)}")
            type_chart_rel_path = None

        # Start building the markdown report
        md_content = f"""# Stakeholder Analysis for {self.politician_name}

## Overview

This report provides a comprehensive analysis of stakeholders related to {self.politician_name} based on media coverage. The analysis identifies key individuals and organizations, categorizes them based on their relationship with {self.politician_name}, and provides strategic insights for each stakeholder group.

## Summary Statistics

- **Total Stakeholders Identified**: {len(stakeholders)}
- **Organizations**: {type_counts.get('organization', 0)}
- **Individuals**: {type_counts.get('person', 0)}
- **Stakeholder Categories**: {len(categories)}

## Stakeholder Distribution by Category

"""

        if category_chart_rel_path:
            md_content += f"![Stakeholder Categories]({category_chart_rel_path})\n\n"

        # Add category analyses
        md_content += """
## Category Analysis

This section provides a detailed analysis of each stakeholder category, including power-interest mapping, network analysis, and political impact assessment.

"""

        for category_name, analysis in category_analyses.items():
            category_info = analysis["category_info"]
            stakeholders_in_category = analysis["stakeholders"]
            analysis_text = analysis["analysis"]

            md_content += f"### {category_name}\n\n"
            md_content += f"{category_info['description']}"
            md_content += f"{category_info['political_implications']}\n\n"
            md_content += f"{analysis_text}\n\n"


            md_content += "\n\n---\n\n"

        # Generate the appendix content for the complete stakeholder list
        appendix_content = f"""# Stakeholder Appendix for {self.politician_name}

This appendix contains a complete list of all identified stakeholders.

| Name | Type | Category | Role | Relationship | Mentions |
|------|------|----------|------|-------------|----------|
"""

        for s in sorted(stakeholders, key=lambda x: x['name']):
            appendix_content += f"| {s['name']} | {s['type']} | {s['category']} | {s['role']} | {s['relationship_type']} | {s['mentions']} |\n"

        # Define a separate output path for the appendix
        appendix_path = os.path.join(
            self.general_folder, "Outputs", "PoliticianAnalysis", "Stakeholders",
            f"StakeholderAnalysisAppendix_{self.politician_name.replace(' ', '_')}.md"
        )

        # Save the appendix document
        with open(appendix_path, 'w', encoding='utf-8') as f:
            f.write(appendix_content)

        logging.info(f"Stakeholder appendix report saved to {appendix_path}")

        # Save the markdown report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logging.info(f"Stakeholder analysis report saved to {output_path}")