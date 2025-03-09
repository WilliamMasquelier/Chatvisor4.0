import os
import re
import json
import collections
import pandas as pd
import marvin
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
from Classes.SimplifiedChatbots import ChatGPT, BigSummarizerGPT
from Classes.DocumentProcessor import DocumentProcessor, CitationProcessor
from Utils.Helpers import *
import logging
import traceback
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import logging

# Helper with dependencies
def ensure_insights_exist(articles_sorted, general_folder, company_name, industry_of_interest, region, language):
    """Validate and fix insights for articles."""
    insights_missing = False
    
    for article in articles_sorted:
        title = article.get('title', 'Unknown Title').replace("/", " ").replace(":", " ")
        expected_path = f"{general_folder}/Outputs/IndividualInsights/Insights_{title}.md"
        
        # Check if insights_path is missing in article dict
        if not article.get('insights_path'):
            if os.path.exists(expected_path):
                # Path exists on disk but not in article dict
                article['insights_path'] = expected_path
                logging.info(f"Added missing insights_path to article: {title}")
            else:
                insights_missing = True
                logging.warning(f"Insights file missing for article: {title}")
        elif not os.path.exists(article['insights_path']):
            insights_missing = True
            logging.warning(f"Insights file missing at path: {article['insights_path']}")
    
    if insights_missing:
        logging.info("Some insights are missing, generating them now...")
        generate_insights_output(
            articles_sorted=articles_sorted,
            company_name=company_name,
            general_folder=general_folder,
            industry_of_interest=industry_of_interest,
            region=region,
            language=language
        )
        # Update paths after generation
        for article in articles_sorted:
            title = article.get('title', 'Unknown Title').replace("/", " ").replace(":", " ")
            article['insights_path'] = f"{general_folder}/Outputs/IndividualInsights/Insights_{title}.md"

# Function to generate the journalist list

def generate_journalist_list_output(articles_sorted: List[Dict], company_name: str, general_folder: str, language: str = "English") -> str:
    """
    Generate a markdown list of journalists and media outlets from the processed articles.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed and sorted articles
        company_name (str): Name of the company being analyzed
        general_folder (str): Base path for output files
        language (str): Output language for the list
        
    Returns:
        str: Generated markdown content
    """
    try:
        logging.info(f"Starting journalist list generation for {company_name}")
        
        # Generate markdown content
        logging.info("Generating markdown content")
        md_content = f"# List of media articles related to {company_name}\n\n"
        md_content += "## Overview\n\n"
        
        # Add summary statistics
        total_articles = len(articles_sorted)
        unique_authors = len(set(article.get('author_name', 'Unknown') for article in articles_sorted))
        unique_outlets = len(set(article.get('media_outlet', 'Unknown') for article in articles_sorted))
        
        md_content += f"- Total Articles: {total_articles}\n"
        md_content += f"- Unique Journalists: {unique_authors}\n"
        md_content += f"- Media Outlets: {unique_outlets}\n\n"
        
        md_content += "## Complete list of articles\n\n"
        
        # Sort articles by date (most recent first) while maintaining the original sorting as secondary criterion
        articles_list = sorted(
            articles_sorted,
            key=lambda x: (
                datetime.strptime(x.get('date', 'January 1, 2024'), '%B %d, %Y').timestamp(),
                x.get('reordered_position', float('inf'))
            ),
            reverse=True
        )

        # Generate the detailed list
        for article in articles_list:
            media_outlet = article.get('media_outlet', 'Unknown')
            author_name = article.get('author_name', 'Unknown')
            date = article.get('date', 'Unknown')
            title = article.get('title', 'Unknown')
            link = article.get('link', None)
            
            media_outlet_hyperlinked = f"[{media_outlet}]({link})" if link else media_outlet
            md_content += f"- **{author_name}**, {media_outlet_hyperlinked} ({date}): *{title}*\n"

        # Add journalist statistics section
        md_content += "\n## Journalist Statistics\n\n"
        author_counts = collections.Counter(
            article.get('author_name', 'Unknown') 
            for article in articles_sorted 
            if article.get('author_name', 'Unknown') != 'Unknown'
        )
        
        if author_counts:
            md_content += "### Top Contributors\n\n"
            for author, count in author_counts.most_common(10):  # Top 10 journalists
                md_content += f"- {author}: {count} articles\n"

        # Add media outlet statistics section
        md_content += "\n## Media Outlet Statistics\n\n"
        outlet_counts = collections.Counter(
            article.get('media_outlet', 'Unknown') 
            for article in articles_sorted
        )
        
        if outlet_counts:
            md_content += "### Coverage by Media Outlet\n\n"
            for outlet, count in outlet_counts.most_common():
                md_content += f"- {outlet}: {count} articles\n"

        # Save markdown file
        output_folder = f"{general_folder}/Outputs/CompiledOutputs"
        output_file_path = os.path.join(output_folder, f"Articles_reference_list_{company_name.replace(' ', '_')}.md")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file_path, 'w', encoding='utf-8') as md_file:
                md_file.write(md_content)
            logging.info(f"Journalist list saved successfully to {output_file_path}")
        except Exception as e:
            logging.error(f"Error saving journalist list: {str(e)}")
            raise

        return md_content

    except Exception as e:
        logging.error(f"Error generating journalist list: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# Function to generate insights for individual articles

def generate_insights_output(articles_sorted: List[Dict], company_name: str, general_folder: str, 
                           industry_of_interest: str, region: str, language: str = 'English') -> str:
    """
    Generate compiled insights from the processed articles with enhanced statistics overview and visualizations.
    """
    try:
        logging.info(f"Starting insights generation for {company_name}")
        
        # Calculate statistics for overview
        total_articles = len(articles_sorted)
        date_range = f"{min(article.get('date', 'Unknown') for article in articles_sorted)} to {max(article.get('date', 'Unknown') for article in articles_sorted)}"
        unique_outlets = len(set(article.get('media_outlet', 'Unknown') for article in articles_sorted))
        unique_authors = len(set(article.get('author_name', 'Unknown') for article in articles_sorted))
        
        # Count articles by outlet
        outlet_counts = collections.Counter(
            article.get('media_outlet', 'Unknown') 
            for article in articles_sorted
        )
        top_outlets = dict(outlet_counts.most_common(10))  # Increased to top 10 for better visualization
        
        # Generate monthly distribution data
        months_data = collections.defaultdict(int)
        for article in articles_sorted:
            try:
                date = datetime.strptime(article.get('date', ''), '%B %d, %Y')
                month_key = date.strftime('%B %Y')
                months_data[month_key] += 1
            except Exception:
                continue
        
        # Sort months chronologically
        sorted_months = sorted(months_data.keys(), 
                             key=lambda x: datetime.strptime(x, '%B %Y'))
        months_data_sorted = {month: months_data[month] for month in sorted_months}
        
        # Create charts
        outlets_chart = create_bar_chart_compiled_insights(
            top_outlets,
            f'Media Coverage Distribution - {company_name}',
            'Media Outlet',
            'Articles',
            rotate_labels=True,
            figsize=(12, 5),
            color='#2e86c1'
        )
        
        monthly_chart = create_bar_chart_compiled_insights(
            months_data_sorted,
            f'Timeline of Media Coverage - {company_name}',
            'Month',
            'Articles',
            rotate_labels=True,
            figsize=(12, 5),
            color='#27ae60'
        )
        
        # Initialize insights document with enhanced overview
        summary_insights_md = f"""
# Compiled Summaries of articles related to {company_name}

## Overview

### Coverage Statistics
| Metric | Value |
|--------|--------|
| Total Articles Analyzed | {total_articles} |
| Coverage Period | {date_range} |
| Media Outlets | {unique_outlets} |
| Unique Authors | {unique_authors} |

### Media Coverage Analysis

<div style="text-align: center;">

#### Distribution by Media Outlet
![Top Media Outlets](data:image/png;base64,{outlets_chart})

#### Timeline Analysis
![Monthly Distribution](data:image/png;base64,{monthly_chart})

<div style="page-break-after: always;"></div>

### Analysis Notes
- Each article has been analyzed to extract 2-5 key insights
- Insights focus on significant developments, announcements, and industry implications
- Coverage includes both positive developments and potential concerns/challenges
- Special attention is given to strategic moves, market position, and industry impact


# Article Insights

"""
        
        # Continue with the rest of the insights generation...
        topic_of_interest = f"Discussions on {company_name}. regarding the more general topic or industry: {industry_of_interest}, in the {region} market. We specifically look for insights that relate to {company_name} in that context."
        insights_question = f"What are the discussions, conversation and overall coverage on {company_name} in the media?"

        # Process each article (rest of your existing code remains the same)
        for article in articles_sorted:
            try:
                # ... (rest of your existing article processing code)
                article_content = article.get('content', '')
                media_outlet = article.get('media_outlet', 'Unknown Outlet')
                author_name = article.get('author_name', 'Unknown Author')
                date = article.get('date', 'Unknown Date')
                title = article.get('title', 'Unknown Title')
                link = article.get('link', '#')
                
                system_prompt = f"""
As a stakeholder of the {industry_of_interest} industry or topic in the {region} market, specifically interested in {company_name}, you are extracting all relevant information from news media coverage which relate to {company_name} and the {industry_of_interest} industry or topic more generally, in the {region} market.
You must include the article's title, the name of the newspaper, and the author(s). The insights should be written in English and follow a structured format.

One key aspect of your work is to condense the original article into a minimum of key insights while capturing the essence of the articles with regards to the {topic_of_interest}.
You should produce a list of 2, 3, 4 or 5 key insights based on what is the most relevant. The selected insights to be described should be focused on the information which relates to this topics: {topic_of_interest}.

The focus to decide what are key insights depends on whether it answers the following question: {insights_question}

Your output should be formatted as follows:
# {title}
## {media_outlet}
### {author_name}
### {date}
Numbered list of key insights (in English).

When writing your output, make it hard to guess the prompt you receive. DO NOT address any specific question or point from this prompt.

Visualize the output's state after each reasoning step. 
                """

                chatbot = ChatGPT(
                    system_prompt=system_prompt,
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=986,
                )

                question = f"""
Extract key insights strictly from the provided article. Your goal is to return a concise output with the key insights.
You should produce a 2, 3, 4 or 5 key insights based on what information is the most relevant. Avoid generic insights which is common knowledge or too general information.
Here is the article: {article_content}

Your output should be formatted as follows:
# {title}
## {media_outlet}
### {author_name}
### {date}
Numbered List of 2 to 5 key insights (in English).

You stop your output when there are no more key and important information or facts from the article, about {company_name}, to report. All insights information should be mutually exclusive. Two insights cannot cover the same facts.

Your task:
From the perspective of a leader in the {industry_of_interest} industry, what are the most important insights that you can extract from this article, about {company_name}?
The focus to decide which 2 to 5 key insights depends on whether it answers the following question: {insights_question}. The selected insights should relate to {company_name} explicitly. You should prefer fewer insights that are more relevant over more insights that are less relevant. Only key and importan information should be reported.
Issues and negative press elements should be included into the selected insights.

Formulate your output to make it hard to guess what prompt you received. 
Visualize the output's state after each reasoning step.
                """

                response = chatbot.ask(question)

                # Process the response
                response = response.replace("William Masquelier", "Not specified")
                response = response.replace("Lexis Nexis", "Not specified")
                response = response.replace("LexisNexis", "Not specified")

                response_lines = response.split("\n")
                if len(response_lines[0]) != 0:
                    title = response_lines[0].replace("#", "").strip()
                    response_lines[0] = f"# [{title}]({link})"
                else:
                    title = response_lines[1].replace("#", "").strip()
                    response_lines[1] = f"# [{title}]({link})"
                response_lines = "\n".join(response_lines)

                # Save individual insight
                title = article.get('title', 'Unknown Title').replace("/", " ").replace(":", " ")
                insights_path = f"{general_folder}/Outputs/IndividualInsights/Insights_{title}.md"
                
                with open(insights_path, "w", encoding='utf-8') as file:
                    file.write(response_lines)

                article['insights_path'] = insights_path
                article['insights_content'] = response_lines

                if 'insights_content' in article and language.lower() != 'english':
                    article['insights_content'] = translate_content(
                    article['insights_content'], 
                    'auto', 
                    language
                    )
                
                # Save translated individual insight
                title = article.get('title', 'Unknown Title').replace("/", " ").replace(":", " ")
                insights_path = f"{general_folder}/Outputs/IndividualInsights/Insights_{title}.md"
                
                with open(insights_path, "w", encoding='utf-8') as file:
                    file.write(article['insights_content'])

            except Exception as e:
                logging.error(f"Error processing article {title}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        # Save compiled insights

        all_insights = []
        for article in articles_sorted:
            if 'insights_content' in article:
                all_insights.append(article['insights_content'] + "\n\n---\n")

        summary_insights_md = "\n".join(all_insights)

        compiled_insights_path = f"{general_folder}/Outputs/CompiledOutputs/CompiledInsights_{company_name}.md"
        with open(compiled_insights_path, "w", encoding='utf-8') as file:
            file.write(summary_insights_md)

        # Save updated article data
        save_data_to_json(articles_sorted, f"{general_folder}/Outputs/CompiledOutputs/ArticlesList.json")

        logging.info(f"Insights generation completed successfully")
        return summary_insights_md

    except Exception as e:
        logging.error(f"Error generating insights: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# Function to generate issue analysis

def generate_issue_analysis_output(articles_sorted: List[Dict], company_name: str, general_folder: str, 
                           industry_of_interest: str, region: str, language: str ) -> str:
    """
    Generate comprehensive issues analysis from the processed articles.
    Uses article content directly with metadata instead of compiled insights.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed and sorted articles
        company_name (str): Name of the company being analyzed
        general_folder (str): Base path for output files
        industry_of_interest (str): Industry relevant to the analysis
        region (str): Geographic region of interest
        language (str): Output language
        
    Returns:
        str: Generated markdown content with comprehensive analysis
    """
    try:
        logging.info(f"Starting issues analysis generation for {company_name}")
        

        # Sort articles based on dates
        for article in articles_sorted:
            if 'date' in article:
                try:
                    date_object = datetime.strptime(article['date'], '%B %d, %Y')
                    article['timestamp'] = date_object.timestamp()
                except ValueError:
                    logging.warning(f"Could not parse date: {article['date']}")
                    article['timestamp'] = 0  # Set to epoch for invalid dates
            else:
                article['timestamp'] = 0  # Set to epoch for articles without dates

        # Sort articles by timestamp in ascending order (chronological)
        articles_sorted.sort(key=lambda x: x.get('timestamp', 0))

        # Compile articles content with metadata
        compiled_content = ""
        for article in articles_sorted:
            metadata_header = f"""
Here are the metadata and reference of the article content below:
# Title: {article.get('title', 'Untitled')}
## Media outlet: {article.get('media_outlet', 'Unknown Media Outlet')}
### Author: {article.get('author_name', 'Anonymous')}
### Date: {article.get('date', 'Unknown Date')}

"""
            content = article.get('content', '')

            metadata_footer = f"""
---
Here are the metadata and reference of the article above:
Media outlet: {article.get('media_outlet', 'Unknown')}
Author: {article.get('author_name', 'Anonymous')}
Date: {article.get('date', 'Unknown Date')}
Title: {article.get('title', 'Untitled')}
---

"""
            compiled_content += metadata_header + content + metadata_footer

        # Save compiled content to a temporary file for BigSummarizerGPT
        temp_compiled_path = f"{general_folder}/Outputs/CompiledOutputs/TempCompiled_{company_name}.md"
        with open(temp_compiled_path, "w", encoding='utf-8') as file:
            file.write(compiled_content)

        # Get business model description
        logging.info("Generating business model description")
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1822
        )

        business_model_question = f"""
Generate a comprehensive business model description for {company_name}, with a focus on its interactions within the market and its relationships with key stakeholders. Describe how {company_name} collaborates, competes, or forms partnerships with other companies in the industry ecosystem, including suppliers, distributors, regulators, and any strategic alliances. Highlight {company_name}'s approach to customer engagement and how it adapts its offerings to meet the needs of its target audience or specific market demands. Provide insights into how {company_name} navigates its competitive landscape and builds relationships that reinforce its position in the industry. Additionally, discuss the company's strategy for addressing external challenges such as regulatory changes, shifting customer expectations, and technological advancements. In this case, we are interested by {company_name}'s business model in {industry_of_interest}, and more specifically in the {region} market.
        """

        business_model_description = chatbot.ask(business_model_question)
        print(business_model_description)

        # Create list of issues using BigSummarizerGPT
        logging.info("Creating list of issues")
        list_issues_insights = f"# List of pains and issues extracted from the media coverage:\n"

        chatbot = BigSummarizerGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1822
        )

        question = f"""
Analyze the following compiled media coverage and extract only the negative issues related to {company_name} as individual bullet points. Follow these rules strictly:

Each bullet point must start on a new line with a single dash and a space ("- ").
Each bullet point must describe exactly one negative issue related to {company_name} and include a comprehensive description of the issue (covering details such as the problem, its consequences, and any affected stakeholders) all within that same line.
Be detailed and descriptive regarding the negative aspects and issues faced by {company_name}. You can report exact quotes or sentences from the article to support your points.
Do not insert any additional line breaks within a bullet point. If multiple sentences are necessary for clarity, combine them into one continuous line.
At the end of each bullet point, include the source reference in the exact format: [Media outlet, Author, date].
If no negative issues are identified, output a single bullet point stating: - No negative aspects or bad press found for {company_name}.
Your response must be formatted exclusively as a bullet point list, with every bullet point self-contained on one line. Each bullet point should end with the article reference which is displayed in the section header, for traceability, of the information input article text: [Media outlet, Author, date].      
"""

        response = chatbot.ask(question, temp_compiled_path)
        print(response)
        
        # Extract and format bullet points
        bullet_points = re.findall(r'(?:^|\n)- .+', response)
        bullet_points = [point.strip() for point in bullet_points]

        if bullet_points:
            list_issues_insights += "\n" + "\n".join(bullet_points) + "\n"
        else:
            list_issues_insights += "\n- No negative aspects or bad press found for this section.\n"

        list_issues_insights = list_issues_insights.replace(" -", "\n-")

        # Save issues list
        issues_list_path = f"{general_folder}/Outputs/CompiledOutputs/IssueListInsights_{company_name}.md"
        with open(issues_list_path, "w") as file:
            file.write(list_issues_insights)

        #with open(issues_list_path, "r") as file:
        #    list_issues_content = file.read()

        # Extract issue categories with descriptions
        logging.info("Extracting issue categories and descriptions")
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2000
        )

        category_question = f"""
Based on the following list of issues faced by {company_name}, identify distinct issue categories and provide descriptions for each.
Format your response strictly as follows:
CATEGORY: [Issue 1]
DESCRIPTION: [Detailed description of what this issue entails]
....
CATEGORY: [Issue N]
DESCRIPTION: [Detailed description of what this issue entails]

For each category:
1. Give it a clear, concise name that refers to the specific issue.
2. Provide a detailed description explaining what the issue(s) consist of.
3. Make sure categories are distinct and don't overlap. each issue should be exclusively distinct from the other listed "CATEGORY".
4. Focus on {company_name}'s specific context and issues.

Bullet point list of issues to analyze:
{list_issues_insights}

You should propose 5, 6, 7, 8 or 9 main issues based on the bullet point list of issues identified.
        """

        categories_response = chatbot.ask(category_question)
        print(categories_response)

        # Parse categories and descriptions
        categories_data = []
        current_category = None
        current_description = None

        for line in categories_response.split('\n'):
            if line.startswith('CATEGORY:'):
                if current_category is not None:
                    categories_data.append({
                        'category': current_category.strip(),
                        'description': current_description.strip() if current_description else '',
                        'issues': []
                    })
                current_category = line.replace('CATEGORY:', '').strip().replace('*', '')
                print(current_category)
                current_description = None
            elif line.startswith('DESCRIPTION:'):
                current_description = line.replace('DESCRIPTION:', '').strip()

        # Add the last category
        if current_category is not None:
            categories_data.append({
                'category': current_category,
                'description': current_description if current_description else '',
                'issues': []
            })

        # Process each bullet point and classify it
        logging.info("Classifying individual issues")
        bullet_points = [point.strip() for point in list_issues_insights.split('\n') if point.strip().startswith('-')]

        for bullet_point in bullet_points:
            classification_prompt = ""
            for category in categories_data:
                classification_prompt += f"\n{category['category']}: {category['description']}"

            category_names = ", ".join(cat['category'] for cat in categories_data)

            chatbot = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=100
            )
            classification = chatbot.ask(
                f"""
Given the following issue categories and their descriptions:
{classification_prompt}

Classify the following issue into one of these categories. Choose the most appropriate category based on the descriptions provided.

Issue to classify:
{bullet_point}

Only output the exact category name that best matches this issue. Your output should only be one of the following values: {category_names}. Nothing else.
                """
            )
            print(classification)

            # Add the classified issue to the appropriate category
            for category in categories_data:
                if category['category'] == classification:
                    category['issues'].append(bullet_point)
                    break
                
        # Format the categorized issues
        categorized_issues = "# Categorized Issues Analysis\n\n"
        for category in categories_data:
            categorized_issues += f"## {category['category']}\n"
            categorized_issues += f"**Description**: {category['description']}\n\n"
            categorized_issues += "**Issues Identified**:\n"
            for issue in category['issues']:
                categorized_issues += f"{issue}\n"
            categorized_issues += "\n---\n\n"

        # Save categorized issues
        categorized_issues_path = f"{general_folder}/Outputs/CompiledOutputs/CategorizedIssues_{company_name}.md"
        with open(categorized_issues_path, "w", encoding='utf-8') as file:
            file.write(categorized_issues)

        # Save raw categories data for further processing
        categories_json_path = f"{general_folder}/Outputs/CompiledOutputs/IssuesCategories_{company_name}.json"
        with open(categories_json_path, 'w', encoding='utf-8') as file:
            json.dump(categories_data, file, indent=4)

        # Generate comprehensive analysis by category
        logging.info("Generating comprehensive category-based analysis")

        # Calculate total number of issues and identify major categories
        total_issues = sum(len(category['issues']) for category in categories_data)
        categories_with_counts = [
            {
                **category,
                'issue_count': len(category['issues']),
                'percentage': (len(category['issues']) / total_issues) * 100
            }
            for category in categories_data
        ]

        # Sort categories by issue count
        sorted_categories = sorted(
            categories_with_counts,
            key=lambda x: x['issue_count'],
            reverse=True
        )

        # Calculate total issues from all categories
        total_issues = sum(cat['issue_count'] for cat in categories_with_counts)

        # Split categories into major (top 80% issues) and minor (bottom 20% issues)
        cumulative = 0
        major_categories = []
        minor_categories = []
        for cat in sorted_categories:
            cumulative += cat['issue_count']
            # If the cumulative proportion is still at or below 80%, add as major.
            # (If you prefer to include the category that pushes you over 80% into major, you can adjust this condition.)
            if cumulative / total_issues <= 0.8:
                major_categories.append(cat)
            else:
                minor_categories.append(cat)

        cumulative_percentage_major = (sum(cat['issue_count'] for cat in major_categories) / total_issues) * 100
        logging.info(f"Selected {len(major_categories)} major categories covering {cumulative_percentage_major:.1f}% of issues")


        # Initialize comprehensive analysis
        header = f"# Executive Summary of Issues and Negative Press Related to {company_name}"

        toc = "## Table of Contents\n"
        for category in major_categories:
            anchor = create_markdown_anchor(category['category'])
            toc += f"- [{category['category']}](#{anchor})\n"

        # Analyze each major category sequentially
        previous_analyses = ""
        comprehensive_analysis = ""
        for idx, category in enumerate(major_categories, 1):
            logging.info(f"Analyzing category {idx}/{len(major_categories)}: {category['category']}")

            chatbot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=3000
            )

            category_analysis_prompt = f"""
You are conducting a comprehensive analysis of issues related to {company_name}, organized category by category. This is Analysis #{idx} of {len(major_categories)}.

Context:
Current Category: {category['category']}
Category Description: {category['description']}
Business Model Overview: {business_model_description}

Background Information:
Previous Analyses Overview:
{previous_analyses}

Issues Identified in this Category:
{chr(10).join(category['issues'])}

Aim to utilise a maximum of the sources and information provided above.

Your Task:
Develop a detailed analysis for this category that adheres to the following requirements:

- Builds upon Previous Analyses: Reference and expand on insights from earlier analyses where relevant, ensuring minimal repetition. However, do not mention in your output that the analysis is built upon previous analyses, only take this into account for writing the analyses down.
- Explains the issue in a chronological and referenced manner. Cite all sources using the following format: [Media Outlet, Date]. the sources must be cited in between squared brackets [].
- Evaluates Severity and Impact: Assess how the identified issues affect {company_name} in terms of operations, reputation, and stakeholder trust. 
- Stakeholder Analysis: Identify and describe the interests and concerns of key stakeholders linked to this category. Cite your source between suared brackets [Media Outlet, Date].
- Implications for the Business Model: Analyze how the identified issues in this category could influence or challenge {company_name}’s business model.
- Use Specific Examples: Incorporate relevant quotes or examples from the issues list to strengthen the analysis. Cite your source between suared brackets [Media Outlet, Date].
- Cite Sources: Properly attribute each issue to its source using the following format: [Media Outlet, Date]. aim to reference most of your sentences when relevant to do so.

Formatting Guidelines, Your analysis should follow this structure:

#### Introduction: Provide a brief summary of the category and its relevance to {company_name}.  Write this section as a plain text with only the following title only: #### Introduction

#### Chronological developments: describe how the issue developed along time.  Write this section as a plain text with only the following title only: #### Chronological developments

#### Impact Analysis and Business Model Implications: Impact Assessment should consider : Reputation risk: What is the potential impact on the organization's image or reputation? Operational impact: Could the issue affect business operations or productivity? Financial implications: Are there any financial consequences, such as lost revenue or lawsuits? Regulatory or legal concerns: Does the issue involve legal or regulatory violations? Discuss potential short- and long-term effects on {company_name}’s business model. (Do not directly respond or adress these questions in your output).  Write this section as a plain text with only the following title only: #### Impact Analysis and Business Model Implications

#### Stakeholder Interests: Detail the perspectives and priorities of stakeholders involved. Stakeholder Perspectives should include: Key audiences: Who are the primary and secondary audiences affected by or interested in the issue? Sentiments: How do different stakeholders perceive the issue? Reactions: What are stakeholders saying or doing in response to the issue? Expectations: What do stakeholders expect from the organization at this time? Reputational and Communications Risks: Examine potential impacts on {company_name}’s reputation and public communication strategies. (Do not directly respond or adress these questions in your output) Write this section as a plain text with only the following title only: #### Stakeholder Interests

Cite your references [in between] squared brackets[]. Use this format from the provided bullet point list: [Media outlet, Author, date] or [Media outlet, date] if no real author are provided in the relevant bullet point. Use this format: [Media outlet, date], if the author is anonymous.
Do not adress this prompt directly in your output. Take your time to reflect.
"""

            category_response = chatbot.ask(category_analysis_prompt)
            print(category_response)

            anchor = create_markdown_anchor(category['category'])
            comprehensive_analysis += f"\n\n<h2 id='{anchor}'>{category['category']} ({category['percentage']:.1f}% of Issues)</h2>\n\n"
            comprehensive_analysis += "\n\n"
            comprehensive_analysis += f"{category_response}\n\n\n"
            previous_analyses += f"\nCategory: {category['category']}\n{category_response}\n\n\n"

        comprehensive_analysis += """\n<div style="page-break-after: always;"></div>\n"""

        # Generate overall conclusion
        conclusion_prompt = f"""
Based on the complete analysis of major issue categories for {company_name}, create a concluding section that:

1. Summarizes the most critical challenges across all categories
2. Identifies common themes or patterns
3. Assesses the overall severity of the issues
4. Discusses potential interactions between different categories
5. Provides a future outlook considering all analyzed issues

Previous analyses:
{previous_analyses}

Format your response as:
## Conclusion and Future Outlook
[Your analysis here]

Keep the conclusion to approximately 400 words.
        """

        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1000
        )

        conclusion = chatbot.ask(conclusion_prompt)
        print(conclusion)

        # After generating the conclusion but before assembling the final analysis, add:

        # Generate introduction
        logging.info("Generating executive introduction")
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1000
        )

        introduction_prompt = f"""
Based on the complete analysis of issues faced by {company_name}, create a compelling introduction that:

1. Provides context about {company_name}'s position in the {industry_of_interest} industry, specifically in the {region} market
2. Summarizes the scope and methodology of the analysis
3. Highlights the key findings and most critical issues identified
4. Outlines the structure of the report
5. Sets appropriate expectations for readers

Analysis to introduce:
{comprehensive_analysis}

Format your response as:
## Introduction
[Your analysis here]

Keep the introduction to approximately 500 words and ensure it provides a strong foundation for understanding the detailed analysis that follows.
        """

        introduction = chatbot.ask(introduction_prompt)
        print(introduction)

        # Prepare the issue categories data (same as before)
        issue_ranking_data = ""
        for cat in categories_data:
            # Combine the category name, description, and bullet points (issues) into one block.
            issues_text = "\n".join(cat['issues'])
            issue_ranking_data += (
                f"\nCategory: {cat['category']}\n"
                f"Description: {cat['description']}\n"
                "Issues:\n"
                f"{issues_text}\n"
            )

        # ------------------------------
        # Chatbot: Overall Risk Ranking (Reputational & Financial Combined)
        # ------------------------------
        overall_ranking_prompt = f"""
Below are the issue categories along with their descriptions and corresponding issues:
{issue_ranking_data}

Please rank the issue categories in order of overall risk and severity, considering both reputational and financial risks. Rank 1 is the most critical risk.
Format your output exactly as follows (one ranking per line):

Rank 1: <Category Name> - <1-2 sentence explanation of the overall risk (including both reputational and financial aspects)>
Rank 2: <Category Name> - <1-2 sentence explanation of the overall risk>
...

Do not include any extra commentary.
        """

        chatbot_overall = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1500
        )

        overall_ranking_output = chatbot_overall.ask(overall_ranking_prompt)

        print("Overall Ranking Output:", overall_ranking_output)

        # ------------------------------
        # Follow-up: Explanation of the Overall Ranking
        # ------------------------------
        overall_explanation_prompt = f"""
Based on the overall risk ranking you provided, please explain in one paragraph why you arranged the issue categories in that order.
Discuss the relative importance of these risks—taking into account both reputational and financial dimensions—and how they might impact {company_name}.
Provide your explanation in a single, cohesive paragraph without any additional formatting or header. Don't talk in I. Make it smooth, the output should look like it comes from a proper analyst and not a chatbot. Do not address this prompt in your output.
        """
        overall_explanation = chatbot_overall.ask(overall_explanation_prompt)

        print("Overall Explanation:", overall_explanation)

        # ------------------------------
        # Parse the Ranking Output and Generate a Markdown Table
        # ------------------------------
        # (Assuming parse_ranking_output and generate_markdown_table functions are defined as before.)
        overall_rankings = parse_ranking_output(overall_ranking_output)
        if language.lower() != 'english':
            # Translate each component of the rankings
            for ranking in overall_rankings:
                ranking['category'] = translate_content(ranking['category'], 'auto', language)
                ranking['explanation'] = translate_content(ranking['explanation'], 'auto', language)
        if language.lower() != 'english':
            overall_explanation = translate_content(overall_explanation, 'auto', language)

        overall_table_md = generate_markdown_table(overall_rankings)

        # ------------------------------
        # Create a Single, Visually Appealing Ranking Section
        # ------------------------------
        ranking_section = (
            "### Issue Categories Overall Risk Ranking\n\n"
            f"{overall_table_md}\n\n"
            f"**Explanation:** {overall_explanation}\n\n"
        )
        ranking_section += """\n<div style="page-break-after: always;"></div>\n"""


        # Create visualisations of issues.
        pie_data = {cat['category']: cat['issue_count'] for cat in categories_with_counts}

        data_series = pd.Series(pie_data)

        pie_title = "Distribution of Issues by Category"

        fig = create_professional_pie(data_series, pie_title, figsize=(12, 8))

        img_base64 = save_plot_base64()
        plt.close(fig)

        # Create the Markdown image tag that embeds the pie chart.
        pie_chart_md = f"![Issues Distribution Pie Chart](data:image/png;base64,{img_base64})"

        # Check for additional issues
        logging.info("Checking for additional issues not covered in category analyses")

        # Get all bullet points, including those from minor categories
        # Compile issues from minor categories (bottom 20%)
        minor_bullet_points = "\n".join([
            "\n".join(category['issues'])
            for category in minor_categories
        ])

        system_prompt = """You are an expert analyst tasked with highlighting important issues that are underrepresented due to their low occurrence (bottom 20% of all identified issues). Your goal is to extract key issues from the minor categories and clearly describe their significance and potential impact on {company_name}. Do not reference or compare these issues to other parts of the report—simply focus on describing the missing issues from the minor categories."""

        chatbot = ChatGPT(
            system_prompt=system_prompt,
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2600,
        )

        question2 = f"""
Below is a list of issues from the minor categories (bottom 20% of all identified issues):
{minor_bullet_points}

A Comprehensive Analysis Summary: {comprehensive_analysis}

Please identify any issues from the minor category list that are not already covered in the comprehensive analysis summary. For each issue, provide:
- A clear title.
- A brief description explaining the issue and its potential impact on {company_name}.

Format your response exactly as follows:

## Additional Issues Identified (Minor Category – Bottom 20% of Issues)
- [Issue Title]: [Brief description of the issue and its potential impact].

Ensure that your output does not repeat or rephrase any content from the comprehensive analysis summary.
        """

        response2 = chatbot.ask(question2)

        if language.lower() != 'english':
            introduction = translate_content(introduction, 'auto', language)
            comprehensive_analysis = translate_content(comprehensive_analysis, 'auto', language)
            response2 = translate_content(response2, 'auto', language)
            conclusion = translate_content(conclusion, 'auto', language)

        # Generate final combined analysis
        final_analysis = (
f"{header}\n\n"
f"{toc}\n\n"
f"{pie_chart_md}\n\n\n"
f"{ranking_section}\n\n"
f"{introduction}\n\n\n"
f"{comprehensive_analysis}\n\n"
f"{response2}\n\n\n"
f"{conclusion}\n\n"
"## Analysis Methodology Note\n"
"This analysis was conducted in two phases:\n"
f"1. Detailed analysis of major issue categories (representing {cumulative_percentage_major:.1f}% of identified issues)\n"
"2. Comprehensive review of all identified issues to ensure complete coverage and identify cross-cutting concerns\n"
        )

        final_analysis = final_analysis.replace("```html","")
        final_analysis = final_analysis.replace("```markdown","")

        # Process citations into super script
        processor = CitationProcessor()
        final_analysis = processor.process_citations_in_markdown(final_analysis)

        # Save final analysis
        final_analysis_path = f"{general_folder}/Outputs/CompiledOutputs/ComprehensiveIssuesAnalysis_{company_name}.md"
        with open(final_analysis_path, "w", encoding='utf-8') as file:
            file.write(final_analysis)

    except Exception as e:
        logging.error(f"Error during issues analysis generation for {company_name}: {e}")
        raise

# Function to generate the topic summaries

def generate_topics_output(articles_sorted: List[Dict], company_name: str, language: str, general_folder: str, region: str,
                         industry_of_interest: str = None) -> str:
    """
    Generate topic-based summaries from the processed articles.
    Automatically checks if insights need to be generated first.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed and sorted articles
        company_name (str): Name of the company being analyzed
        language (str): Output language for the summaries
        general_folder (str): Base path for output files
        region (str): Geographic region of interest
        industry_of_interest (str): Industry relevant to the analysis
        
    Returns:
        str: Generated markdown content with topic summaries
    """
    try:
        logging.info(f"Starting topic summaries generation for {company_name}")

        # Validate required parameters
        if industry_of_interest is None or not industry_of_interest.strip():
            raise ValueError("industry_of_interest is required for generating topic summaries")
            
        # Check if we need insights for our summaries
        compiled_insights_path = f"{general_folder}/Outputs/CompiledOutputs/CompiledInsights_{company_name}.md"
        if not os.path.exists(compiled_insights_path):
            logging.info("Compiled insights not found, generating insights first")
            print("Compiled insights not found, generating insights first")
            generate_insights_output(
                articles_sorted=articles_sorted,
                company_name=company_name,
                general_folder=general_folder,
                industry_of_interest=industry_of_interest,
                region=region,
                language=language
            )
            print("Compiled insights generated")

        # Check if categories exist, if not extract them
        if not any('category' in article for article in articles_sorted):
            logging.info("Categories not found, extracting categories")
            articles_sorted = extract_categories(
                articles_sorted=articles_sorted, 
                company_name=company_name,
                industry_of_interest=industry_of_interest,
                region=region
            )
            # Save updated articles data
            save_data_to_json(articles_sorted, f"{general_folder}/Outputs/CompiledOutputs/ArticlesList.json")
            
        else:
            logging.info("Using existing categories")

        # Generate one-sentence descriptions
        logging.info("Generating article descriptions")
        system_prompt = """You are a helpful assistant. Your role is to describe in one single sentence what a given news media article says about a company. The final goal of this exercise is to be able to extract general themes and topics from the article. The one sentence you have to write should be focussed on a given company."""
        compiled_sentences = ""

        for article in articles_sorted:
            article_content = article.get('content', '')
            chatbot = ChatGPT(
                system_prompt=system_prompt,
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=350,
            )

            question = f"""
Please write a single sentence about the content of the news article with regards to {company_name}. Your output should only consist of that one sentence.
This one sentence should highlight the main topic or theme of the article from the perspective of {company_name}. We are interested about what is said on {company_name} in the article.

Here is the article: {article_content}
            """

            response = chatbot.ask(question)
            article['one_sentence_description'] = response
            compiled_sentences += response + "\n"

        # Define categories
        logging.info("Defining topic categories")
        system_prompt = """You are a helpful assistant. Your role is to define topic categories based on a series of one-sentence descriptions of news articles related to a company. The goal is to identify exclusive, non-overlapping topic categories based on the media coverage of the company."""
        chatbot = ChatGPT(
            system_prompt=system_prompt,
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1000,
        )

        question = f"""
You will be provided with a document named compiled_sentences. This document contains a series of one-sentence descriptions, each summarizing a news article related to {company_name}. Your task is to identify a maximum of 10 exclusive, non-overlapping topic categories based on the media coverage of the company. However, it is better and prefered if fewer categories are sufficient to cover the main aspects of the media coverage.

Follow these guidelines:

Topic Categories: Define categories that are neither too general nor too specific. Ensure the categories are mutually exclusive, meaning no two categories should cover the same subject matter.

Clarity: Each category should have a clear focus, reflecting distinct aspects of the media coverage related to {company_name}.

Output Format: List the categories in a bullet-point format with a brief description (1-2 sentences) explaining each category.

here is the compiled_sentences document: {compiled_sentences}

Be sure to focus on key themes present in the document and avoid redundant or overly broad topics. The fewer the number of categories, the better, as long as they are distinct and cover the main aspects of the media coverage.
Avoid defining categories that are too semantically similar or overlapping. For instance, "Financial Performance" and "Economic Growth" are too closely related to be separate categories. For example, Staffing Shortages, Labor Relations, Working Conditions and Recruitment Challenges should be grouped under a single category like "Human Resources Issues".
        """

        response = chatbot.ask(question)
        print(response)
        category_titles = re.findall(r'\*\*(.*?)\*\*', response)

        # Categorize articles

        logging.info("Categorizing articles")
        for article in articles_sorted:
            article_content = article.get('content', '')
            chatbot = ChatGPT(
                system_prompt=f"""You are a helpful assistant. Your role is to categorize a given news media article about {company_name} into one of the predefined categories. Your output should consist solely of the category name, chosen from the provided list of categories.""",
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=50,
            )

            question = f"""
Please categorize the following article about {company_name} into one of the predefined categories. 
Your output should only consist of the category name.
Here is the article content: {article_content}

Based on the content of the article, choose the most appropriate category from the following list: {category_titles}
Your output should solely be the name of the category chosen and nothing else.
            """

            article['category'] = chatbot.ask(question)

        # Filter and sort categories
        filtered_posts, kept_categories = filter_top_categories(articles_sorted)
        category_titles = [cat for cat in category_titles if cat in kept_categories]

        # Sort articles based on dates
        for article in articles_sorted:
            if 'date' in article:
                date_object = datetime.strptime(article['date'], '%B %d, %Y')
                article['timestamp'] = date_object.timestamp()

        articles_sorted = sorted(articles_sorted, key=lambda x: x['timestamp'])

        # Prepare articles by category
        category_articles = defaultdict(list)
        for article in filtered_posts:
            category = article.get('category')
            if category in kept_categories:
                category_articles[category].append(article)

        sorted_categories = sorted(category_articles.keys(), 
                                 key=lambda x: len(category_articles[x]), 
                                 reverse=True)

        # After categorizing articles and before summary generation
        logging.info("Validating insights availability")
        ensure_insights_exist(
            articles_sorted=filtered_posts,
            general_folder=general_folder,
            company_name=company_name,
            industry_of_interest=industry_of_interest,
            region=region,
            language=language
        )

        # Generate topic summaries
        print(f"Category Articles Keys: {list(category_articles.keys())}")
        for category, articles in category_articles.items():
            print(f"Category: {category}, Article Count: {len(articles)}")

        logging.info("Generating category summaries")

        # Initialize collections
        category_summaries = []
        article_sources_by_category = defaultdict(list)
        temp_summaries = ""

        # First add debug logging
        logging.info(f"Number of articles pre-processing: {len(filtered_posts)}")
        for category in sorted_categories:
            logging.info(f"Articles in {category}: {len(category_articles[category])}")

        # Initialize collections for summaries and sources
        category_summaries = []
        article_sources_by_category = {}
        
        # Process categories and gather all summaries first
        for idx, category in enumerate(sorted_categories, 1):
            category_number = idx  # This will be 1, 2, 3, etc.
            logging.info(f"Processing category {category_number}: {category}")
            articles = category_articles.get(category, [])
            
            if not articles:
                logging.warning(f"No articles found for category: {category}")
                continue
                
            compiled_insights = ""
            article_sources = []
            
            for article in articles:
                # Try to get insights content
                insights_content = None
                if insights_path := article.get('insights_path'):
                    try:
                        insights_content = read_insights_content(insights_path)
                    except Exception as e:
                        logging.error(f"Error reading insights for {article.get('title')}: {e}")

                if not insights_content:
                    logging.warning(f"No insights found for article: {article.get('title')}, regenerating...")
                    # Regenerate insights for this specific article
                    generate_insights_output(
                        articles_sorted=[article],
                        company_name=company_name,
                        general_folder=general_folder,
                        industry_of_interest=industry_of_interest,
                        region=region,
                        language=language
                    )
                    # Try reading again
                    if insights_path := article.get('insights_path'):
                        insights_content = read_insights_content(insights_path)

                if insights_content:
                    date_str = article.get('date', 'Unknown date')
                    compiled_insights += f"Article: {article['title']} (Date: {date_str})\n{insights_content}\n\n"

                    source = f"- [{article['title']}]({article.get('link', '#')}), {article.get('author_name', 'Unknown')}, {article.get('media_outlet', 'Unknown')}, {date_str}"
                    article_sources.append(source)
                    logging.info(f"Added content for article: {article['title']}")

            if compiled_insights:
                system_prompt = f"""You are a helpful assistant. Your role is to provide a detailed summary of media coverage for a specific category related to {company_name}. 
The compiled insights are structured in a chronological order. It is important to reflect the evolution of the media coverage along time in your summary.
Based on the compiled insights from multiple articles, create a comprehensive summary that captures the key points, trends, and developments within this category."""
                chatbot = ChatGPT(
                    system_prompt=system_prompt,
                    model_name="chatgpt-4o-latest",
                    temperature=0,
                    max_tokens=3500,
                )
                
                summary = chatbot.ask(f"""
Please provide a detailed summary of the media coverage for the category: {category}. Base your summary on the following compiled insights from multiple articles:

{compiled_insights}

Structure your output as follows:

# {category_number}.{category}

## {category_number}.1. Overview
Provide a brief overview of the general trends and key themes in the media coverage for this category.

## {category_number}.2. Chronological Analysis
### {category_number}.2.1 Early Coverage
Summarize the initial media coverage and key events.

### {category_number}.2.2 Developing Trends
Describe how the coverage evolved over time, highlighting significant shifts or new developments.

### {category_number}.2.3 Recent Developments
Focus on the most recent media coverage and current state of affairs.

## {category_number}.3. Stakeholder Perspectives
### {category_number}.3.1 [Stakeholder Group 1]
....
### {category_number}.3.X [Stakeholder Group X]
(Summarize perspectives from different stakeholders, preferably specific individuals or entities mentionned in the coverage. Provide insights into their positions, concerns, actions or opinions. Mention tangible names of stakeholders from the compiled insights you have received.)

## {category_number}.4. Implications and Future Outlook
Discuss the potential implications of the media coverage and provide insights into future trends or developments.

Additional Guidelines:
1. Ensure your summary comprehensively describes the overall media coverage for this category.
2. Reflect the evolution of the media coverage over time in your analysis.
3. At the end of each sentence, cite the source you fetched the information from. Use the format: [Media Outlet, Date].
4. Avoid citing the same sources twice.
5. Do not include a separate sources section at the end.
6. Take your time and visualize your output at each step of the reasoning process.
7. If a section is not applicable based on the available information, you may omit it, but maintain the overall structure.
8. Make your output as long as it is possible or needed to contain all the most relevant information and insights.
9. Maintain consistent section numbering starting with {category_number} throughout all headers.

Your structured summary should provide a clear and comprehensive analysis of the media coverage, making it easy for readers to navigate and understand the key points and developments in this category.
                """)
                
                if language.lower() != 'english':
                    summary = translate_content(summary, 'auto', language)
                
                # Store results
                category_summaries.append(summary)
                article_sources_by_category[category] = article_sources
                
                # Save individual category summary
                safe_category = re.sub(r'[^\w\-_\. ]', '_', category)
                summary_path = os.path.join(general_folder, "Outputs", "TopicsSummaries", f"Summary_{safe_category}.md")
                os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                
                summary_with_anchors = re.sub(
                    r'^(#+) (.+)$',
                    lambda m: f'<a name="{re.sub(r"\W+", "-", m.group(2).lower())}"></a>\n\n{m.group(0)}',
                    summary,
                    flags=re.MULTILINE
                )
                
                with open(summary_path, "w", encoding='utf-8') as file:
                    file.write(f"{summary_with_anchors}\n\n**Sources**:\n" + "\n".join(article_sources))

        # Add validation logging
        logging.info("Category summaries collection:")
        for category in sorted_categories:
            logging.info(f"- {category}: {len(article_sources_by_category.get(category, []))} articles")

        # Initialize document with title
        compiled_topics_summaries = f"# Media coverage - Topics Summaries - {company_name}\n\n"

        # First assemble all category summaries
        temp_summaries = ""
        for idx, (category, summary) in enumerate(zip(sorted_categories, category_summaries)):
            summary_with_anchors = re.sub(
                r'^(#+) (.+)$',
                lambda m: f'<a name="{re.sub(r"\W+", "-", m.group(2).lower())}"></a>\n\n{m.group(0)}',
                summary,
                flags=re.MULTILINE
            )

            sources_section = "**Sources**:\n" + "\n".join(article_sources_by_category[category])
            page_break = "\n<div style=\"page-break-after: always;\"></div>\n" if idx < len(sorted_categories) - 1 else "\n"

            temp_summaries += f"{summary_with_anchors}\n\n{sources_section}{page_break}"

        # Add validation logging
        logging.info(f"Generated summaries length: {len(temp_summaries)}")

        if not temp_summaries:
            logging.error("No category summaries were generated")
            raise ValueError("Failed to generate category summaries")

        # Generate TOC and introduction
        toc = generate_toc(temp_summaries, max_level=2)
        logging.info(f"TOC length: {len(toc)}")

        category_counts = {category: len(articles) for category, articles in category_articles.items()}

        data_series = pd.Series(category_counts)

        # Define the title and figure size for the pie chart
        pie_title = "Distribution of Articles by Topic Category"
        fig = create_professional_pie(data_series, pie_title, figsize=(8, 6))

        # Convert the plot to a base64 string using your helper function
        img_base64 = save_plot_base64()
        plt.close(fig)  # Close the figure to free up resources

        # Create the markdown image tag that embeds the pie chart
        pie_chart_md = f"![Topic Distribution Pie Chart](data:image/png;base64,{img_base64})"

        # Generate introduction using the assembled summaries
        logging.info("Generating executive introduction")
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2000
        )

        introduction_prompt = f"""
Based on the complete topic-based analysis of media coverage for {company_name}, create an executive introduction that:

1. Provides context about {company_name}'s media coverage.
2. Summarizes the scope of the analysis and the time period covered
3. Highlights the main topics identified and their relative importance
4. Explains how the analysis is structured and what readers can expect to learn

Analysis to summarize:
{temp_summaries}

Format your response as:
## Introduction
[Your introduction text here]

Keep the introduction to approximately 500 words and ensure it provides a clear roadmap for understanding the detailed topic analyses that follow. Structure it as plain text with paragraphs like in academic papers.
        """

        introduction = chatbot.ask(introduction_prompt)

        if language.lower() != 'english':
            introduction = translate_content(introduction, 'auto', language)

        if not temp_summaries:
            logging.error("No category summaries were generated")
            raise ValueError("Failed to generate category summaries")

        # Assemble final document in correct order
        compiled_topics_summaries = (
            f"# Media coverage - Topics Summaries - {company_name}\n\n"
            f"{toc}\n\n"
            f"{pie_chart_md}\n\n"
            f"{introduction}\n\n"
            f"{temp_summaries}"
        )

        # Add final validation
        logging.info(f"Final document length: {len(compiled_topics_summaries)}")
        if len(compiled_topics_summaries.strip()) < len(introduction.strip()):
            logging.error("Final document appears to be incomplete")
            raise ValueError("Final document missing content")

        # Process citations into super script
        processor = CitationProcessor()
        compiled_topics_summaries = processor.process_citations_in_markdown(compiled_topics_summaries)
        
        # Save complete topics summary
        topics_summaries_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", f"TopicsSummaries{company_name}.md")
        with open(topics_summaries_path, "w", encoding='utf-8') as file:
            file.write(compiled_topics_summaries)

        logging.info("Topic summaries generation completed successfully")
        return compiled_topics_summaries

    except Exception as e:
        logging.error(f"Error generating topic summaries: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def generate_analytics_output(articles_sorted: list, company_name: str, general_folder: str, 
                              industry_of_interest: str, region: str, language: str = 'English') -> str:
    """
    Generate media analytics report with sentiment analysis and visualizations.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed and sorted articles.
        company_name (str): Name of the company being analyzed.
        general_folder (str): Base path for output files.
        industry_of_interest (str): Industry relevant to the analysis.
        region (str): Geographic region of interest.
        language (str, optional): Output language for the analytics report. Defaults to 'English'.
        
    Returns:
        str: Generated markdown content with media analytics.
    """
    try:
        logging.info(f"Starting media analytics generation for {company_name}")
        
        # Check if categories exist, if not extract them
        if not any('category' in article for article in articles_sorted):
            logging.info("Categories not found, extracting categories")
            articles_sorted = extract_categories(
                articles_sorted=articles_sorted,
                company_name=company_name,
                industry_of_interest=industry_of_interest,
                region=region
            )
            # Save updated articles data
            save_data_to_json(articles_sorted, f"{general_folder}/Outputs/CompiledOutputs/ArticlesList.json")

        # Perform sentiment analysis only if not already defined
        logging.info("Performing sentiment analysis on articles")
        system_prompt = f"""You are a helpful assistant. Your role is to assess the overall tone of a news article, specifically focusing on the article's content about {company_name}."""
        
        for article in articles_sorted:
            # Only run tone analysis if not already defined
            if 'tone' not in article or not article['tone']:
                chatbot = ChatGPT(
                    system_prompt=system_prompt,
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=50,
                )
                question = f"""
Please assess the tone of the news article with specific regard to: {company_name}. Focus on how the article portrays {company_name} in terms of its actions, reputation, performance, and impact. The tone should be categorized as one of the following:

Positive: The article reflects well on {company_name}, highlighting favorable aspects.
Neutral: The article presents information about {company_name} in a balanced, objective manner without any strong positive or negative bias.
Negative: The article is critical of {company_name}, highlighting challenges, controversies, failures, or unfavorable developments.

Provide the final tone assessment as "Positive," "Neutral," or "Negative."

Here is the article content: {article['content']}

Your output should solely be one of these three words based on your assessment: "Positive", "Neutral", or "Negative". Nothing else
                """
                response = chatbot.ask(question)
                print(response)
                article['tone'] = response
            else:
                logging.info("Tone already defined for article, skipping tone analysis.")

            # Only run sentiment score analysis if not already defined
            if 'sentiment score' not in article or article['sentiment score'] is None:
                chatbot = ChatGPT(
                    system_prompt=system_prompt,
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=50,
                )
                question = f"""
Perform a detailed sentiment analysis of the article provided below, focusing exclusively on how it describes {company_name}. Your task is to evaluate the overall sentiment expressed in the article regarding {company_name} by carefully considering the tone, language, context, and any descriptive cues related to {company_name}.

Guidelines for Scoring:

-5: The article is extremely critical and conveys a highly negative sentiment toward {company_name}.
-4: The article offers notable criticism, highlighting significant negative aspects of {company_name}.
-3: The sentiment is moderately negative, with clear indications of disapproval of {company_name}.
-2: Some negative remarks about {company_name} are present, though they are not predominant.
-1: The article shows mild negativity or slight disapproval of {company_name}.
0: The portrayal of {company_name} is neutral with no significant positive or negative sentiment.
1: The article exhibits slight positive sentiment or mild approval of {company_name}.
2: The tone is moderately positive, suggesting a favorable view of {company_name}.
3: The article is largely favorable, displaying clear positive sentiment about {company_name}.
4: The content strongly praises {company_name}.
5: The article is exceptionally complimentary, demonstrating an extremely positive sentiment about {company_name}.

Instructions:
Analyze the language, tone, and context used in the article with respect to {company_name}.
Based solely on the observations, assign one sentiment score from the list: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].
Your final output must be exactly one of these values and nothing else.
Article Content: {article['content']}

Only output one of these values: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], and nothing else.
                """
                response = chatbot.ask(question)
                print(response)
                sentiment_score = extract_sentiment_score(response)
                article['sentiment score'] = sentiment_score if sentiment_score is not None else 'Invalid response'
            else:
                logging.info("Sentiment score already defined for article, skipping sentiment analysis.")

        # Save updated articles data
        save_data_to_json(articles_sorted, f"{general_folder}/Outputs/CompiledOutputs/ArticlesList.json")
        articles_sorted = load_data_from_json(f"{general_folder}/Outputs/CompiledOutputs/ArticlesList.json")

        # Create DataFrame for analysis
        df = pd.DataFrame(articles_sorted)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        
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

        # Generate each chart with error handling
        chart_generators = {
            'media_outlet_pie': (generate_media_outlet_pie_chart, [df]),
            'media_outlet_tone': (generate_media_outlet_tone_chart, [df]),
            'overall_sentiment': (generate_overall_sentiment_trend, [df, company_name]),
            'sentiment_by_category': (generate_sentiment_trends_by_category, [df]),
            'articles_by_category': (generate_articles_per_category, [df]),
            'category_tone': (generate_category_tone_chart, [df]),
            'top_journalists': (generate_top_journalists_chart, [df, company_name])
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
                stats = {
                    'total_articles': len(df),
                    'date_range': f"from {df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}",
                    'avg_sentiment': df['sentiment score'].mean(),
                    'median_sentiment': df['sentiment score'].median()
                }
            except Exception as e:
                logging.error(f"Error calculating statistics: {str(e)}")
                stats = {
                    'total_articles': len(df),
                    'date_range': "Error calculating date range",
                    'avg_sentiment': 0,
                    'median_sentiment': 0
                }

        try:
            media_outlet_stats = []
            for outlet in df['media_outlet'].unique():
                try:
                    outlet_df = df[df['media_outlet'] == outlet]
                    media_outlet_stats.append({
                        'outlet': outlet,
                        'articles': len(outlet_df),
                        'avg_sentiment': outlet_df['sentiment score'].mean(),
                        'median_sentiment': outlet_df['sentiment score'].median()
                    })
                except Exception as e:
                    logging.error(f"Error processing outlet {outlet}: {str(e)}")
                    continue
            
            media_outlet_stats.sort(key=lambda x: x['articles'], reverse=True)
        except Exception as e:
            logging.error(f"Error generating media outlet statistics: {str(e)}")
            media_outlet_stats = []

        # Generate final report with error handling for missing charts
        try:
            markdown_content = generate_markdown_report(
            company_name=company_name,
            total_articles=stats['total_articles'],
            date_range=stats['date_range'],
            avg_sentiment=stats['avg_sentiment'],
            median_sentiment=stats['median_sentiment'],
            media_outlet_pie_chart=charts.get('media_outlet_pie', ''),
            top_journalists_chart=charts.get('top_journalists', ''),
            media_outlet_tone_chart=charts.get('media_outlet_tone', ''),
            overall_sentiment_trend=charts.get('overall_sentiment', ''),
            media_outlet_stats=media_outlet_stats,
            articles_per_category=charts.get('articles_by_category', ''),
            category_tone_chart=charts.get('category_tone', ''),
            sentiment_trends_by_category=charts.get('sentiment_by_category', ''),
            df=df,
            general_folder=general_folder,
            language=language
        )
            
        except Exception as e:
            logging.error(f"Error generating markdown report: {str(e)}")
            markdown_content = f"Error generating media analytics report: {str(e)}"

        # Save the report
        try:
            media_analytics_path = f'{general_folder}/Outputs/CompiledOutputs/MediaAnalytics{company_name}.md'
            with open(media_analytics_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        except Exception as e:
            logging.error(f"Error saving media analytics report: {str(e)}")

        logging.info("Media analytics report generation completed")
        return markdown_content

    except Exception as e:
        logging.error(f"Error in generate_analytics_output: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error generating media analytics: {str(e)}"

def generate_stakeholder_quotes(articles_sorted: List[Dict], company_name: str, general_folder: str, language: str = 'English') -> str:
    """
    Generate stakeholder analysis from the processed articles.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed and sorted articles
        company_name (str): Name of the company being analyzed
        general_folder (str): Base path for output files
        language (str, optional): Output language for the stakeholder analysis. Defaults to 'English'.
        
    Returns:
        str: Generated markdown content with stakeholder analysis
    """
    try:
        logging.info(f"Starting stakeholder analysis generation for {company_name}")
        
        # Process stakeholder information
        stakeholder_quotes = process_stakeholder_info(company_name, articles_sorted)
        
        # Process and clean the markdown table
        stakeholder_quotes_processed = process_markdown_table(stakeholder_quotes)
        
        # Save the processed content
        output_folder = os.path.join(general_folder, "Outputs", "CompiledOutputs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save both markdown and CSV versions
        md_filename = os.path.join(output_folder, f"StakeholderQuotes_{company_name}.md")
        csv_filename = os.path.join(output_folder, f"StakeholderQuotes_{company_name}.csv")
        
        # Create the full markdown document
        md_document = f"""# Stakeholder Analysis Report - {company_name}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
This report contains stakeholder quotes and sentiments related to {company_name}.

## Data Table
{stakeholder_quotes_processed}

## Notes
- Quotes have been deduplicated and sorted alphabetically by stakeholder name
- Translations are provided where the original quote is not in English
- Sentiment analysis is based on the context and content of each quote
"""
        
        # Save markdown file
        with open(md_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(md_document)
            
        # Extract and save CSV data
        lines = stakeholder_quotes_processed.strip().split('\n')
        header_row = [col.strip() for col in lines[1].split('|')[1:-1]]
        
        data_rows = []
        for line in lines[3:]:
            if '|' in line:
                row = [cell.strip() for cell in line.split('|')[1:-1]]
                data_rows.append(row)
        
        with open(csv_filename, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header_row)
            writer.writerows(data_rows)
        
        logging.info("Stakeholder analysis generation completed successfully")
        return md_document

    except Exception as e:
        logging.error(f"Error in generate_stakeholder_quotes: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error generating stakeholder analysis: {str(e)}"

def generate_consolidated_stakeholder_analysis(company_name: str, articles: list, general_folder: str, language: str) -> str:
    """
    Generate a consolidated stakeholder analysis by combining quotes from the same stakeholder
    and analyzing their overall opinion and sentiment towards the company.
    
    Args:
        company_name (str): Name of the company being analyzed
        articles (list): List of processed articles
        general_folder (str): Base path for output files
        language (str): Target language for the analysis
        
    Returns:
        str: Generated markdown content with consolidated stakeholder analysis
    """
    try:
        # Check for existing stakeholder quotes file
        existing_quotes_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", f"StakeholderQuotes_{company_name}.md")
        
        if not os.path.exists(existing_quotes_path):
            logging.info("No existing stakeholder quotes file found, generating new quotes")
            # Generate the stakeholder quotes first
            md_document = generate_stakeholder_quotes(articles, company_name, general_folder)
            
            if md_document.startswith("Error"):
                raise ValueError(f"Failed to generate stakeholder quotes: {md_document}")
        else:
            logging.info(f"Found existing stakeholder quotes file: {existing_quotes_path}")
            with open(existing_quotes_path, 'r', encoding='utf-8') as f:
                md_document = f.read()
                print(md_document)

        # Extract table section
        table_start = md_document.find('| Stakeholder Name |')
        if table_start == -1:
            raise ValueError("Could not find stakeholder table in document")
            
        table_end = md_document.find('##', table_start)
        if table_end == -1:
            table_end = len(md_document)
            
        raw_stakeholder_data = md_document[table_start:table_end].strip()
        
        # Parse stakeholders
        stakeholders = parse_stakeholder_table(raw_stakeholder_data)

        # Generate consolidated analysis
        consolidated_md = f"""# Consolidated Stakeholder Analysis - {company_name}

## Overview
This analysis consolidates stakeholder opinions and sentiments, providing a comprehensive view of key perspectives on {company_name}. Each stakeholder's quotes have been analyzed in context to understand their overall position and potential impact on the company.

## Stakeholder Perspectives

| Stakeholder Name | Role/Position | Opinion and Sentiment Analysis |
|-----------------|---------------|--------------------------------|
"""
        
        # Create list of stakeholders with their quote counts for sorting
        stakeholder_data = []
        
        # Process each stakeholder
        for stakeholder_name, data in stakeholders.items():
            # Skip if no meaningful data
            if len(data['quotes']) == 0:
                continue
                
            # Prepare context for analysis by joining quotes and context
            context = "\n".join([
                f"Quote: {quote}\nContext: {ctx}"
                for quote, ctx in zip(data['quotes'], data['context'])
            ])
            
            # Analyze stakeholder role
            role_chatbot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=50
            )
            
            role_prompt = f"""
Based on the following context, determine a clear, concise role or position for the stakeholder {stakeholder_name} in relation to {company_name}.

Previously identified role(s): {', '.join(data['roles'])}

Context:
{context}

Requirements:
1. Provide a single role or position in 1-5 words maximum
2. Be specific and relevant to {company_name} or its industry
3. Focus on the stakeholder's professional capacity or expertise
4. Use concise, clear terminology
5. Include the entity or organisation the stakeholder is associated with, if the information is available.

Example good responses:
- "Chief Technology Officer of [Institution NAME]"
- "Industry Analyst at [Entity NAME]"
- "Solar Energy Expert"
- "Investment Director for [Entity NAME]"
- "Consumer"

Example bad responses:
- "A highly experienced professional in the field" (too long)
- "Person involved in operations" (too vague)
- "Stakeholder" (too generic)

Output only the role/position, nothing else.
"""
            
            role_analysis = role_chatbot.ask(role_prompt).strip()
            logging.debug(f"Determined role for {stakeholder_name}: {role_analysis}")
            
            # Initialize analysis chatbot for opinion/sentiment
            opinion_chatbot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1500
            )
            
            # Calculate number of sentences based on quote count
            sentence_count = len(data['quotes'])
            
            analysis_prompt = f"""
Analyze the following stakeholder's opinions and sentiment towards {company_name}. The stakeholder is {stakeholder_name}, who has been identified as: {role_analysis}.

Context and Quotes:
{context}

Provide a {sentence_count} sentence analysis that covers:
1. The stakeholder's overall position towards {company_name}
2. The consistency or evolution of their views
3. The potential impact of their opinions on {company_name}'s reputation or operations
4. Any notable patterns in their sentiment ({', '.join(data['sentiments'])})

Important: Your response must be exactly {sentence_count} sentences - no more, no less.
Format your response as a single paragraph without any headers or bullet points.
"""
            
            analysis = opinion_chatbot.ask(analysis_prompt)

            # Translate analysis if language is not English
            if language.lower() != 'english':
                analysis = translate_content(analysis, 'auto', language)
                logging.info(f"Successfully translated stakeholder analysis for {stakeholder_name}")
            
            # Store stakeholder data with quote count for later sorting
            stakeholder_data.append({
                'name': stakeholder_name,
                'quote_count': len(data['quotes']),
                'role': role_analysis,
                'analysis': analysis
            })
        
        # Sort stakeholders by quote count (descending)
        sorted_stakeholders = sorted(stakeholder_data, key=lambda x: x['quote_count'], reverse=True)
        
        # Add rows to markdown table in sorted order with quote counts
        for stakeholder in sorted_stakeholders:
            stakeholder_with_count = f"{stakeholder['name']} [{stakeholder['quote_count']} quotes]"
            consolidated_md += f"| {stakeholder_with_count} | {stakeholder['role']} | {stakeholder['analysis']} |\n"
        
        # Save the consolidated analysis
        output_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", f"ConsolidatedStakeholderAnalysis_{company_name}.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(consolidated_md)
        
        return consolidated_md

    except Exception as e:
        logging.error(f"Error in generate_consolidated_stakeholder_analysis: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error generating consolidated stakeholder analysis: {str(e)}"

# Journalist functions
   
def generate_journalist_article_list(articles_sorted: List[Dict], journalist_name: str, general_folder: str, language: str = "English") -> str:
    """
    Generate a markdown list of articles written by a specific journalist.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed and sorted articles
        journalist_name (str): Name of the journalist being analyzed
        general_folder (str): Base path for output files
        language (str): Output language for the list
        
    Returns:
        str: Generated markdown content
    """
    try:
        logging.info(f"Starting article list generation for {journalist_name}")
        
        # Generate markdown content
        logging.info("Generating markdown content")
        md_content = f"# List of media articles redacted by {journalist_name}\n\n"
        md_content += "## Overview\n\n"
        
        # Add summary statistics
        total_articles = len(articles_sorted)
        unique_outlets = len(set(article.get('media_outlet', 'Unknown') for article in articles_sorted))
        
        md_content += f"- Total Articles: {total_articles}\n"
        md_content += f"- Media Outlets: {unique_outlets}\n\n"
        
        md_content += "## Complete list of articles\n\n"
        
        # Sort articles by date (most recent first) while maintaining the original sorting as secondary criterion
        articles_list = sorted(
            articles_sorted,
            key=lambda x: (
                datetime.strptime(x.get('date', 'January 1, 2024'), '%B %d, %Y').timestamp(),
                x.get('reordered_position', float('inf'))
            ),
            reverse=True
        )

        # Generate the detailed list
        for article in articles_list:
            media_outlet = article.get('media_outlet', 'Unknown')
            date = article.get('date', 'Unknown')
            title = article.get('title', 'Unknown')
            link = article.get('link', None)
            
            media_outlet_hyperlinked = f"[{media_outlet}]({link})" if link else media_outlet
            md_content += f"- {media_outlet_hyperlinked} ({date}): *{title}*\n"

        # Add media outlet statistics section
        md_content += "\n## Media Outlet Statistics\n\n"
        outlet_counts = collections.Counter(
            article.get('media_outlet', 'Unknown') 
            for article in articles_sorted
        )
        
        if outlet_counts:
            md_content += "### Coverage by Media Outlet\n\n"
            for outlet, count in outlet_counts.most_common():
                md_content += f"- {outlet}: {count} articles\n"

        # Save markdown file
        output_folder = os.path.join(general_folder, "Outputs", "CompiledOutputs")
        output_file_path = os.path.join(output_folder, f"Articles_list_{journalist_name.replace(' ', '_')}.md")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file_path, 'w', encoding='utf-8') as md_file:
                md_file.write(md_content)
            logging.info(f"Article list saved successfully to {output_file_path}")
        except Exception as e:
            logging.error(f"Error saving article list: {str(e)}")
            raise

        return md_content

    except Exception as e:
        logging.error(f"Error generating article list: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def generate_journalist_profile(articles_sorted: List[Dict], journalist_name: str, news_folder_path: str, 
                              language: str = 'English', force_reprocess: bool = False) -> str:
    """
    Generate comprehensive profile analysis of a specific journalist based on their articles.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed articles
        journalist_name (str): Name of the journalist being analyzed
        news_folder_path (str): Path to the folder containing news articles
        language (str): Output language for the analysis
        force_reprocess (bool): If True, reprocess everything even if saved data exists
        
    Returns:
        str: Generated markdown content with journalist analysis
    """
    try:
        logging.info(f"Starting journalist profile analysis for {journalist_name}")
        
        # Set up directory structure
        general_folder = setup_journalist_directories(news_folder_path, journalist_name)
        processed_articles_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", "ProcessedArticles.json")
        categorized_articles_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", f"CategorizedArticles_{journalist_name}.json")
        
        # Initialize compiled_sentences
        compiled_sentences = ""
        
        if not force_reprocess and os.path.exists(categorized_articles_path):
            logging.info("Loading previously processed articles")
            data = load_data_from_json(categorized_articles_path)
            articles_sorted = data.get('articles', [])
            compiled_sentences = "\n".join([
                article.get('one_sentence_description', '') 
                for article in articles_sorted 
                if 'one_sentence_description' in article
            ])

        else:
            # Generate one-sentence descriptions
            logging.info("Generating article descriptions")
            compiled_sentences = ""
            system_prompt = """You are a helpful assistant. Your role is to describe in one single sentence what a given news media article's main topic and angle is."""

            for article in articles_sorted:
                article_content = article.get('content', '')
                chatbot = ChatGPT(
                    system_prompt=system_prompt,
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=350,
                )

                question = f"""
Please write a single sentence summarizing this article's main topic and {journalist_name}'s angle or approach.
Focus on both the subject matter and how {journalist_name} covers it.

Article: {article_content}
                """

                response = chatbot.ask(question)
                print(response)
                article['one_sentence_description'] = response
                compiled_sentences += response + "\n"

            # Save processed articles
            save_data_to_json({"categories": [], "articles": articles_sorted}, categorized_articles_path)

        # Check if categories need to be generated
        need_categorization = True
        if not force_reprocess and os.path.exists(categorized_articles_path):
            logging.info("Loading previously categorized articles")
            saved_data = load_data_from_json(categorized_articles_path)
            
            # Check if ALL articles have categories
            all_categorized = all('category' in article for article in articles_sorted)
            
            if all_categorized:
                need_categorization = False
                # Reconstruct categories_data from saved data
                categories_data = []
                for category_info in saved_data['categories']:
                    category = {
                        'category': category_info['category'],
                        'description': category_info['description'],
                        'articles': []
                    }
                    # Find articles belonging to this category
                    for article in articles_sorted:
                        if article.get('category') == category_info['category']:
                            category['articles'].append(article)
                    categories_data.append(category)
            else:
                logging.info("Some articles missing categories - will recategorize")
                
        if need_categorization:
            logging.info("Defining coverage categories")
            system_prompt = f"""You are a media analyst. Your role is to identify the main recurring stories, narratives, and series of connected events that {journalist_name} covers across multiple articles."""
            
            chatbot = ChatGPT(
                system_prompt=system_prompt,
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1000,
            )

            question = f"""
Based on these article summaries, identify the main recurring stories or narrative threads in {journalist_name}'s articles.
Create 5-7 distinct categories that represent specific stories, events, or series of connected events that appear across multiple articles.

Article summaries:
{compiled_sentences}

For each category:
1. Give it a name that describes the specific story/narrative (e.g., "Tech Company X Layoff Series" rather than just "Tech Industry")
2. Explain what specific events, developments, or connected stories this narrative encompasses
3. Focus on identifying stories that span multiple articles or connected events rather than broad topics

Format as:
CATEGORY: [Story/Narrative Name]
DESCRIPTION: [Explanation of the specific story thread and how it develops across articles]
"""

            categories_response = chatbot.ask(question)
            print(categories_response)

            categories_data = []
            current_category = None
            current_description = None
        
            for line in categories_response.split('\n'):
                # Remove markdown heading markers and whitespace
                clean_line = line.lstrip('#').strip()
                if clean_line.startswith('CATEGORY:'):
                    if current_category is not None:
                        categories_data.append({
                            'category': current_category.strip(),
                            'description': current_description.strip() if current_description else '',
                            'articles': []
                        })
                    current_category = clean_line.replace('CATEGORY:', '').strip()
                    current_description = None
                elif clean_line.startswith('DESCRIPTION:'):
                    current_description = clean_line.replace('DESCRIPTION:', '').strip()


            # Add the last category
            if current_category is not None:
                categories_data.append({
                    'category': current_category,
                    'description': current_description if current_description else '',
                    'articles': []
                })

            # Add "Other" category automatically
            categories_data.append({
                'category': 'Other',
                'description': 'Articles that do not clearly align with specific narrative threads or recurring stories',
                'articles': []
            })

            # Check if articles already have categories
            all_categorized = all('category' in article for article in articles_sorted)
            print(f"Are all articles categorized? {all_categorized}")

            if not all_categorized:
                # Categorize articles
                logging.info("Categorizing articles")
                for article in articles_sorted:
                    if 'category' not in article:  # Only categorize if not already done
                        classification_prompt = ""
                        for category in categories_data:
                            classification_prompt += f"\n{category['category']}: {category['description']}"
                        
                        # Use chatbot to classify the article
                        chatbot = ChatGPT(
                            model_name="gpt-4o-mini",
                            temperature=0,
                            max_tokens=200
                        )
                        
                        classification = chatbot.ask(
                            f"""
Given these coverage categories for the journalist: {journalist_name}:
{classification_prompt}

Classify this article into one of these categories. Choose the most appropriate category based on the descriptions provided.

Article content:
{article['content']}

Only output the exact category name that best matches this article.
                            """
                        )
                        print(classification)
                        
                        # Add debugging
                        print(f"Trying to classify article: {article.get('title', 'Unknown')}")
                        print(f"Classification received: {classification}")
                        print(f"Available categories: {[cat['category'] for cat in categories_data]}")
                        
                        # Add article to appropriate category and save category in article
                        for category in categories_data:
                            cleaned_category = category['category'].replace('**', '').strip()
                            cleaned_classification = classification.replace('**', '').strip()
                            if cleaned_category == cleaned_classification:
                                category['articles'].append(article)
                                article['category'] = cleaned_category  # Save category in article
                                break
                
                # Save the updated articles with their categories
                print("Saving categorized articles...")
                save_data_to_json(articles_sorted, processed_articles_path)
        pass

        # Save in a format that avoids circular references
        save_data = {
            'categories': [
                {
                    'category': cat['category'].replace('**', '').strip(),  # Remove asterisks
                    'description': cat['description']
                }
                for cat in categories_data
            ],
            'articles': []
        }

        # For debugging
        processed_articles = set()  # Keep track of processed articles

        # Add articles with safer category assignment
        for article in articles_sorted:
            # Find the category for this article
            article_category = None
            for cat in categories_data:
                cat_name = cat['category'].replace('**', '').strip()  # Remove asterisks

                # Check if article is in category by title
                article_titles_in_category = {a.get('title', '') for a in cat['articles']}
                if article.get('title', '') in article_titles_in_category:
                    article_category = cat_name
                    processed_articles.add(article.get('title', ''))
                    break
                
            # Create article data with category
            article_data = article.copy()
            if article_category:
                article_data['category'] = article_category
            else:
                if article.get('title', '') not in processed_articles:  # Only warn about truly uncategorized articles
                    print(f"\nWARNING: No category found for article: {article.get('title', 'Unknown title')}")
                article_data['category'] = 'Uncategorized'

            save_data['articles'].append(article_data)

        print(f"\nSuccessfully categorized {len(processed_articles)} articles")
        save_data_to_json(save_data, categorized_articles_path)
        
        # Generate profile analysis
        logging.info("Generating profile analysis")
        profile_md = f"""
# Journalist Profile Analysis: {journalist_name}

## Main Stories / Topics in {journalist_name}'s Media Coverage
"""
        # Sort categories by number of articles, keeping 'Other' last
        categories_data = sorted(
            categories_data,
            key=lambda x: (
                1 if x['category'] == 'Other' else 0,  # Force 'Other' to end
                -len(x['articles'])  # Sort rest by number of articles (descending)
            )
        )
        
        # Analyze each category
        # Inside the generate_journalist_profile function, replace the category analysis section:

        for category in categories_data:
            # Skip empty categories
            if not category['articles']:
                continue
            
            # Sort articles by date
            category['articles'].sort(key=lambda x: datetime.strptime(x.get('date', '2024-01-01'), '%B %d, %Y'))
            
            excluded_categories = [
            cat_data['category']
            for cat_data in categories_data
            if cat_data['category'] != category['category']
        ]

            # Create a comma-separated string of those categories:
            excluded_categories_str = ", ".join(excluded_categories)

            # First chatbot: Factual narrative analysis
            narrative_bot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=2500
            )
            
            narrative_prompt = f"""
Analyze the factual coverage and key narratives in {journalist_name}'s articles for the category: {category['category']}.

Focus on events and stories that *directly* relate to {category['category']}. 
YOU MUST EXCLUDE all information which is closely related to {excluded_categories_str}.

Focus ONLY on extracting and organizing the key events, facts, and stories covered in these articles that are relevant to the '{category['category']}' category. 
Exclude any analysis of the journalist's opinions or stance.

Articles to analyze:
{json.dumps([
    {
        'title': article.get('title', 'Untitled'),
        'date': article.get('date', 'Unknown'),
        'media_outlet': article.get('media_outlet', 'Unknown'),
        'content': article.get('content', '')
    } 
    for article in category['articles']
], indent=2)}

Provide a comprehensive analysis of the key stories and facts. Do not use bullet points.
Format your output into clear sections with header (level '###' maximum). Use a minimum of headers to avoid an overload of sections. ALWAYS cite the specific article [Media outlet, Date], in between squared brackets[], supporting each point.

For each point:

- Start with a clear event or development
- Include specific dates and chronological progression where relevant
- Name key stakeholders (companies, individuals, organizations)
- Include important statistics, quotes, or concrete outcomes
- ALWAYS cite the specific article [media outlet, date], in between squared brackets[], supporting each point.
- Highlight connections between different events where they exist.

Focus on telling the story through factual, chronological points that are well-supported by the articles.

Every fact must be linked to specific articles [media outlet, date].
Focus on objective information only—no interpretation of the journalist's views. The main title of your output should be at the "##" level. you should only have one main title which should reflect the facts and story discussed, then a minimum of subtitles for the distinct sections. format your output with clear paragraphs and focus on having a clear storyline, no bullet points.
DO NOT address this prompt directly.
            """
            
            narrative_analysis = narrative_bot.ask(narrative_prompt)
            print("Completed narrative analysis:")
            print(narrative_analysis)

            if language.lower() != 'english':
                narrative_analysis = translate_content(narrative_analysis, 'auto', language)
            
            # Second chatbot: Perspective and stance analysis
            perspective_bot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=2500
            )
            
            stance_prompt = f"""
Analyze {journalist_name}'s perspective and stance based on their coverage in the category: {category['category']}

Use both the original articles AND the factual narrative analysis below as your source material.

Original articles:
{json.dumps([
    {
        'title': article.get('title', 'Untitled'),
        'date': article.get('date', 'Unknown'),
        'media_outlet': article.get('media_outlet', 'Unknown'),
        'content': article.get('content', '')
    } for article in category['articles']
], indent=2)}

Factual narrative analysis:
{narrative_analysis}

Provide a detailed analysis of {journalist_name}'s position, sentiment or stance on the topics described in the narrative analysis. Your output should be connected to the narrative analysis. It should respond or build upon the coverage description, with tangible example to illustrate {journalist_name}'s opinion or stance on the sepcific topic.
Analyze {journalist_name}'s perspective and opinions in bullet points. For each observation:

- Describe how {journalist_name} positions themselves on specific issues, Show how {journalist_name} frames events and developments, specifically the ones described in the narrative analysis.
- Note any apparent biases or preferences in their coverage. specifically if this bias or preference is in favour of a certain individual or organisation.
- Highlight his treatment of different stakeholders, how does he consider them and what is his sentiment towards them.
- Support each point with specific quotes and article references [media outlet, date].
- Adopt a similar structure as the Factual narrative analysis. Do not use bullet points. Structure your output with clear paragraphs. You main header should be at the '##' level and other subheaders at the '###' levels.

Focus on building a clear picture of the journalist's viewpoint through concrete examples and evidence. It should give an idea of what matters or not to {journalist_name}, what does he stand for through his coverage.

Every observation must be supported by specific quotes or examples from the articles. Cite your sources in between squared brackets[].
Focus on identifying patterns across multiple articles pieces rather than single instances.
Highlight any evolution in the journalist's perspective over time, if applicable.
DO NOT address this prompt directly.
"""
        
            stance_analysis = perspective_bot.ask(stance_prompt)
            print("Completed stance analysis:")
            print(stance_analysis)

            if language.lower() != 'english':
                stance_analysis = translate_content(stance_analysis, 'auto', language)
            
            # Add to markdown document
            profile_md += f"""      
{f"## Miscellaneous Coverage" if category['category'] == 'Other' else ''}{'' if category['category'] == 'Other' else ''}
{narrative_analysis}

{stance_analysis}
        
#### Articles covering this topic:
"""
            
            # Add article list
            for article in category['articles']:
                profile_md += f"- [{article.get('title', 'Untitled')}]({article.get('link', '#')}) - {article.get('media_outlet', 'Unknown')}, {article.get('date', 'Unknown Date')}\n"
            
            profile_md += "\n---\n"

            print(profile_md)

        # Generate TOC with headings up to level 2
        toc = generate_toc(profile_md, max_level=2)
        
        # Generate introduction (or use fallback)
        try:
            introduction = generate_introduction(profile_md, journalist_name)
            if language.lower() != 'english':
                introduction = translate_content(introduction, 'auto', language)
        except Exception as e:
            introduction = f"## Introduction\nThis analysis examines the work and coverage patterns of {journalist_name} based on {len(articles_sorted)} articles."
        
        # Prepend TOC and introduction to the markdown document
        profile_md = f"{toc}\n\n{introduction}\n\n" + profile_md
        
        # Generate conclusion
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1500
        )
        
        conclusion_prompt = f"""
Based on the complete analysis of {journalist_name}'s coverage topics, create a concluding section.

Topical coverage:
{profile_md}

Keep the conclusion to approximately 300 words.
"""
        
        conclusion = chatbot.ask(conclusion_prompt)
        print(conclusion)

        if language.lower() != 'english':
            conclusion = translate_content(conclusion, 'auto', language)
            
        profile_md += conclusion

        processor = CitationProcessor()
        profile_md = processor.process_citations_in_markdown(profile_md)
        
        # Save the analysis
        output_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                 f"JournalistProfile_{journalist_name.replace(' ', '_')}.md")
        
        # Make sure the directory exists before writing
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(profile_md)

        return profile_md

    except Exception as e:
        logging.error(f"Error in generate_journalist_profile: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def analyze_journalist_topic_coverage(articles_sorted: List[Dict], journalist_name: str, 
                                    topic_focus: str, general_folder: str, 
                                    language: str = 'English') -> str:
    """
    Generate analysis of a journalist's coverage of a specific topic.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed articles
        journalist_name (str): Name of the journalist being analyzed
        topic_focus (str): Topic to analyze in the journalist's coverage
        general_folder (str): Base path for output files
        language (str): Output language for the analysis
        
    Returns:
        str: Generated markdown content with topic analysis
    """
    try:
        logging.info(f"Starting topic analysis for {journalist_name} on {topic_focus}")
        
        # Filter articles for topic relevance
        topic_relevant_articles = []
        system_prompt = f"""You are a helpful assistant. Your role is to determine whether an article substantively discusses the topic: {topic_focus}."""
        
        for article in articles_sorted:
            chatbot = ChatGPT(
                system_prompt=system_prompt,
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=100
            )
            
            question = f"""
Assess whether the provided article relates to the topic of focus or not. The topic of interest is: {topic_focus}
You should assess it to be related if {topic_focus} is discussed even though not central to the article.
You should determine the topic to be not relevant if {topic_focus} or synonyms or similar terms are not discussed at all in the article.

Article content:
{article.get('content', '')}

Respond only with "Yes" if the article discusses this topic, or "No" if it doesn't. Your output should only be "Yes" or "No" and nothing else.
"""
            
            response = chatbot.ask(question)
            logging.debug(f"Article relevance response: {response}")
            
            if response.strip().lower() == "yes":
                topic_relevant_articles.append(article)
        
        if not topic_relevant_articles:
            return f"No articles found discussing {topic_focus} in {journalist_name}'s coverage."
        
        # Compile content for analysis
        compiled_content = ""
        for article in topic_relevant_articles:
            metadata = f"""
Title: {article.get('title', 'Untitled')}
Media outlet: {article.get('media_outlet', 'Unknown')}
Date: {article.get('date', 'Unknown')}
---
"""
            compiled_content += metadata + article.get('content', '') + "\n\n"
        
        # Save compiled content for BigSummarizerGPT
        temp_compiled_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                        f"List_{journalist_name}_{topic_focus}.md")
        os.makedirs(os.path.dirname(temp_compiled_path), exist_ok=True)
        
        with open(temp_compiled_path, "w", encoding='utf-8') as f:
            f.write(compiled_content)

        # Generate bullet point analysis
        chatbot = BigSummarizerGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2500
        )
        
        bullet_points_prompt = f"""
Analyze the compiled articles to create a comprehensive bullet point list of {journalist_name}'s coverage of {topic_focus}.

For each point:
1. Describe what the journalist says about {topic_focus}
2. Include specific examples, quotes, or evidence
3. Reference the source (media outlet and date) for each point
4. Maintain chronological order where relevant

Format each bullet point to start with a dash (-) and include the source reference in brackets at the end. Your output should solely consist of these bullet points. Be detailed in your reporting.
"""
        
        bullet_points = chatbot.ask(bullet_points_prompt, temp_compiled_path)
        print("Generated bullet points:")
        print(bullet_points)
        
        # Generate structured analysis
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2500
        )
        
        analysis_prompt = f"""
Based on the following bullet points summarizing {journalist_name}'s coverage of {topic_focus}, create a structured analysis.

Bullet points to analyze:
{bullet_points}

Provide a comprehensive analysis to describe {journalist_name}'s coverage on {topic_focus}. Start your output with: "## Analysis of {journalist_name}'s {topic_focus} Coverage"

Format the response with clear section headers and maintain proper markdown formatting. Your output must be comprehensive and descriptive about the coverage. It should include a maximum of [references].
Include relevant source citations throughout the analysis. Reference the source [media outlet and date] for each point, use squared brackets for the references.
"""
        
        structured_analysis = chatbot.ask(analysis_prompt)
        print(structured_analysis)
        
        # Generate sentiment analysis
        chatbot = BigSummarizerGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2500
        )
        sentiment_prompt = f"""
Analyze the compiled articles to extract a bullet point list interpreting {journalist_name}'s opinion, stance, and attitude regarding {topic_focus}.

For each point:
1. Identify any stance, opinion, or sentiment expressed by {journalist_name} about {topic_focus}
2. Provide direct quotes or tangible examples to support the interpretation
3. Reference the source [media outlet and date] for each point, use squared brackets for the references.
4. Maintain chronological order where relevant

Format each bullet point to start with a dash (-) and include the source reference [in between] squared brackets at the end. Be detailed in your reporting.
"""
        
        sentiment_bullet_points = chatbot.ask(sentiment_prompt, temp_compiled_path)
        print(sentiment_bullet_points)
        
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2500
        )

        sentiment_analysis_prompt = f"""
Based on the following bullet points interpreting {journalist_name}'s opinion, stance, and attitude regarding {topic_focus}, create a structured analysis.

Coverage Analysis:
{structured_analysis}

Sentiment Bullet Points:
{sentiment_bullet_points}

Provide a comprehensive analysis describing {journalist_name}'s stance on {topic_focus}. Start your output with: "## Analysis of {journalist_name}'s Stance on {topic_focus}"

Format the response with clear section headers and proper markdown formatting. Your output must be detailed, highlighting patterns in sentiment, specific arguments, and potential biases. Ensure it smoothly connects with the previous coverage analysis.
Include relevant [source citations] throughout the analysis, relating the sentences you are writing with the sources where you got your information from. Cite your source in between squared brackets []. Use the following format for citations: [Media Outlet, date]. Cite your references in the text and not at the end.
"""
        
        sentiment_structured_analysis = chatbot.ask(sentiment_analysis_prompt)
        print(sentiment_structured_analysis)
        
        # Assemble final output
        final_analysis = f"""# Topic Analysis: {journalist_name}'s Coverage of {topic_focus}

## Introduction

This report provides an in-depth analysis of {journalist_name}'s coverage of {topic_focus}. It is structured into two main sections:

- Coverage Analysis: This section examines how {journalist_name} has reported on {topic_focus}, highlighting key themes, arguments, and evidence from their articles. It provides a detailed synthesis of their reporting, including sources and references.
- Sentiment Analysis: The second section delves deeper into {journalist_name}'s stance and sentiment regarding {topic_focus}. By analyzing tone, language, and recurring viewpoints, it assesses potential biases, underlying opinions, and the broader narrative conveyed in their reporting.

This document aims to offer a clear and structured perspective on how {journalist_name} has shaped public understanding of {topic_focus}, drawing from a selection of their published work.

---

{structured_analysis}

---
<div style="page-break-after: always;"></div>
---

{sentiment_structured_analysis}

---

## Supporting Articles
The following articles were analyzed for this report:
"""
        
        for article in topic_relevant_articles:
            media_outlet = article.get('media_outlet', 'Unknown')
            date = article.get('date', 'Unknown')
            title = article.get('title', 'Unknown')
            link = article.get('link', None)
            
            media_outlet_hyperlinked = f"[{media_outlet}]({link})" if link else media_outlet
            final_analysis += f"- **{title}**, {media_outlet_hyperlinked} ({date})\n"
        
        # Translate if needed
        if language.lower() != 'english':
            final_analysis = translate_content(final_analysis, 'auto', language)

        # Process citations
        processor = CitationProcessor()
        final_analysis = processor.process_citations_in_markdown(final_analysis)

        # Save the analysis
        output_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                 f"TopicAnalysis_{journalist_name.replace(' ', '_')}_{topic_focus.replace(' ', '_')}.md")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_analysis)
        
        return final_analysis
        
    except Exception as e:
        logging.error(f"Error in analyze_journalist_topic_coverage: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def generate_journalist_analysis_output(articles_sorted: List[Dict], journalist_name: str, general_folder: str, 
                           language: str) -> str:
    """
    Generate comprehensive journalist analysis with visualizations and AI-powered insights.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed and sorted articles
        journalist_name (str): Name of the journalist being analyzed
        general_folder (str): Base path for output files
        language (str): Output language for the analysis
        
    Returns:
        str: Generated markdown content with journalist analytics
    """
    
    distribution_header = "Distribution of Coverage"
    distribution_intro = f"""The following two sections will illustrate the distribution of {journalist_name}'s coverage across various media outlets and general topics. This will be followed by a brief description summarizing the nature of their coverage."""

    media_outlet_header = "Media Outlet Distribution"
    media_outlet_desc = f"This chart shows how {journalist_name}'s articles are distributed across different media outlets:"

    category_dist_header = "Coverage Category Distribution"
    category_dist_desc = "This chart shows the distribution of articles across different main categories:"

    category_by_media_header = "Category Distribution by Media Outlet"
    category_by_media_desc = f"The following visualization shows how different categories are distributed across the top media outlets where {journalist_name} publishes most frequently:"

    narrative_header = "Distribution of Narrative Coverage"
    narrative_desc = f"The following chart shows how {journalist_name}'s articles are distributed across different narrative threads and storylines:"

    narrative_explanation = f"""The chart above illustrates the primary stories and recurring themes in {journalist_name}'s coverage. 
    Articles are grouped by narrative threads representing connected stories and developments over time."""

    org_analysis_header = "Analysis of Most Discussed Organizations"
    org_chart_desc = "The following chart shows the top 10 most frequently mentioned organizations in the coverage, with their tone distribution:"

    people_analysis_header = "Analysis of Most Discussed People"
    people_chart_desc = "The following chart shows the top 10 most frequently mentioned people in the coverage, with their tone distribution:"

    # Translate strings if needed
    if language.lower() != 'english':
        distribution_header = translate_content(distribution_header, 'auto', language)
        distribution_intro = translate_content(distribution_intro, 'auto', language)
        media_outlet_header = translate_content(media_outlet_header, 'auto', language)
        media_outlet_desc = translate_content(media_outlet_desc, 'auto', language)
        category_dist_header = translate_content(category_dist_header, 'auto', language)
        category_dist_desc = translate_content(category_dist_desc, 'auto', language)
        category_by_media_header = translate_content(category_by_media_header, 'auto', language)
        category_by_media_desc = translate_content(category_by_media_desc, 'auto', language)
        narrative_header = translate_content(narrative_header, 'auto', language)
        narrative_desc = translate_content(narrative_desc, 'auto', language)
        narrative_explanation = translate_content(narrative_explanation, 'auto', language)
        org_analysis_header = translate_content(org_analysis_header, 'auto', language)
        org_chart_desc = translate_content(org_chart_desc, 'auto', language)
        people_analysis_header = translate_content(people_analysis_header, 'auto', language)
        people_chart_desc = translate_content(people_chart_desc, 'auto', language)

    try:
        logging.info(f"Starting journalist analytics generation for {journalist_name}")
        
        # Step 1: Determine main categories and classify articles
        main_categories = determine_main_categories(articles_sorted, journalist_name, general_folder)
        classified_articles = classify_articles(journalist_name, articles_sorted, main_categories, general_folder)
        
        # Step 2: Extract entities and analyze sentiments
        articles_with_entities = extract_entities(classified_articles, journalist_name, general_folder)
        articles_with_sentiments = analyze_all_sentiments(articles_with_entities, journalist_name, general_folder)
        
        # Step 3: Generate initial categorization markdown
        categorization_md = generate_categorization_markdown(articles_with_sentiments, journalist_name, general_folder)
        
        # Step 4: Update markdown with sentiment information
        updated_markdown = update_markdown_with_sentiments(
            articles_with_sentiments,
            categorization_md,
            journalist_name,
            general_folder
        )

        save_data_to_json(articles_with_sentiments, f"{general_folder}/Outputs/CompiledOutputs/CategorizedArticles_{journalist_name}.json")
        
        # Create DataFrame for analysis
        df = pd.DataFrame(articles_with_sentiments)

        # Generate media outlet distribution chart
        media_counts = df['media_outlet'].value_counts()
        total_outlets = media_counts.sum()
        threshold = 0.04  # 4% threshold

        # Filter small slices
        main_outlets = media_counts[media_counts/total_outlets >= threshold]
        other_outlets = media_counts[media_counts/total_outlets < threshold]

        if not other_outlets.empty:
            main_outlets['Others'] = other_outlets.sum()

        # Create professional pie chart
        fig = create_professional_pie(
            main_outlets,
            'Distribution of Articles by Media Outlet',
            figsize=(16, 8)
        )
        media_outlets_chart = save_plot_base64()
        plt.close(fig)

        # Generate category distribution chart 
        category_counts = df['main_category'].value_counts()
        total_categories = category_counts.sum()
        threshold = 0.04  # 4% threshold

        main_categories = category_counts[category_counts/total_categories >= threshold]
        other_categories = category_counts[category_counts/total_categories < threshold]

        if not other_categories.empty:
            main_categories['Others'] = other_categories.sum()

        fig = create_professional_pie(
            main_categories,
            f'Distribution of Articles by Category - {journalist_name}',
            figsize=(16, 8)
        )
        categories_chart = save_plot_base64()
        plt.close(fig)

        # Prepare data for AI analysis
        category_analysis_data = {}
        for category in df['main_category'].unique():
            category_articles = df[df['main_category'] == category]
            category_analysis_data[category] = {
                'count': len(category_articles),
                'subcategories': category_articles.groupby('subcategory').size().to_dict(),
                'topics': category_articles['specific_topic'].tolist()
            }

        # Generate AI analysis of coverage
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1000
        )

        analysis_prompt = f"""
Provide a high-level description of {journalist_name}'s coverage based on the following categories and corresponding structured data.
The data shows the distribution of the articles across different categories, subcategories, and specific topics.

Coverage Data:
{json.dumps(category_analysis_data, indent=2)}

Format your response as a plain text structured with few and clear paragraphs without headers.
"""

        ai_analysis = chatbot.ask(analysis_prompt)

        # Generate the markdown report
        markdown_content = f"""
# Analytics Report: {journalist_name}

## {distribution_header}

{distribution_intro}

### {media_outlet_header}
{media_outlet_desc}

![Media Outlet Distribution](data:image/png;base64,{media_outlets_chart})

### {category_dist_header}
{category_dist_desc}

![Category Distribution](data:image/png;base64,{categories_chart})

<div style="page-break-before: always;"></div>
"""

        # Add AI analysis
        markdown_content += f"""\n ## What is {journalist_name} writing about? \n"""
        if language.lower() != 'english':
            ai_analysis = translate_content(ai_analysis, 'auto', language)
        markdown_content += ai_analysis
        
        # Get top 6 media outlets
        media_counts = df['media_outlet'].value_counts()
        top_media = media_counts.head(6)

        # Create data dictionary for multiple pie charts
        media_cat_data = {}
        for outlet in top_media.index:
            outlet_data = df[df['media_outlet'] == outlet]['main_category'].value_counts()
            media_cat_data[outlet] = outlet_data

        # Create multiple pie charts
        fig = create_multiple_pie_charts(
            media_cat_data,
            f'Category Distribution Across Top Media Outlets - {journalist_name}'
        )

        # Save plot and close figure
        media_cats_chart = save_plot_base64()
        plt.close(fig)

        markdown_content += f"""
### {category_by_media_header}
{category_by_media_desc}

![Category Distribution by Media](data:image/png;base64,{media_cats_chart})
<div style="page-break-before: always;"></div>
        """

        # Check if articles already have narrative stories
        need_categorization = not all('narrative_story' in article for article in articles_with_sentiments)

        if need_categorization:
            logging.info("Narrative stories not found in articles. Running categorization...")
            # Get categorized articles with narrative stories
            categories_data, articles_with_sentiments = get_coverage_categories(
                articles_with_sentiments, journalist_name, general_folder
            )
        else:
            logging.info("Using existing narrative stories from articles")
        
        # Create DataFrame for analysis using narrative_story field
        df = pd.DataFrame(articles_with_sentiments)

        # Generate narrative distribution visualization
        narrative_counts = df['narrative_story'].value_counts()
        total_articles = len(df)
        threshold = 0.04

        main_narratives = narrative_counts[narrative_counts/total_articles >= threshold]
        other_narratives = narrative_counts[narrative_counts/total_articles < threshold]

        if not other_narratives.empty:
            main_narratives['Other Narratives'] = other_narratives.sum()

        # Clean narrative names by removing asterisks
        main_narratives.index = main_narratives.index.str.replace('**', '')

        # Create and save narrative distribution chart
        fig = create_professional_pie(
            main_narratives,
            f'Distribution of Narrative Threads in {journalist_name}\'s Coverage',
            figsize=(16, 8)
        )
        narrative_coverage_chart = save_plot_base64()
        plt.close(fig)

        # Get narrative distribution and sample articles
        narrative_counts = df['narrative_story'].value_counts()
        top_3_narratives = narrative_counts.head(3)

        # Collect sample articles for each narrative
        narrative_samples = {}
        for narrative in top_3_narratives.index:
            narrative_articles = df[df['narrative_story'] == narrative]
            if len(narrative_articles) > 10:
                narrative_articles = narrative_articles.sample(n=10)
            narrative_samples[narrative] = [
                {
                    'content': row['content'],
                    'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date'],
                    'media_outlet': row.get('media_outlet', 'Unknown'),
                    'tone': row.get('tone', 'Neutral'),
                    'sentiment_score': row.get('sentiment score', 0)
                }
                for _, row in narrative_articles.iterrows()
            ]

        # Generate narrative analysis
        narrative_analysis_prompt = f"""
Analyze the top 3 narrative threads in {journalist_name}'s coverage. For each narrative, explain both:
1. What the narrative is about - the key events, developments, or themes that make up this story thread
2. {journalist_name}'s apparent stance, opinion or perspective on these topics based on their coverage

Narratives to analyze:

{json.dumps({
    narrative: {
        'article_count': int(count),
        'sample_articles': narrative_samples[narrative]
    }
    for narrative, count in top_3_narratives.items()
}, indent=2)}

Provide a flowing analysis in 2-3 paragraphs without any headers. Focus on connecting the narrative content with {journalist_name}'s perspective.
Keep the analysis concise but insightful. Consider tone and sentiment patterns in your analysis.
"""

        chatbot = ChatGPT(
            system_prompt="You are a media analysis expert specializing in identifying narrative patterns and journalistic perspectives in news coverage.",
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=2000
        )

        narrative_analysis = chatbot.ask(narrative_analysis_prompt)

        if language.lower() != 'english':
            narrative_analysis = translate_content(narrative_analysis, 'auto', language)

        markdown_content += f"""
### {narrative_header}
{narrative_desc}

![Narrative Coverage Distribution](data:image/png;base64,{narrative_coverage_chart})

{narrative_analysis}
"""

# ------------------------------
# Organizations Data Processing (unchanged)
# ------------------------------
        org_data = []
        try:
            categorized_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                            f"CategorizedArticles_{journalist_name}.json")
            logging.info(f"Loading categorized data from: {categorized_path}")
            with open(categorized_path, 'r', encoding='utf-8') as f:
                articles_with_entities = json.load(f)
            for article in articles_with_entities:
                logging.info(f"Processing organizations from article: {article.get('title')}")
                for org in article.get('organizations', []):
                    if org and isinstance(org, dict) and 'name' in org:
                        org_data.append({
                            'name': org['name'],
                            'type': org.get('type', 'Unknown'),
                            'tone': org.get('tone', 'Neutral'),
                            'sentiment_score': org.get('sentiment_score', 0),
                            'description': org.get('description', ''),
                            'article_content': article.get('content', '')
                        })
            logging.info(f"Collected {len(org_data)} organizations")
        except Exception as e:
            logging.error(f"Error loading categorized data: {str(e)}")
            logging.error(traceback.format_exc())

        org_df = pd.DataFrame(org_data)
        org_chart = ""
        org_analysis = ""

        if not org_df.empty and 'name' in org_df.columns:
            # Use top 10 for the graph...
            top_orgs_graph = org_df['name'].value_counts().head(10)
            if not top_orgs_graph.empty:
                org_tone_df = pd.DataFrame([
                    org_df[org_df['name'] == org]['tone'].value_counts()
                    for org in top_orgs_graph.index
                ], index=top_orgs_graph.index).fillna(0)
                fig, ax = create_stacked_bar_chart(
                    data=org_tone_df,
                    title='Top 10 Most Mentioned Organizations with Tone Distribution',
                    xlabel='Organization',
                    ylabel='Number of Mentions'
                )
                org_chart = save_plot_base64()
                plt.close(fig)

            # For analysis, use top 3 organizations
            top_orgs_analysis_series = org_df['name'].value_counts().head(3)
            top_orgs_data = []
            for org_name in top_orgs_analysis_series.index:
                org_articles = org_df[org_df['name'] == org_name]
                if len(org_articles) > 25:
                    org_articles = org_articles.sample(n=25)
                org_info = {
                    'name': org_name,
                    'mentions': len(org_articles),
                    'avg_sentiment': org_articles['sentiment_score'].mean(),
                    'tones': org_articles['tone'].value_counts().to_dict(),
                    'descriptions': org_articles['description'].tolist(),
                    'articles_content': org_articles['article_content'].tolist()
                }
                top_orgs_data.append(org_info)

            top_orgs_analysis_prompt = f"""
Analyze how {journalist_name} covers the following top 3 organizations.
Focus on key narrative themes and overall sentiment patterns.
Data:
{json.dumps(top_orgs_data, indent=2)}
Provide your analysis in a single, concise paragraph.
"""
            top_orgs_analysis = chatbot.ask(top_orgs_analysis_prompt)

            # Extreme Sentiment Analysis for organizations
            remaining_orgs = org_df[org_df['name'].isin(top_orgs_graph.index.difference(top_orgs_analysis_series.index))]
            org_sentiments = remaining_orgs.groupby('name')['sentiment_score'].mean().to_dict()
            sorted_negative = sorted(org_sentiments.items(), key=lambda x: x[1])
            most_negative_orgs = sorted_negative[:3]
            sorted_positive = sorted(org_sentiments.items(), key=lambda x: x[1], reverse=True)
            most_positive_orgs = sorted_positive[:3]

            extreme_negative_data = []
            for org, sentiment in most_negative_orgs:
                org_articles = org_df[org_df['name'] == org]
                if len(org_articles) > 25:
                    org_articles = org_articles.sample(n=25)
                org_info = {
                    'name': org,
                    'average_sentiment': sentiment,
                    'mentions': len(org_articles),
                    'tones': org_articles['tone'].value_counts().to_dict(),
                    'descriptions': org_articles['description'].tolist(),
                    'articles_content': org_articles['article_content'].tolist()
                }
                extreme_negative_data.append(org_info)

            extreme_positive_data = []
            for org, sentiment in most_positive_orgs:
                org_articles = org_df[org_df['name'] == org]
                if len(org_articles) > 25:
                    org_articles = org_articles.sample(n=25)
                org_info = {
                    'name': org,
                    'average_sentiment': sentiment,
                    'mentions': len(org_articles),
                    'tones': org_articles['tone'].value_counts().to_dict(),
                    'descriptions': org_articles['description'].tolist(),
                    'articles_content': org_articles['article_content'].tolist()
                }
                extreme_positive_data.append(org_info)

            positive_extreme_analysis_prompt = f"""
Analyze how {journalist_name} covers the following three organizations that portray the most positive sentiment.
Data:
{json.dumps(extreme_positive_data, indent=2)}
Provide your analysis in a single, concise paragraph.
"""
            negative_extreme_analysis_prompt = f"""
Analyze how {journalist_name} covers the following three organizations that portray the most negative sentiment.
Data:
{json.dumps(extreme_negative_data, indent=2)}
Provide your analysis in a single, concise paragraph.
"""
    
            positive_extreme_chatbot = ChatGPT(
                system_prompt="You are a media analysis expert focusing on extreme positive portrayals. Provide a concise, single-paragraph analysis based on the data provided.",
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1500
            )
            positive_extreme_analysis = positive_extreme_chatbot.ask(positive_extreme_analysis_prompt)

            negative_extreme_chatbot = ChatGPT(
                system_prompt="You are a media analysis expert focusing on extreme negative portrayals. Provide a concise, single-paragraph analysis based on the data provided.",
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1500
            )
            negative_extreme_analysis = negative_extreme_chatbot.ask(negative_extreme_analysis_prompt)

            org_analysis = f"{top_orgs_analysis}\n\n{positive_extreme_analysis}\n\n{negative_extreme_analysis}"
        else:
            org_analysis = "No organization data was found in the articles analyzed."
            
        if org_chart:
            markdown_content += f"""
<div style="page-break-before: always;"></div>

## {org_analysis_header}

{org_chart_desc}

![Top Organizations](data:image/png;base64,{org_chart})

### Analysis of Organizations
"""
            if language.lower() != 'english':
                org_analysis = translate_content(org_analysis, 'auto', language)
            markdown_content += org_analysis
        else:
            markdown_content += f"""
## Analysis of Organizations

No significant organization mentions were found in the analyzed articles.
"""

# ------------------------------
# People Data Processing
# ------------------------------
        people_data = []
        try:
            for article in articles_with_entities:
                logging.info(f"Processing people from article: {article.get('title')}")
                for person in article.get('people', []):
                    if person and isinstance(person, dict) and 'name' in person:
                        people_data.append({
                            'name': person['name'],
                            'role': person.get('role', 'Unknown'),
                            'tone': person.get('tone', 'Neutral'),
                            'sentiment_score': person.get('sentiment_score', 0),
                            'context': person.get('context', ''),
                            'article_content': article.get('content', '')
                        })
            logging.info(f"Collected {len(people_data)} people")
        except Exception as e:
            logging.error(f"Error processing people data: {str(e)}")

        people_df = pd.DataFrame(people_data)
        people_chart = ""
        people_analysis = ""

        if not people_df.empty and 'name' in people_df.columns:
            # Create the graph using the top 10 people
            top_people_graph = people_df['name'].value_counts().head(10)
            if not top_people_graph.empty:
                people_tone_df = pd.DataFrame([
                    people_df[people_df['name'] == person]['tone'].value_counts()
                    for person in top_people_graph.index
                ], index=top_people_graph.index).fillna(0)
                fig, ax = create_stacked_bar_chart(
                    data=people_tone_df,
                    title='Top 10 Most Mentioned People with Tone Distribution',
                    xlabel='Person',
                    ylabel='Number of Mentions'
                )
                people_chart = save_plot_base64()
                plt.close(fig)

            # For analysis, use the top 3 people
            top_people_analysis_series = people_df['name'].value_counts().head(3)
            top_people_data = []
            # --- Instantiate the chatbot once (outside the loop) ---
            people_chatbot = ChatGPT(
                system_prompt="You are a media analysis expert specializing in identifying narrative patterns and journalistic perspectives in news coverage.",
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=2000
            )
            for person_name in top_people_analysis_series.index:
                person_articles = people_df[people_df['name'] == person_name]
                if len(person_articles) > 25:
                    person_articles = person_articles.sample(n=25)
                person_info = {
                    'name': person_name,
                    'role': person_articles['role'].iloc[0] if not person_articles['role'].empty else 'Unknown',
                    'mentions': len(person_articles),
                    'avg_sentiment': person_articles['sentiment_score'].mean(),
                    'tones': person_articles['tone'].value_counts().to_dict(),
                    'contexts': person_articles['context'].tolist(),
                    'articles_content': person_articles['article_content'].tolist()
                }
                top_people_data.append(person_info)

            top_people_analysis_prompt = f"""
Analyze how {journalist_name} covers the following top 3 individuals.
Focus on key narrative themes and overall sentiment patterns.
Data:
{json.dumps(top_people_data, indent=2)}
Provide your analysis in a single, concise paragraph, without indenting your reponse.
"""
            top_people_analysis = people_chatbot.ask(top_people_analysis_prompt)
            
            # Extreme Sentiment Analysis for People
            remaining_people = people_df[people_df['name'].isin(top_people_graph.index.difference(top_people_analysis_series.index))]
            people_sentiments = remaining_people.groupby('name')['sentiment_score'].mean().to_dict()
            sorted_negative = sorted(people_sentiments.items(), key=lambda x: x[1])
            most_negative_people = sorted_negative[:3]
            sorted_positive = sorted(people_sentiments.items(), key=lambda x: x[1], reverse=True)
            most_positive_people = sorted_positive[:3]
            
            extreme_negative_data = []
            for person, sentiment in most_negative_people:
                person_articles = people_df[people_df['name'] == person]
                if len(person_articles) > 25:
                    person_articles = person_articles.sample(n=25)
                person_info = {
                    'name': person,
                    'average_sentiment': sentiment,
                    'mentions': len(person_articles),
                    'tones': person_articles['tone'].value_counts().to_dict(),
                    'contexts': person_articles['context'].tolist(),
                    'articles_content': person_articles['article_content'].tolist()
                }
                extreme_negative_data.append(person_info)
            
            extreme_positive_data = []
            for person, sentiment in most_positive_people:
                person_articles = people_df[people_df['name'] == person]
                if len(person_articles) > 25:
                    person_articles = person_articles.sample(n=25)
                person_info = {
                    'name': person,
                    'average_sentiment': sentiment,
                    'mentions': len(person_articles),
                    'tones': person_articles['tone'].value_counts().to_dict(),
                    'contexts': person_articles['context'].tolist(),
                    'articles_content': person_articles['article_content'].tolist()
                }
                extreme_positive_data.append(person_info)
            
            positive_extreme_analysis_prompt = f"""
Analyze how {journalist_name} covers the following three individuals that portray the most positive sentiment.
Data:
{json.dumps(extreme_positive_data, indent=2)}
Provide your analysis in a single, concise paragraph.
"""
            negative_extreme_analysis_prompt = f"""
Analyze how {journalist_name} covers the following three individuals that portray the most negative sentiment.
Data:
{json.dumps(extreme_negative_data, indent=2)}
Provide your analysis in a single, concise paragraph.
"""
    
            positive_extreme_chatbot = ChatGPT(
                system_prompt="You are a media analysis expert focusing on extreme positive portrayals. Provide a concise, single-paragraph analysis based on the data provided.",
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1500
            )
            positive_extreme_analysis = positive_extreme_chatbot.ask(positive_extreme_analysis_prompt)

            negative_extreme_chatbot = ChatGPT(
                system_prompt="You are a media analysis expert focusing on extreme negative portrayals. Provide a concise, single-paragraph analysis based on the data provided.",
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1500
            )
            negative_extreme_analysis = negative_extreme_chatbot.ask(negative_extreme_analysis_prompt)

            people_analysis = f"{top_people_analysis}\n\n{positive_extreme_analysis}\n\n{negative_extreme_analysis}"
        else:
            people_analysis = "No data about individuals was found in the articles analyzed."

        # --- Instead of checking people_chart, always add the people analysis if available ---
        markdown_content += f"""
<div style="page-break-before: always;"></div>

## {people_analysis_header}

{people_chart_desc}

![Top People](data:image/png;base64,{people_chart})

### Analysis of People
        """

        if language.lower() != 'english':
            people_analysis = translate_content(people_analysis, 'auto', language)
        markdown_content += people_analysis

        # Save the report
        report_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                    f"Analytics_{journalist_name}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logging.info("Journalist analysis report generation completed")
        return markdown_content
        
    except Exception as e:
        logging.error(f"Error in generate_journalist_analysis: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error generating journalist analysis: {str(e)}"