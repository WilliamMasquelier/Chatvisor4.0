# Media Coverage Analysis Tool

## Overview

This Media Coverage Analysis Tool is a comprehensive solution for analyzing news media coverage of companies and journalists. The system processes articles from various sources, extracts meaningful insights, and generates detailed reports in multiple formats. The tool supports two primary analysis modes:

1. **Media Coverage Analysis**: Analyze how a specific company is portrayed across various media outlets
2. **Journalist Analysis**: Examine a journalist's coverage patterns, topics, and sentiment across their articles

## Key Features

### Company Analysis Features

- **Articles Reference List**: Generate structured lists of articles about a company with metadata
- **Summary Insights**: Extract key points from each article with visualizations of coverage distribution
- **Comprehensive Issues Analysis**: Identify, categorize, and analyze issues and negative press
- **Topic Summaries**: Group articles by topic and provide detailed analysis of each topic area
- **Media Analytics**: Generate visualizations and sentiment analysis across media outlets
- **Stakeholder Quotes Analysis**: Extract and analyze quotes from stakeholders
- **Consolidated Stakeholder Analysis**: Aggregate stakeholder opinions and perspectives

### Journalist Analysis Features

- **Articles List**: Categorize and list articles by a specific journalist
- **Journalist Coverage Profile**: Analyze a journalist's coverage patterns and preferences
- **Topic-Focused Analysis**: Examine how a journalist covers specific topics
- **Journalist Analytics Report**: Comprehensive report with visualizations on coverage patterns

## Architecture and Components

### Main Application File

- `MediaCoverageAnalysis.py`: Contains the main application logic and Gradio interface

### Core Classes

- `DocumentProcessor`: Handles extraction and processing of articles from PDF and DOCX files
- `CitationProcessor`: Processes citations in markdown for professional formatting
- `ProgramSummaryTracker`: Tracks execution metrics like processing time and token usage
- `ChatGPT` and `BigSummarizerGPT`: Wrapper classes for interacting with language models, handling prompting, retries, and response processing. These classes abstract the complexity of working with AI models and ensure consistent outputs.

### Utility Modules

- `Helpers.py`: Contains general helper functions for file processing, visualization, and analysis
- `Outputs.py`: Contains functions that generate specific output types

## Output Analysis Content and Insights

### Company Analysis Outputs

#### 1. Articles Reference List (`generate_journalist_list_output`)
This output provides a comprehensive catalog of all media articles related to the target company, including:
- Total article count and unique journalists/media outlets statistics
- Chronologically ordered list of articles with author, outlet, date, and title
- Journalist statistics showing top contributors
- Media outlet statistics showing coverage distribution
- Hyperlinks to original articles where available

#### 2. Summary Insights (`generate_insights_output`)
This analysis extracts essential information from each article with:
- Visualizations showing media coverage distribution by outlet and publication timeline
- 2-5 key insights from each article, focusing on the most significant developments, announcements, and implications
- Special attention to strategic moves, market position, and industry impact
- Identification of positive developments and potential challenges/concerns

#### 3. Comprehensive Issues Analysis (`generate_issue_analysis_output`)
This in-depth analysis identifies and categorizes negative issues and challenges:
- Executive summary with visualization of issue distribution
- Risk ranking of issue categories by severity and impact
- Chronological analysis of how each issue developed over time
- Impact assessment on reputation, operations, and stakeholder trust
- Analysis of stakeholder perspectives on each issue
- Business model implications for each issue category
- Conclusion with future outlook and cross-cutting concerns

#### 4. Topic Summaries (`generate_topics_output`)
This analysis organizes coverage into distinct topic categories:
- Distribution visualization showing article volume by topic
- For each topic:
  - Chronological analysis of early, developing, and recent coverage
  - Stakeholder perspectives from different entities involved
  - Implications and future outlook for the topic
  - Citations to specific media sources throughout
  - List of all articles covering the topic

#### 5. Media Analytics (`generate_analytics_output`)
This data-driven analysis shows sentiment patterns across the media landscape:
- Media outlet distribution visualization
- Sentiment trend analysis showing how coverage sentiment evolved over time
- Journalist analysis showing which writers cover the company most frequently
- Category analysis showing positive/negative coverage by topic
- Detailed analysis of coverage peaks and sentiment shifts with explanations
- Analysis of most discussed organizations and people with sentiment indicators

#### 6. Stakeholder Quotes Analysis (`generate_stakeholder_quotes`)
This analysis extracts direct quotes from key stakeholders:
- Organized table of stakeholders, their roles, and direct quotes
- Translation of non-English quotes
- Sentiment analysis for each stakeholder statement
- Context for understanding the stakeholder's perspective
- Deduplication and normalization of similar quotes

#### 7. Consolidated Stakeholder Analysis (`generate_consolidated_stakeholder_analysis`)
This higher-level analysis summarizes stakeholder opinions:
- Consolidated view of each stakeholder's overall position toward the company
- Analysis of opinion consistency or evolution over time
- Assessment of the potential impact of stakeholder opinions on reputation
- Identification of notable patterns in stakeholder sentiment

### Journalist Analysis Outputs

#### 1. Journalist's Articles List (`generate_journalist_article_list`)
This provides a structured list of a journalist's articles:
- Total articles count and media outlet statistics
- Chronologically ordered list of articles with outlet, date, and title
- Media outlet distribution showing where the journalist's work appears
- Hyperlinks to original articles where available

#### 2. Journalist Profile Analysis (`generate_journalist_profile`)
This comprehensive analysis examines the journalist's coverage patterns:
- Main topics and narratives in the journalist's body of work
- For each coverage category:
  - Detailed narrative analysis showing how the journalist covers key stories
  - Analysis of the journalist's perspective and stance on topics
  - Evidence-based assessment using direct quotes and references
  - Evolution of the journalist's position over time, if applicable
- List of articles for each coverage category

#### 3. Topic-Focused Analysis (`analyze_journalist_topic_coverage`)
This targeted analysis examines how the journalist covers a specific topic:
- Introduction with scope and methodology
- Comprehensive coverage analysis showing how the journalist reports on the topic
- Sentiment analysis revealing the journalist's stance on the topic
- Supporting evidence through direct quotes and examples
- Chronological development of the journalist's coverage
- List of all articles discussing the specified topic

#### 4. Journalist Analytics Report (`generate_journalist_analysis_output`)
This data-driven analysis provides visualizations and AI-powered insights:
- Distribution of coverage across media outlets and categories
- Narrative thread analysis showing recurring stories and themes
- Analysis of top journalists' coverage and sentiment
- Analysis of most discussed organizations with sentiment distribution
- Analysis of most discussed people with sentiment distribution
- AI-generated insights explaining coverage patterns and sentiment trends

### Analysis Functions

#### Company-Specific Analysis

- `generate_issue_analysis_output()`: Generate comprehensive issues analysis
- `generate_topics_output()`: Generate topic-based analysis of coverage
- `generate_analytics_output()`: Generate visualizations and media analytics
- `generate_stakeholder_quotes()`: Extract and analyze stakeholder quotes
- `generate_consolidated_stakeholder_analysis()`: Provide holistic stakeholder analysis

#### Journalist-Specific Analysis

- `generate_journalist_profile()`: Analyze a journalist's overall coverage patterns
- `analyze_journalist_topic_coverage()`: Analyze a journalist's coverage of specific topics
- `generate_journalist_analysis_output()`: Generate comprehensive journalist analytics

## Input Data

The tool accepts the following input formats:
- PDF files containing articles
- DOCX files with article content (separated by "--")

## Processing Pipeline

### Media Coverage Analysis Pipeline

1. **Document Processing**: Extract articles from PDF/DOCX files
2. **Metadata Extraction**: Extract title, author, date, media outlet
3. **Relevance Filtering**: Filter articles based on relevance to target company
4. **Deduplication**: Remove duplicate articles
5. **Insight Generation**: Extract key insights from each article
6. **Categorization**: Group articles by topic/category
7. **Entity Extraction**: Identify organizations and people mentioned
8. **Sentiment Analysis**: Analyze sentiment toward the company and entities
9. **Report Generation**: Generate comprehensive reports based on user selections

### Journalist Analysis Pipeline

1. **Document Processing**: Extract articles by the target journalist
2. **Metadata Extraction**: Extract article metadata
3. **Category Determination**: Identify main coverage categories
4. **Narrative Analysis**: Identify recurring stories and coverage patterns
5. **Entity Extraction**: Identify organizations and people mentioned
6. **Sentiment Analysis**: Analyze sentiment patterns in the journalist's coverage
7. **Report Generation**: Generate comprehensive profile and analytics

## AI Integration

The tool leverages various AI models through the `ChatGPT` and `BigSummarizerGPT` classes for:
- Content extraction and summarization
- Categorization and topic modeling
- Entity recognition
- Sentiment analysis
- Narrative and perspective analysis

The AI system employs a multi-step process for each analysis:
1. **Initial processing**: Extracting key information from raw articles
2. **Classification**: Categorizing content into relevant topic areas
3. **Synthesis**: Combining information from multiple articles for coherent analysis
4. **Pattern recognition**: Identifying trends, shifts, and notable developments
5. **Sentiment analysis**: Evaluating tone and sentiment toward entities and topics
6. **Report generation**: Creating coherent, structured reports from the analysis

## Extending the Tool

The modular architecture allows for easy extension:
- Add new analysis types by creating new output generation functions
- Integrate new visualization types in the helpers module, while leveraging existing functions for streamlined layout
- Add support for additional input formats