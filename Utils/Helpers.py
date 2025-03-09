import os
import re
import json
import collections
import time
import pandas as pd
import marvin
import fitz
from pathlib import Path
from datetime import datetime, timedelta, date
from tabulate import tabulate
from scipy.spatial.distance import cosine
from scipy.signal import find_peaks
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from Classes.SimplifiedChatbots import ChatGPT, BigSummarizerGPT
from Classes.DocumentProcessor import DocumentProcessor, CitationProcessor
#from Utils.Outputs import *
from difflib import SequenceMatcher
from collections import OrderedDict
from collections import defaultdict
import logging
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import base64
import io
from pathlib import Path
import pypandoc
import tempfile
from logging.handlers import RotatingFileHandler
import sys

def extract_hyperlinks(pdf_path):
    """
    Extract hyperlinks from a PDF file with enhanced error handling and debugging.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of extracted hyperlinks
    """
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return []
            
        # Open PDF and extract links
        doc = fitz.open(pdf_path)
        links = []
        
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                link_list = page.get_links()
                
                for link in link_list:
                    if link.get('uri') or link.get('kind') == fitz.LINK_URI:
                        uri = link.get('uri', '')
                        if uri and uri not in links:  # Avoid duplicates
                            links.append(uri)
                            print(f"Found link in {os.path.basename(pdf_path)}: {uri}")
                
                # Also look for URL patterns in text
                text = page.get_text()
                # Basic URL pattern matching
                url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
                text_urls = re.findall(url_pattern, text)
                
                for url in text_urls:
                    if url not in links:  # Avoid duplicates
                        links.append(url)
                        print(f"Found URL in text of {os.path.basename(pdf_path)}: {url}")
                        
            except Exception as e:
                print(f"Error processing page {page_num} in {pdf_path}: {str(e)}")
                continue
                
        doc.close()
        return links
        
    except Exception as e:
        print(f"Error processing PDF {os.path.basename(pdf_path)}: {str(e)}")
        return []

def add_links_to_articles(articles):
    """
    Add hyperlinks to articles with enhanced error handling.
    
    Args:
        articles (list): List of article dictionaries
        
    Returns:
        list: Updated articles with hyperlinks
    """
    for article in articles:
        try:
            if 'file_path' not in article:
                print(f"No file path found for article: {article.get('title', 'Unknown')}")
                article['link'] = None
                continue
                
            file_path = article['file_path']
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                article['link'] = None
                continue
                
            hyperlinks = extract_hyperlinks(file_path)
            
            if hyperlinks:
                article['link'] = hyperlinks[0]  # Use first hyperlink found
                print(f"Added link to article {article.get('title', 'Unknown')}: {hyperlinks[0]}")
            else:
                print(f"No hyperlinks found in {file_path}")
                article['link'] = None
                
        except Exception as e:
            print(f"Error processing article {article.get('title', 'Unknown')}: {str(e)}")
            article['link'] = None
            
    return articles

def check_input_paths(pdf_folder_path=None, docx_file_path=None):
        """
        Check if the provided paths exist and are accessible.
        Returns True if at least one valid path is provided and accessible.
        """
        valid_path_found = False

        if pdf_folder_path:
            if os.path.exists(pdf_folder_path):
                print(f"✓ PDF folder path exists: {pdf_folder_path}")
                # Check if it contains any PDF files
                pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]
                if pdf_files:
                    print(f"✓ Found {len(pdf_files)} PDF files:")
                    for pdf in pdf_files:
                        print(f"  - {pdf}")
                    valid_path_found = True
                else:
                    print("✗ No PDF files found in the folder")
            else:
                print(f"✗ PDF folder path does not exist: {pdf_folder_path}")

        if docx_file_path:
            if os.path.exists(docx_file_path):
                print(f"✓ DOCX file exists: {docx_file_path}")
                valid_path_found = True
            else:
                print(f"✗ DOCX file does not exist: {docx_file_path}")

        return valid_path_found

def parse_relative_date(date_str):
    """
    Parse relative date strings like "2 years ago" and convert them to absolute dates.
    Returns date in "Month Day, Year" format.
    """
    now = datetime.now()
    
    # Handle relative dates
    if 'ago' in date_str.lower():
        match = re.search(r'(\d+)\s*(year|month|week|day)s?\s*ago', date_str.lower())
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            
            if unit == 'year':
                delta = timedelta(days=amount * 365)
            elif unit == 'month':
                delta = timedelta(days=amount * 30)
            elif unit == 'week':
                delta = timedelta(weeks=amount)
            else:  # days
                delta = timedelta(days=amount)
                
            result_date = now - delta
            return result_date.strftime('%B %d, %Y')
    
    # Handle existing absolute date formats
    try:
        # Try parsing standard format first
        parsed_date = datetime.strptime(date_str, '%B %d, %Y')
        return parsed_date.strftime('%B %d, %Y')
    except ValueError:
        pass
    
    try:
        # Try parsing other common formats
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d'):
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%B %d, %Y')
            except ValueError:
                continue
    except Exception:
        pass
    
    # If no valid date format is found, return a default date
    return 'January 1, 2024'

def clean_date_string(date_str):
    """
    Clean and standardize date strings before parsing.
    """
    # Remove extra whitespace and normalize format
    date_str = ' '.join(date_str.split())
    
    # Handle common variations
    date_str = date_str.replace('- (extract the date in the format Month Day, Year - IN ENGLISH)', '')
    date_str = date_str.replace('[Date not provided in the document]', 'January 1, 2024')
    
    # Remove any parenthetical notes
    date_str = re.sub(r'\([^)]*\)', '', date_str)
    
    return date_str.strip()

def process_article_date(article):
    """
    Process the date field in an article dictionary.
    """
    if 'date' not in article:
        article['date'] = 'January 1, 2024'
        return article
    
    try:
        # Clean the date string
        cleaned_date = clean_date_string(article['date'])
        
        # Parse the date
        parsed_date = parse_relative_date(cleaned_date)
        
        # Update the article
        article['date'] = parsed_date
        
        # Convert to timestamp for sorting
        article['timestamp'] = datetime.strptime(parsed_date, '%B %d, %Y').timestamp()
        
    except Exception as e:
        print(f"Error processing date '{article.get('date', 'No date')}': {str(e)}")
        # Set default date if parsing fails
        article['date'] = 'January 1, 2024'
        article['timestamp'] = datetime.strptime('January 1, 2024', '%B %d, %Y').timestamp()
    
    return article

def ensure_directory_exists(file_path):
    """
    Ensure that the directory containing the specified file path exists.
    If not, create the directory.
    """
    try:
        os.makedirs(file_path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {file_path}: {str(e)}")
        raise

def process_pdfs(file_folder):
    processed_files = []
    error_files = []

    for filename in os.listdir(file_folder):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(file_folder, filename)
            try:
                with fitz.open(file_path) as doc:
                    # Just try to access the first page to check if the PDF is valid
                    doc.load_page(0)
                processed_files.append(file_path)
            except Exception as e:
                error_files.append((filename, str(e)))

    return processed_files, error_files

def get_files(folder_path):
    processed_files = []
    error_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                with fitz.open(file_path) as doc:
                    # Just try to access the first page to check if the PDF is valid
                    doc.load_page(0)
                processed_files.append(file_path)
            except Exception as e:
                error_files.append((filename, str(e)))

    if error_files:
        print("The following files could not be processed:")
        for filename, error in error_files:
            print(f"- {filename}: {error}")
    
    return processed_files

def save_data_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data successfully saved to '{filename}'.")

def load_data_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    print(f"Data successfully loaded from '{filename}'.")
    return data

def extract_metadata(articles: List[Dict]) -> List[Dict]:
    for article in articles:
        chatbot = ChatGPT(model_name="chatgpt-4o-latest", max_tokens=300, temperature=0)

        prompt = f"""
You will be given a document from which you have to extract the metadata. You must extract its title, the author, the media outlet, the date of publication and the media type (whether the outlet is a 'National' outlet, 'Negional' outlet or an 'Industry-specific/Trade-press' outlet). 
In case the author is not explicitly mentioned at the beginning of the document, you should return the value "Anonymous".
When extracting the media outlet, you should return the name of the media outlet without any extension like ".com", ".nl", ".fr", etc.
The value of the media type should be one of the following: 'National', 'Regional' or 'Industry-specific/Trade-press'.
                           
Here is the document:{article['content']}

Format your response as follows:
Title: [Title of the document]
Author: [Author of the document]
Media Outlet: [Name of the media outlet]
Date of Publication: [Date of publication]- (extract the date in the format Month Day, Year - IN ENGLISH)
Media type: ['National', 'Regional' or 'Industry-specific/Trade-press']

Here is an example of document provided: Document(metadata='source': "KnowledgeBase/CompanyAnalysis/Embraer/MediaCoverage/NewsEmbraer/What near-disasters, 'SNL' jabs mean for Alaska 's reputation.PDF", 'page': [1, 2, 3], page_content='Page 1 of 3\nWhat near-disasters, \'SNL\' jabs mean for Alaska \'s reputation\nWhat near-disasters, \'SNL\' jabs mean for Alaska\'s reputation - Correction \nAppended\nThe Seattle Times\nJanuary 25, 2024 Thursday\n Correction Appended\nCopyright 2024 The Seattle Times Company All Rights Reserved\nSection: Pg. A 1\nLength: 1340 words\nByline: Renata Geraldo, Seattle Times staff reporter\nBody\nWith two flights narrowly escaping disaster just months apart, Alaska Airlines was again in the national spotlight \nover the weekend, this time with a "Saturday Night Live" skit.\nFeaturing "Saltburn" star Jacob Elordi and "SNL" regulars Kenan Thompson and Heidi Gardner, the skit, which \naired last weekend, parodied an Alaska Airlines ad. "Our new slogan is \'Alaska Airlines: You didn\'t die, and you got \na cool story,\' " said Gardner, who played a fli......')
Here is the output you should provide:
Title: What near-disasters, 'SNL' jabs mean for Alaska 's reputation
Author: Renata Geraldo
Media Outlet: The Seattle Times
Date of Publication: January 25, 2024
Media type: Regional

Here is a second example of document provided: Document(metadata='source': 'KnowledgeBase/CompanyAnalysis/Embraer/MediaCoverage/NewsEmbraer/TUI fly lance une nouvelle destination depuis la Belgique.PDF', 'page': [1], page_content="Page 1 of 1\nTUI fly lance une nouvelle destination depuis la Belgique\nTUI fly lance une nouvelle destination depuis la Belgique\nLe Soir\njeudi 8 février 2024\nCopyright 2024 Rossel & Cie. S.A. tous droits réservés\nSection: NEWS\nLength: 156 words\nBody\n Par la rédaction\n L a compagnie aérienne TUI fly lance ce jeudi une nouvelle destination qui reliera l'aéroport d'Anvers à Oujda \n(Maroc) pendant l'été. Deux vols par semaine seront opérés, les mercredis et les dimanches, du 26 juin au 22 \nseptembre 2024.\n TUI fly répond ainsi à une demande importante de la communauté marocaine de la région d'Anvers de pouvoir se \nrendre directement à Oujda. Cette nouvelle ligne permet de rendre facilement visite aux familles et amis au Maroc. \nLes vols sont opérés avec l'Embraer E195-E2, avion moderne et plus durable, qui avait été mis en service à \nl'aéroport d'Anvers l'été dernier.\n Cette nouvelle destination porte donc à 15 le nombre de destinations desservies par TUI fly au départ d'Anvers, \navec, entre autres plusieurs destinations vers l'Espagne....
Here is the output you should provide:
Title: TUI fly lance une nouvelle destination depuis la Belgique
Author: Par la rédaction
Media Outlet: Le Soir
Date of Publication: February 8, 2024
Media type: National

Here is a third example of document provided: Document(metadata='source': 'KnowledgeBase/CompanyAnalysis/Embraer/MediaCoverage/NewsEmbraer/No Headline In Original(2).PDF', 'page': [1, 2], page_content='Page 1 of 2\nNo Headline In Original\nNo Headline In Original\nFlight International\nApril 25, 2024\nCopyright 2024 DVV Media International Ltd All Rights Reserved\nSection: IN FOCUS\nLength: 613 words\nBody\nEmbraer starts E190F flight testing\nConverted freighter makes maiden sortie, as airframer touts potential of civil cargo role for C-390 military transport\nAlfred Chua Sao Jose dos Campos\nHoward Hardee Sacramento\nEmbraer has performed the first flight of its passenger-to-freighter (P2F) conversion, with the modified E190 taking \nto the skies over Sao Jose dos Campos on 5 April.\nThe E190F – a 2010-built example first operated by Avianca El Salvador – flew for about 2h, allowing the Embraer \nteam to complete an \xadinitial evaluation of the aircraft. A second sortie followed five days later.\nThe jet will undergo further flight testing \xadbefore being delivered to US lessor Regional One, \xadEmbraer says.\nPreviously, the company had stated its intention to deliver the E190F in the second quarter of 2024.\n“We are very pleased with E190F’s and E195F’s fast progress during the testing period,” says Embraer chief \n\xadexecutive Francisco Gomes Neto.....
Here is the output you should provide:
Title: No Headline In Original
Author: Alfred Chua, Howard Hardee
Media Outlet: Flight International
Date of Publication: April 25, 2024
Media Type: Industry-specific/Trade-press

Here is a fourth example of document provided: Document(metadata='source': 'KnowledgeBase/MediaCoverageAnalytics/Philips/NewsPhilipsFull/Spectaculaire koerswinst Philips na schikking van apneuaffaire.PDF', 'page': [1, 2], page_content="Page 1 of 2\nSpectaculaire koerswinst Philips na schikking van apneuaffaire\nSpectaculaire koerswinst Philips na schikking van apneuaffaire\nHet Financieele Dagblad\n30 april 2024 dinsdag 12:00 AM GMT\nCopyright 2024 FD Mediagroep B.V. All Rights Reserved\nSection: PAGINA 3; Blz. 3\nLength: 473 words\nBody\nVervolg van pagina 1\nPhilips maakte de schikking gisterochtend bekend, gelijktijdig met de presentatie van zijn cijfers over het eerste \nkwartaal. In het akkoord erkent Philips geen schuld voor mogelijke schade bij patiënten, die vreesden dat ze \nkanker, astma en andere aandoeningen hadden opgelopen door het gebruik van de Philips-apparaten. Het \nconcern kondigde gisteren ook aan dat zijn verzekeraars ruim €0,5 mrd van de kosten zullen vergoeden.\nDe schikking betreft zowel de claims voor medische schade als voor medische controle voor patiënten. \nClaimadvocaten hadden meer dan 760 zaken aangebracht. Nog eens 60.000 patiënten hadden zich laten \nregistreren om in een later stadium aanspraak te kunnen maken op een vergoeding. Hoeveel geld afzonderlijke \npatiënten ontvangen, is nog niet duidelijk. Het is aan de advocaten om het totaalbedrag onder hun cliënten te \nverdelen. Philips heeft zich stevig verweerd tegen de schadeclaims. Volgens het bedrijf is uit onderzoek gebleken \ndat de uitstoot van schadelijke stoffen te gering was om ernstige gezondheidsschade te kunnen veroorzaken. \nPhilips stemde toch in met een schikking omdat de uitkomst van rechtszaken moeilijk te voorspellen is. In het \nAmerikaanse rechtssysteem kunnen jury's soms onverwacht grote schadevergoedingen toekennen. Eerder \nbereikte Philips al een akkoord over economische schade voor patiënten, evenals een schikkingsakkoord met de \nAmerikaanse medische toezichthouder, de FDA. Met de nieuwe, tweeledige schikking zijn de totale kosten voor het \nafhandelen van de apneuaffaire opgelopen tot €5,2 mrd, zo blijkt uit berekeningen van het FD.....
Here is the output you should provide:
Title: Spectaculaire koerswinst Philips na schikking van apneuaffaire
Author: Anonymous
Media Outlet: Het Financieele Dagblad
Date of Publication: 30 april, 2024
Media type: National

Here is a fifth example of document provided: Document(metadata='source': 'KnowledgeBase/MediaCoverageAnalytics/Swissport/NewsSwissport/Ontluisterend rapport over bagagetillen op Schiphol_ medewerkers sjouwen zich nog tien jaar een breu.PDF', 'page': [1, 2, 3], page_content="Page 1 of 3\nOntluisterend rapport over bagagetillen op Schiphol: medewerkers sjouwen zich nog tien jaar een breuk\nOntluisterend rapport over bagagetillen op Schiphol: medewerkers sjouwen \nzich nog tien jaar een breuk\nPZC.nl\n16 september 2023 zaterdag 01:00 AM GMT\nCopyright 2023 DPG Media B.V. All Rights Reserved\nLength: 2147 words\nByline: David Bremmer\nBody\nHet gaat jaren duren voordat de honderden bagagemedewerkers op Schiphol niet langer  veel te zware koffers \nmeer hoeven te tillen, zoals de Arbeidsinspectie eist. Een nieuw rapport schetst een somber beeld van de \nbagageafhandeling op Schiphol: de tijd heeft er stilgestaan. Zelf zeggen Schiphol en de zes bagageafhandelaars \nhet werk snel flink lichter te kunnen maken.De Arbeidsinspectie wil korte metten maken met het vele getil en \ngesjouw op Schiphol. Bij drukte tillen medewerkers al snel 200 koffers per u......")
Here is the output you should provide:
Title: Ontluisterend rapport over bagagetillen op Schiphol: medewerkers sjouwen zich nog tien jaar een breuk
Author: David Bremmer
Media Outlet: PZC
Date of Publication: September 16, 2023
Media type: Regional
"""

        response = chatbot.ask(prompt)
        try:
            lines = response.split("\n")
            article['title'] = lines[0].split(": ", 1)[1].strip() if len(lines) > 0 else "Untitled"
            article['author_name'] = lines[1].split(": ", 1)[1].strip() if len(lines) > 1 else "Anonymous"
            article['media_outlet'] = lines[2].split(": ", 1)[1].strip() if len(lines) > 2 else "Unknown"
            
            if len(lines) > 3:
                article['date'] = lines[3].split(": ", 1)[1].strip()
            else:
                article['date'] = 'January 1, 2024'

            if len(lines) > 4:
                article['media_type'] = lines[4].split(": ", 1)[1].strip()
            else:
                article['media_type'] = 'Unknown'
            
            article = process_article_date(article)
            
        except Exception as e:
            logging.error(f"Error extracting metadata: {str(e)}")
            # Set default values if extraction fails
            article['title'] = article.get('title', 'Untitled')
            article['author_name'] = article.get('author_name', 'Anonymous')
            article['media_outlet'] = article.get('media_outlet', 'Unknown')
            article['date'] = article.get('date', 'January 1, 2024')
            article['media_type'] = article.get('media_type', 'Unknown')
            article = process_article_date(article)
    
    return articles

def clean_articles(articles):
    cleaned_articles = []
    
    for article in articles:
        author_name = article['author_name']
        author_name = author_name.title()  # Capitalize the first letter of each word
        
        # Ensure small letters for non-proper nouns (preserve 'van', 'de', etc.)
        author_name = re.sub(r'\b(Van|De|Der)\b', lambda m: m.group(0).lower(), author_name)
        
        article['author_name'] = author_name
        cleaned_articles.append(article)
    
    return cleaned_articles

def filter_duplicates(articles, embeddings, similarity_threshold=0.92, content_similarity_threshold=0.9):
    def text_similarity(text1, text2):
        return SequenceMatcher(None, text1, text2).ratio()

    unique_articles = []
    unique_embeddings = []
    deleted_articles = []

    for idx, (article, emb) in enumerate(zip(articles, embeddings)):
        is_duplicate = False
        for unique_idx, unique_emb in enumerate(unique_embeddings):
            embedding_similarity = 1 - cosine(emb, unique_emb)
            
            if embedding_similarity > similarity_threshold:
                content_similarity = text_similarity(article['content'], unique_articles[unique_idx]['content'])
                
                if content_similarity > content_similarity_threshold:
                    is_duplicate = True
                    deleted_articles.append((article, unique_articles[unique_idx]))
                    break

        if not is_duplicate:
            unique_articles.append(article)
            unique_embeddings.append(emb)

    print("Deleted Articles:")
    for deleted, kept in deleted_articles:
        print(f"Deleted: {deleted['file_path']} (Duplicate of {kept['file_path']})")

    return unique_articles

# Additional helper function for embeddings
def get_embeddings(articles, embedding_model):
    embeddings = []
    for article in articles:
        embeddings.append(embedding_model.encode(article['content'], normalize_embeddings=False))
    return embeddings

# Initialize the embedding model (you might want to do this outside the function in a global scope)
embeddings_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def filter_relevant_articles(articles, company_name, industry_of_interest, region):
    filtered_articles = []
    system_prompt = f"""You are a helpful assistant. Your role is to decide whether or not a given article is relevant to a given company. If the given company is not a specific entity but a topic or industry, make sure that the article's content is directly and closely related to that topic or industry. The company or topic to be considered is the following: {company_name}."""
    model_name = "gpt-4o-mini"  # You might want to make this configurable
    temperature = 0
    max_tokens = 50

    for article in articles:
        article_content = article.get('content', '')
        
        # Only trigger the chatbot if 'relevance' is not already set
        if 'relevance' not in article or not article['relevance']:
            chatbot = ChatGPT(
                system_prompt=system_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            question = f"""
You will be provided with a news media article and a company, topic or industry name. Here is the company or industry of interest: {company_name}. It operates in the {industry_of_interest}, with a prefered focus on the {region} market. You should only respond to the following question: Is the article relevant? Your output should only be: "Yes" or "No". Nothing else. Your task is to determine if the article is sufficiently relevant to {company_name} based on the following criterion:

Centrality: The article is considered relevant if {company_name} is central to the article's content, meaning the focus of the conversation revolves around {company_name}.

If {company_name} is only mentioned a few times without being a primary focus, the article should be considered not relevant.

Here is the article you will be evaluating: {article_content}

Your response should be based on whether the article meets the relevance criteria for {company_name}.
Your output should only be "Yes" or "No", based on your assessment. Nothing else.
            """
            response = chatbot.ask(question)
            print(response)
            article['relevance'] = response
        else:
            logging.info("Relevance already defined for article, skipping chatbot evaluation.")

        if article['relevance'] != 'No':
            filtered_articles.append(article)

    return filtered_articles

def filter_top_categories(posts, keep_percentage=90):
    category_counts = collections.Counter(post['category'] for post in posts)
    total_posts = sum(category_counts.values())
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    cumulative_percentage = 0
    keep_categories = set()
    
    for category, count in sorted_categories:
        if count < 2:
            continue
        cumulative_percentage += (count / total_posts) * 100
        keep_categories.add(category)
        if cumulative_percentage >= keep_percentage:
            break
    
    filtered_posts = [post for post in posts if post['category'] in keep_categories]
    return filtered_posts, keep_categories

def extract_sentiment_score(response):
    try:
        match = re.search(r'(-?[0-5])', response.strip())
        if match:
            return int(match.group(1))
        else:
            raise ValueError("Invalid sentiment score format.")
    except Exception as e:
        print(f"Error extracting sentiment score: {e}")
        return None

def save_plot_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_bar_chart_compiled_insights(data: dict, title: str, xlabel: str, ylabel: str, rotate_labels: bool = False, 
                    figsize: tuple = (8, 4), color: str = '#1f77b4') -> str:
    """
    Create a bar chart with improved styling and compact layout.
    
    Args:
        data (dict): Dictionary of labels and values
        title (str): Chart title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        rotate_labels (bool): Whether to rotate x-axis labels
        figsize (tuple): Figure size in inches
        color (str): Bar color
        
    Returns:
        str: Base64 encoded image
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars
    bars = ax.bar(range(len(data)), list(data.values()), 
                 color=color, alpha=0.7,
                 edgecolor=color, linewidth=1)
    
    # Customize title and labels
    ax.set_title(title, fontsize=12, pad=15, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    
    # Customize x-axis labels
    if rotate_labels:
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(list(data.keys()), rotation=45, ha='right', fontsize=9)
    else:
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(list(data.keys()), fontsize=9)
    
    # Customize y-axis
    ax.yaxis.set_tick_params(labelsize=9)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # Customize grid and spines
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_sentiment_graph(df, media_outlet, ax):
    df_outlet = df[df['media_outlet'] == media_outlet].sort_values('date')
    df_outlet['moving_avg'] = df_outlet['sentiment score'].rolling(window=30, min_periods=1).mean()
    
    ax.plot(df_outlet['date'], df_outlet['moving_avg'], linewidth=2, color='#1f77b4')
    ax.fill_between(df_outlet['date'], df_outlet['moving_avg'], alpha=0.3, color='#1f77b4')
    
    ax.set_title(media_outlet, fontsize=12, pad=10)
    ax.set_xlabel('Date', fontsize=10, labelpad=5)
    ax.set_ylabel('Sentiment Score', fontsize=10, labelpad=5)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(['30-Day Moving Average'], loc='upper left', fontsize=8)

def create_category_sentiment_graph(df, category, ax):
    df_category = df[df['category'] == category].sort_values('date')
    if df_category.empty:
        ax.text(0.5, 0.5, f"No data for {category}", ha='center', va='center')
    else:
        df_category['moving_avg'] = df_category['sentiment score'].rolling(window=30, min_periods=1).mean()
        
        ax.plot(df_category['date'], df_category['moving_avg'], linewidth=2, color='#1f77b4')
        ax.fill_between(df_category['date'], df_category['moving_avg'], alpha=0.3, color='#1f77b4')
        
        ax.set_title(f'Sentiment Scores: {category}', fontsize=12, pad=10)
        ax.set_xlabel('Date', fontsize=10, labelpad=5)
        ax.set_ylabel('Sentiment Score', fontsize=10, labelpad=5)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        y_min, y_max = df_category['moving_avg'].min(), df_category['moving_avg'].max()
        if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        else:
            ax.set_ylim(-5, 5)  # Set a default range
        
        min_date, max_date = df_category['date'].min(), df_category['date'].max()
        if not pd.isnull(min_date) and not pd.isnull(max_date):
            ax.annotate(f'Start: {min_date.strftime("%b %Y")}', xy=(min_date, df_category.loc[df_category['date'] == min_date, 'moving_avg'].values[0]),
                        xytext=(10, 10), textcoords='offset points', ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=8)
            ax.annotate(f'End: {max_date.strftime("%b %Y")}', xy=(max_date, df_category.loc[df_category['date'] == max_date, 'moving_avg'].values[0]),
                        xytext=(-10, 10), textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=8)
        
        ax.legend(['30-Day Moving Average'], loc='upper left', fontsize=8)

def create_horizontal_bar_chart(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(data.index, data.values, color='skyblue', edgecolor='navy')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(data.values):
        ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig, ax

def create_stacked_bar_chart(data, title, xlabel, ylabel):
    total_counts = data.sum(axis=1)
    data_sorted = data.loc[total_counts.sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    color_map = {'Negative': 'darkred', 'Neutral': 'gray', 'Positive': 'darkgreen'}
    data_sorted.plot(kind='bar', stacked=True, ax=ax, color=[color_map.get(tone, 'blue') for tone in data_sorted.columns])
    
    percentages = data_sorted.div(data_sorted.sum(axis=1), axis=0) * 100
    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if height > 0:
                percentage = percentages.iloc[j, i]
                color = bar.get_facecolor()
                text_color = 'white' if sum(color[:3]) < 1.5 else 'black'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, 
                        f'{percentage:.1f}%', ha='center', va='center', 
                        color=text_color, fontsize=9, fontweight='bold')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Tone')
    plt.tight_layout()
    return fig, ax

def generate_media_outlet_pie_chart(df):
    """Generate pie chart showing distribution of articles across media outlets."""
    try:
        # Get value counts of media outlets
        outlet_counts = df['media_outlet'].value_counts()
        
        # Create professional pie chart using your existing function
        fig = create_professional_pie(
            data=outlet_counts,
            title="Distribution of Articles by Media Outlet",
            figsize=(16,10)
        )
        
        # Convert to base64 for markdown embedding
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        logging.error(f"Error generating media outlet pie chart: {str(e)}")
        return None

def generate_media_outlet_tone_chart(df):
    tone_media_counts = df.groupby(['media_outlet', 'tone']).size().unstack(fill_value=0)
    fig, ax = create_stacked_bar_chart(tone_media_counts, 'Number of Articles per Media Outlet (Divided by Tone)', 'Media Outlet', 'Number of Articles')
    chart = save_plot_base64()
    plt.close()
    return chart

def generate_overall_sentiment_trend(df, company_name):
    plt.figure(figsize=(16, 7))
    df['moving_avg'] = df['sentiment score'].rolling(window=30).mean()
    plt.plot(df['date'], df['moving_avg'], linewidth=2, color='#1f77b4')
    plt.fill_between(df['date'], df['moving_avg'], alpha=0.3, color='#1f77b4')
    plt.title(f'30-Day Moving Average of Sentiment Scores for {company_name}', fontsize=20, pad=20)
    plt.xlabel('Date', fontsize=14, labelpad=10)
    plt.ylabel('Sentiment Score', fontsize=14, labelpad=10)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['30-Day Moving Average'], loc='upper left')
    plt.tight_layout()
    chart = save_plot_base64()
    plt.close()
    return chart

def generate_sentiment_trends_by_category(df):
    categories = df['category'].value_counts()
    eligible_categories = categories[categories > 15].head(9).index
    num_categories = len(eligible_categories)
    rows = (num_categories + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=(20, 5*rows))
    axs = axs.flatten()
    for i, category in enumerate(eligible_categories):
        create_category_sentiment_graph(df, category, axs[i])
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    chart = save_plot_base64()
    plt.close()
    return chart

def generate_articles_per_category(df):
    category_counts = df['category'].value_counts()
    fig, ax = create_horizontal_bar_chart(category_counts, 'Number of Articles per Category', 'Number of Articles', 'Category')
    chart = save_plot_base64()
    plt.close()
    return chart

def generate_category_tone_chart(df):
    # Create tone category counts
    tone_category_counts = df.groupby(['category', 'tone']).size().unstack(fill_value=0)
    
    # Format category labels with line breaks
    new_index = []
    for label in tone_category_counts.index:
        words = label.split()
        mid = len(words) // 2
        new_label = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        new_index.append(new_label)
    tone_category_counts.index = new_index
    
    # Create the chart using the helper function
    fig, ax = create_stacked_bar_chart(tone_category_counts, 
                                     'Number of Articles per Category (Divided by Tone)', 
                                     'Category', 
                                     'Number of Articles')
    chart = save_plot_base64()
    plt.close()
    return chart

def generate_top_journalists_chart(df, company_name):
    top_authors = df[df['author_name'] != 'Anonymous']
    author_counts = top_authors['author_name'].value_counts().head(10)
    author_sentiments = top_authors.groupby('author_name')['sentiment score'].mean()
    author_outlets = top_authors.groupby('author_name')['media_outlet'].apply(lambda x: ', '.join(x.unique()))

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(author_counts.index, author_counts.values, color='skyblue', edgecolor='navy')
    ax.set_title(f'Top 10 Journalists Writing on {company_name}', fontsize=16)
    ax.set_xlabel('Journalist', fontsize=12)
    ax.set_ylabel('Number of Articles', fontsize=12)

    ax.set_xticks(range(len(author_counts.index)))
    ax.set_xticklabels(author_counts.index, rotation=45, ha='right', fontsize=10)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom')

    plt.tight_layout()
    chart = save_plot_base64()
    plt.close()
    return chart

def read_insights_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""

def generate_toc(content, max_level=2):
    """
    Generates a Table of Contents (TOC) for markdown content.

    Parameters:
    content (str): The markdown content.
    max_level (int): The maximum heading level to include in the TOC (default: 6).

    Returns:
    str: The generated TOC as a string.
    """
    headings = re.findall(r'^(#{1,6}) (.+)$', content, re.MULTILINE)
    toc = "## Table of Contents\n\n"

    for heading_marks, heading_text in headings:
        level = len(heading_marks)
        if level <= max_level:  # Only include headings up to max_level
            link = re.sub(r'\W+', '-', heading_text.lower()).strip('-')
            toc += f"{'  ' * (level - 1)}- [{heading_text}](#{link})\n"
    
    return toc

def extract_categories(articles_sorted: List[Dict], company_name: str, industry_of_interest: str, region: str) -> List[Dict]:
    """
    Extract categories from articles and assign them to each article.
    
    Args:
        articles_sorted (List[Dict]): List of preprocessed articles
        company_name (str): Name of the company being analyzed
        industry_of_interest (str): Industry sector being analyzed
        region (str): Geographic region of interest
        
    Returns:
        List[Dict]: Updated articles list with categories assigned
    """
    try:
        logging.info("Starting category extraction process")
        
        # Generate one-sentence descriptions
        compiled_sentences = ""
        system_prompt = """You are a helpful assistant. Your role is to describe in one single sentence what a given news media article says about a company. The final goal of this exercise is to be able to extract general themes and topics from the article. The one sentence you have to write should be focussed on a given company."""
        
        for article in articles_sorted:
            article_content = article.get('content', '')
            chatbot = ChatGPT(
                system_prompt=system_prompt,
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=350,
            )

            question = f"""
Please write a single sentence about the content of the news article. The one sentence description should highlight in which regards does the article relate to {company_name}. Your output should only consist of that one sentence.
This one sentence should highlight the main topic or theme of the article from the perspective of {company_name}. We are interested about what is said on {company_name} in the article and on this overall topic or industry: {industry_of_interest} in the {region} market.

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

Clarity: Each category should have a clear focus, reflecting distinct aspects of the media coverage related to {company_name}. Secondary focus should be on the general topic or industry: {industry_of_interest} in the {region} market.

Output Format: List the categories in a bullet-point format with a brief description (1-2 sentences) explaining each category. do not produce a numbered list but just a bullet point list starting with "-" symbol.

Here is the compiled_sentences document: {compiled_sentences}

Be sure to focus on key themes present in the document and avoid redundant or overly broad topics. The fewer the number of categories, the better, as long as they are distinct and cover the main aspects of the media coverage.
Avoid defining categories that are too semantically similar or overlapping. For instance, "Financial Performance" and "Economic Growth" are too closely related to be separate categories. For example, Staffing Shortages, Labor Relations, Working Conditions and Recruitment Challenges are too closely related too and should be grouped under a single category like "Human Resources Issues".
        """

        response = chatbot.ask(question)
        category_titles = [
            re.sub(r'^\d+\.\s*', '', title.strip())
            for title in re.findall(r'\*\*(.*?)\*\*', response)
        ]

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

        logging.info("Category extraction completed successfully")
        return articles_sorted

    except Exception as e:
        logging.error(f"Error in category extraction: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def generate_markdown_report(company_name, total_articles, date_range, avg_sentiment, median_sentiment,
                             media_outlet_pie_chart, top_journalists_chart, media_outlet_tone_chart,
                             overall_sentiment_trend, media_outlet_stats,
                             articles_per_category, category_tone_chart, sentiment_trends_by_category, df, general_folder, language='English'):
    # Define translatable strings
    pie_chart_text = "The pie chart below shows the distribution of articles across different media outlets."
    org_chart_text = "The following chart shows the top 10 most frequently mentioned organizations in the coverage, with their tone distribution:"
    org_analysis_header = "Coverage Analysis of Key Organizations"
    people_chart_text = "The following chart shows the top 10 most frequently mentioned people in the coverage, with their tone distribution:"
    people_analysis_header = "Coverage Analysis of Key People"
    
    # Translate if needed and remove any '#' characters
    if language.lower() != "english":
        pie_chart_text = translate_content(pie_chart_text, "English", language).replace("#", "")
        org_chart_text = translate_content(org_chart_text, "English", language).replace("#", "")
        org_analysis_header = translate_content(org_analysis_header, "English", language).replace("#", "")
        people_chart_text = translate_content(people_chart_text, "English", language).replace("#", "")
        people_analysis_header = translate_content(people_analysis_header, "English", language).replace("#", "")


    # Start building the markdown content
    markdown_content = f"""
# {company_name} - Media Analytics Report

## Table of Contents
1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Analysis of Coverage Peaks](#analysis-of-coverage-peaks)
4. [Proportion of Articles by Media Outlet](#proportion-of-articles-by-media-outlet)
5. [Top Journalists](#top-journalists)
6. [Media Outlet Statistics](#media-outlet-statistics)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Category Analysis](#category-analysis)
9. [Analysis of Most Discussed Organizations](#analysis-of-most-discussed-organizations)
10. [Analysis of Most Discussed People](#analysis-of-most-discussed-people)

## Data Overview
- Total number of articles: {total_articles}
- Date range: {date_range}
- Average sentiment score: {avg_sentiment:.2f}
- Median sentiment score: {median_sentiment:.2f}
"""
    # Insert Publication Timeline Section (including ChatGPT peak analysis)
    publication_timeline_section = generate_publication_timeline_section(df, company_name, language)
    markdown_content += publication_timeline_section

    markdown_content += f"""
## Proportion of Articles by Media Outlet
{pie_chart_text}

![Proportion of Articles by Media Outlet](data:image/png;base64,{media_outlet_pie_chart})

## Top Journalists
![Top 10 Journalists](data:image/png;base64,{top_journalists_chart})

### Top 10 Journalists and Their Media Outlets
| Journalist | Media Outlet(s) | Number of Articles | Average Sentiment |
|------------|----------------|-------------------|-------------------|
"""

    top_authors = df[df['author_name'] != 'Anonymous']
    author_counts = top_authors['author_name'].value_counts().head(10)
    author_sentiments = top_authors.groupby('author_name')['sentiment score'].mean()
    author_outlets = top_authors.groupby('author_name')['media_outlet'].apply(lambda x: ', '.join(x.unique()))

    for author, count in author_counts.items():
        avg_sentiment_author = author_sentiments[author]
        markdown_content += f"| {author} | {author_outlets[author]} | {count} | {avg_sentiment_author:.2f} |\n"

    journalist_analysis_markdown = generate_top_journalists_analysis(df, company_name)
    if language.lower() != "english":
         journalist_analysis_markdown = translate_content(journalist_analysis_markdown, "English", language)
    markdown_content += journalist_analysis_markdown

    markdown_content += f"""

### Articles per Media Outlet (Divided by Tone)
![Articles per Media Outlet (Divided by Tone)](data:image/png;base64,{media_outlet_tone_chart})

## Media Outlet Statistics
| Media Outlet | Number of Articles | Average Sentiment | Median Sentiment |
|--------------|---------------------|-------------------|-------------------|
"""

    for stat in media_outlet_stats:
        markdown_content += f"| {stat['outlet']} | {stat['articles']} | {stat['avg_sentiment']:.2f} | {stat['median_sentiment']:.2f} |\n"

    markdown_content += f"""
## Sentiment Analysis
![Overall Sentiment Trend](data:image/png;base64,{overall_sentiment_trend})
"""
    # Add sentiment evolution analysis
    sentiment_analysis_content = generate_sentiment_analysis_section(df, company_name)
    if language.lower() != "english":
         sentiment_analysis_content = translate_content(sentiment_analysis_content, "English", language)
    markdown_content += sentiment_analysis_content

    markdown_content += f"""
## Category Analysis
### Number of Articles per Category
![Number of Articles per Category](data:image/png;base64,{articles_per_category})

### Articles per Category (Divided by Tone)
![Articles per Category (Divided by Tone)](data:image/png;base64,{category_tone_chart})

### Sentiment Trends by Category
![Sentiment Trends by Category](data:image/png;base64,{sentiment_trends_by_category})
"""
    
        # Process and organize the dataframe
    articles_sorted = df.sort_values(by='date')
    articles_list = articles_sorted.to_dict('records')

    # Function to check if an article needs entities extracted
    def needs_entity_extraction(article):
        return (
            'organizations' not in article or 
            not article['organizations'] or 
            'people' not in article or 
            not article['people']
        )

    # Function to check if an article needs sentiment analysis
    def needs_sentiment_analysis(article):
        # Check organizations
        if 'organizations' in article:
            for org in article['organizations']:
                if 'sentiment_score' not in org or 'tone' not in org:
                    return True
        
        # Check people
        if 'people' in article:
            for person in article['people']:
                if 'sentiment_score' not in person or 'tone' not in person:
                    return True
        
        return False

    # Check which articles need processing
    articles_needing_entities = []
    articles_needing_sentiment = []
    articles_ready = []

    for article in articles_list:
        if needs_entity_extraction(article):
            articles_needing_entities.append(article)
        elif needs_sentiment_analysis(article):
            articles_needing_sentiment.append(article)
        else:
            articles_ready.append(article)

    # Process articles that need entities
    if articles_needing_entities:
        processed_with_entities = extract_entities(articles_needing_entities, company_name, general_folder)
        # Add these to articles needing sentiment analysis
        articles_needing_sentiment.extend(processed_with_entities)

    # Process articles that need sentiment analysis
    if articles_needing_sentiment:
        processed_with_sentiment = analyze_all_sentiments(articles_needing_sentiment, company_name, general_folder)
        # Add these to ready articles
        articles_ready.extend(processed_with_sentiment)

    # At this point, all articles have both entities and sentiments
    processed_articles = sorted(articles_ready, key=lambda x: x.get('timestamp', 0))

    # Process organization data
    org_data = []
    for article in processed_articles:
        for org in article.get('organizations', []):
            org_data.append({
                'name': org['name'],
                'type': org['type'],
                'tone': org['tone'],
                'sentiment_score': org.get('sentiment_score', 0),
                'description': org['description'],
                'article_content': article['content']
            })

    # Create a DataFrame and get the top 10 organizations by mention frequency
    org_df = pd.DataFrame(org_data)
    top_orgs = org_df['name'].value_counts().head(10)

    # Process people data
    people_data = []
    for article in processed_articles:
        for person in article.get('people', []):
            people_data.append({
                'name': person['name'],
                'role': person['role'],
                'tone': person['tone'],
                'sentiment_score': person.get('sentiment_score', 0),
                'context': person['context'],
                'article_content': article['content']
            })

    # Create a DataFrame and get the top 10 people by mention frequency
    people_df = pd.DataFrame(people_data)
    top_people = people_df['name'].value_counts().head(10)

    # Build tone count DataFrame for organizations
    org_tone_df = pd.DataFrame([
        org_df[org_df['name'] == org]['tone'].value_counts()
        for org in top_orgs.index
    ], index=top_orgs.index).fillna(0)

    fig, ax = create_stacked_bar_chart(
        data=org_tone_df,
        title='Top 10 Most Mentioned Organizations with Tone Distribution',
        xlabel='Organization',
        ylabel='Number of Mentions'
    )
    org_chart = save_plot_base64()
    plt.close()

    # Build tone count DataFrame for people
    people_tone_df = pd.DataFrame([
        people_df[people_df['name'] == person]['tone'].value_counts()
        for person in top_people.index
    ], index=top_people.index).fillna(0)

    fig, ax = create_stacked_bar_chart(
        data=people_tone_df,
        title='Top 10 Most Mentioned People with Tone Distribution',
        xlabel='Person',
        ylabel='Number of Mentions'
    )
    people_chart = save_plot_base64()
    plt.close()

    # --- REPLACE THIS BLOCK: Organization Analysis ---
    # For graphing purposes, you already built org_tone_df from top_orgs (top 10).
    # For analysis, use the top 3 organizations.
    top_orgs_analysis_series = org_df['name'].value_counts().head(3)
    top_orgs_data = []
    for org_name in top_orgs_analysis_series.index:
        org_articles = org_df[org_df['name'] == org_name]
        # Limit to 25 articles per organization
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
    
    # Use one chatbot call for the top 3 organization analysis
    org_top_prompt = f"""
Analyze how the following top 3 organizations are portrayed in the media coverage.
Focus on key narratives, sentiment patterns, and overall portrayal.
Data:
{json.dumps(top_orgs_data, indent=2)}
Provide your analysis in a single, concise paragraph.
    """
    
    chatbot = ChatGPT(
        system_prompt="You are a media analysis expert specializing in news coverage analysis.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500
    )
    top_orgs_analysis = chatbot.ask(org_top_prompt)
    
    # Extreme Sentiment Analysis for organizations (for those outside the top 3)
    remaining_orgs = org_df[org_df['name'].isin(org_df['name'].value_counts().head(10).index.difference(top_orgs_analysis_series.index))]
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
    
    positive_extreme_prompt = f"""
Analyze how the following three organizations (with the most positive sentiment) are portrayed in the media.
Data:
{json.dumps(extreme_positive_data, indent=2)}
Provide your analysis in a single, concise paragraph.
    """
    negative_extreme_prompt = f"""
Analyze how the following three organizations (with the most negative sentiment) are portrayed in the media.
Data:
{json.dumps(extreme_negative_data, indent=2)}
Provide your analysis in a single, concise paragraph.
    """
    
    positive_extreme_chatbot = ChatGPT(
        system_prompt="You are a media analysis expert focusing on extreme positive portrayals.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500
    )
    positive_extreme_analysis = positive_extreme_chatbot.ask(positive_extreme_prompt)
    
    negative_extreme_chatbot = ChatGPT(
        system_prompt="You are a media analysis expert focusing on extreme negative portrayals.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500
    )
    negative_extreme_analysis = negative_extreme_chatbot.ask(negative_extreme_prompt)
    
    org_analysis = f"{top_orgs_analysis}\n\n{positive_extreme_analysis}\n\n{negative_extreme_analysis}"

    # For graphing, you already built people_tone_df from the top 10 people.
    # For analysis, use the top 3 people.
    top_people_analysis_series = people_df['name'].value_counts().head(3)
    top_people_data = []
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
    
    people_top_prompt = f"""
Analyze how the following top 3 individuals are portrayed in the media coverage.
Focus on key narrative themes, sentiment, and overall portrayal.
Data:
{json.dumps(top_people_data, indent=2)}
Provide your analysis in a single, concise paragraph.
    """
    chatbot = ChatGPT(
        system_prompt="You are a media analysis expert specializing in news coverage analysis.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500
    )
    top_people_analysis = chatbot.ask(people_top_prompt)
    
    # Extreme Sentiment Analysis for people
    remaining_people = people_df[people_df['name'].isin(people_df['name'].value_counts().head(10).index.difference(top_people_analysis_series.index))]
    people_sentiments = remaining_people.groupby('name')['sentiment_score'].mean().to_dict()
    sorted_negative = sorted(people_sentiments.items(), key=lambda x: x[1])
    most_negative_people = sorted_negative[:3]
    sorted_positive = sorted(people_sentiments.items(), key=lambda x: x[1], reverse=True)
    most_positive_people = sorted_positive[:3]
    
    extreme_negative_people_data = []
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
        extreme_negative_people_data.append(person_info)
    
    extreme_positive_people_data = []
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
        extreme_positive_people_data.append(person_info)
    
    positive_extreme_people_prompt = f"""
Analyze how the following three individuals (with the most positive sentiment) are portrayed in the media.
Data:
{json.dumps(extreme_positive_people_data, indent=2)}
Provide your analysis in a single, concise paragraph.
    """
    negative_extreme_people_prompt = f"""
Analyze how the following three individuals (with the most negative sentiment) are portrayed in the media.
Data:
{json.dumps(extreme_negative_people_data, indent=2)}
Provide your analysis in a single, concise paragraph.
    """
    
    positive_extreme_people_chatbot = ChatGPT(
        system_prompt="You are a media analysis expert focusing on extreme positive portrayals.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500
    )
    positive_extreme_people_analysis = positive_extreme_people_chatbot.ask(positive_extreme_people_prompt)
    
    negative_extreme_people_chatbot = ChatGPT(
        system_prompt="You are a media analysis expert focusing on extreme negative portrayals.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500
    )
    negative_extreme_people_analysis = negative_extreme_people_chatbot.ask(negative_extreme_people_prompt)
    
    people_analysis = f"{top_people_analysis}\n\n{positive_extreme_people_analysis}\n\n{negative_extreme_people_analysis}"

    if language.lower() != "english":
         people_analysis = translate_content(people_analysis, "auto", language)
         org_analysis = translate_content(org_analysis, "auto", language)

    # Add analyses to markdown content
    markdown_content += f"""\n
## Analysis of Most Discussed Organizations

{org_chart_text}

![Top Organizations](data:image/png;base64,{org_chart})

### {org_analysis_header}
{org_analysis}

## Analysis of Most Discussed People

{people_chart_text}

![Top People](data:image/png;base64,{people_chart})

### {people_analysis_header}
{people_analysis}
\n"""

    return markdown_content


def process_stakeholder_info(company_name, articles):
    """
    Extracts stakeholder information from a collection of articles.
    
    Args:
        company_name (str): Name of the company being analyzed
        articles_sorted (list): List of processed articles
        
    Returns:
        str: Markdown table containing stakeholder information
    """
    summary_md = f"""
# Stakeholder quotes retrieved from Media Coverage - {company_name}
| Stakeholder Name | Role | Stakeholder quote related to {company_name} | Translation | Sentiment |
| --- | --- | --- | --- | --- | --- |
"""
    
    for article in articles:
        article_link = article.get('link', '#')
        
        # First chatbot: Extract quotes
        quote_extractor = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=800,
        )       
        quotes_response = quote_extractor.ask(f"""
You are tasked with extracting stakeholder information and direct quotes from a given news article related to {company_name}. Your output should be in the form of a markdown table without column titles or any additional information. Each table row should contain:
1. The name of a real person who has expressed a quote in the provided article
2. His role or relation with regards to {company_name}.
3. Their original quote about {company_name} exactly as it appears in the article, without any modifications or translations

Guidelines:
- Include only real people: Exclude organizations, companies, or any other non-person entities
- Direct connection: Ensure that each selected individual is explicitly connected to {company_name} or explicitly express his opinion on {company_name}.
- Only include those whose statements are directly about {company_name}
- Exclude individuals whose comments are not directly relevant to {company_name} or are too general
- Pertinent stakeholders only: Only include the most relevant individuals
- Do not include stakeholders who are not directly associated with {company_name}

Output Format Requirements:
- Each line must contain exactly two columns separated by "|" symbols
- Column 1: Stakeholder's name
- Column 2: Stakeholder's role or link to {company_name}. If not described in the article, try to interpret its role. If it is not straightforward to interpret, mention "Not mentioned".
- Column 3: Original quote in its original language

Example of correct formatting:
| Pierre Dupont | Consumer | "Cette décision de {company_name} est très importante." |
| John Smith | {company_name}'s CEO |"We need to review {company_name}'s proposal carefully." |
| Hans Weber | Financial analyst | "Die Strategie von {company_name} ist überzeugend." |

If no relevant individuals with quotes are found, return "None."

Article:
{article['content']}
        """)
        
        print("Quotes extraction response:", quotes_response)
        
        # Second chatbot: Add translations
        translator = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=1200,
        )
        
        translation_response = translator.ask(f"""
You are tasked with adding English translations to a set of stakeholder quotes. For each quote:
- If the quote is in English, add "N.A." as the translation
- If the quote is in any other language, provide an accurate English translation

Input format is a markdown table with stakeholder names and quotes.
Output should be a markdown table with three columns: name, original quote, and translation.

IMPORTANT RULES:
- Never translate English quotes to English - use "N.A." instead.
- Only and always translate non-English quotes.
- If the original quote, in the third column of the md table is not in english, translate it and add the translation in the 4th column, as described below, but that is only when the quote is not in English.
- Keep original quotes exactly as they are.
- Maintain the markdown table format, without including any horizontal divider lines

Example input:
| Pierre Dupont | Consumer | "Cette décision est importante." |
| John Smith | {company_name}'s corporate lawyer | "We need to review this carefully." |

Example output:
| Pierre Dupont | Consumer | "Cette décision est importante." | "This decision is important." |
| John Smith | {company_name}'s corporate lawyer | "We need to review this carefully." | N.A. |

Example input:
| [Person Name 1] | Person Role 1 | [Original Quote in another language (e.g. Dutch, French, German, Spanish,eyc..)] |
| [person Name 2] | Person Role 2 | [Original Quote in English] |

Example output:
| [person Name 1] | Person Role 1 | [Original Quote in another language (e.g. Dutch, French, German, Spanish,etc..)] | [Translated Quote] |
| [person Name 2] | Person Role 2 | [Original Quote in English] | N.A. |

If the input is "None", return "None". if the original quote is not in English, make sure it is translated. Add "N.A" if the quote is already in English.

Input quotes:
{quotes_response}
        """)
        
        print("Translation response:", translation_response)

        chatbot2 = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=1200,
        )

        response2 = chatbot2.ask(f"""
You are tasked with analyzing the following list of stakeholders and returning only the lines that represent real people. Disregard any lines that mention entities or organizations. Remove all rows which are table dividers using "-".

Input:
{translation_response}

Guidelines:
1. Only include lines that represent real people or individuals. Only return lines that mention a specific person's name in the first column.
2. Exclude any lines that mention companies, organizations or entities.
3. If a line is ambiguous, err on the side of exclusion.
4. Exclude mentions of John Doe or Jane Smith. Exclude the author of the article or other people which are not directly related to {company_name}.
5. Maintain the original format of each line (markdown table row). Keep the whole line, including the stakeholder description. Each line must be a valid markdown table row with 4 columns, strictly.
6. If no lines represent real people, return "None".
7. Filter out too general or irrelevant stakeholders. Only include stakeholders that express direct opinions on {company_name} or are directly involved with the company.
8. Remove all rows which are only containing the "-" symbol as your output should only be the rows that represent real people.
9. Maintain the markdown table format, without including any horizontal divider lines

Please provide the filtered list of stakeholders below:
        """)
        
        print("Filtered response:", response2)

        chatbot3 = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=800,
        )

        response3 = chatbot3.ask(f"""
You are tasked with analyzing the impact and implications of stakeholder quotes for {company_name}'s reputation, business, and industry position. Your analysis should focus on how each quote affects {company_name}, not just the quote's general tone.

CRITICAL ANALYSIS GUIDELINES:
1. Business Impact Assessment
- Evaluate how the quote affects {company_name}'s:
  * Market position and competitive advantage
  * Stakeholder trust and relationships
  * Regulatory compliance and legal standing
  * Public perception and brand value

2. Context-Sensitive Analysis
- Consider the broader industry context
- Evaluate current market conditions and challenges
- Account for regulatory environment and compliance requirements
- Factor in {company_name}'s current strategic objectives

3. Stakeholder Influence Assessment
- Consider the stakeholder's role and influence
- Evaluate the potential reach and impact of their statement
- Assess how their opinion might affect other stakeholders

SENTIMENT CLASSIFICATION:
Positive (from {company_name}'s perspective):
- Statements that support or validate {company_name}'s strategy
- Comments that could enhance market position or stakeholder trust
- Quotes that defend {company_name} against criticism
- Statements highlighting {company_name}'s strengths or improvements

Negative (from {company_name}'s perspective):
- Statements that could damage {company_name}'s reputation or credibility
- Comments raising concerns about {company_name}'s practices or decisions
- Quotes that could trigger regulatory scrutiny on {company_name}
- Even positively-worded criticism (e.g., "They're making progress, but still far behind competitors")

Neutral (from {company_name}'s perspective):
- Factual statements without significant impact on {company_name}
- Balanced observations that neither help nor harm
- Technical or procedural comments without clear implications for {company_name}

FORMAT REQUIREMENTS:
- Maintain the original markdown table format
- Add sentiment classification as the final column
- Use only "Positive," "Negative," or "Neutral" in the sentiment column

Example Analysis:
Original quote: "The company is making impressive progress in sustainability."
Surface tone: Positive
Deeper analysis: Could be Negative if competitors are far ahead or if it implies previous poor performance

Example Input:
| John Weber | Expert | "The technological improvements at {company_name} are impressive, but they're still years behind industry leaders." | N.A. |
| Maria Chen | Lawyer | "{company_name} follows all regulations perfectly." | N.A. |
| Pierre Dubois | Business partner | "We are pleased to partner with {company_name} on this groundbreaking initiative." | N.A. |

Example Output:
| John Weber | Expert | "The technological improvements at {company_name} are impressive, but they're still years behind industry leaders." | N.A. | Negative |
| Maria Chen | Lawyer | "{company_name} follows all regulations perfectly." | N.A. | Neutral |
| Pierre Dubois | Business partner | "We are pleased to partner with {company_name} on this groundbreaking initiative." | N.A. | Positive |

Article Context:
{article['content']}

Quotes Table:
{response2}

Your output must contain the initial Quotes table provided with an additional column containing the appropriate sentiment classifications. You must include all the previous md table columns, even if it contains the value "N.A.", in the Quotes Table provided. Do not change anything from the provided initial Quotes Table. Your output should therefore have 4 columns. Return "None" if the Quotes Table is "None." Your response must be returned as a md table. Remove all rows which are only containing the "-" symbol , Maintain the markdown table format, without including any horizontal divider lines.
        """)
        
        print("Sentiment response:", response3)

        if response3.strip().lower() not in ["none", "none."]:
            response_lines = response3.strip().split("\n")
            for line in response_lines:
                if "none" not in line.lower():
                    # Split the line into its components
                    parts = line.split('|')
                    if len(parts) >= 4:  # Ensure we have enough columns
                        stakeholder_name = parts[1].strip()
                        # Add hyperlink to stakeholder name
                        parts[1] = f" [{stakeholder_name}]({article_link}) "
                        # Rejoin the line with the hyperlinked stakeholder name
                        hyperlinked_line = '|'.join(parts)
                        summary_md += f"{hyperlinked_line}| \n"
        
    return summary_md

def process_markdown_table(md_content):
    """
    Process a markdown table by:
    1. Sorting rows alphabetically by stakeholder name
    2. Removing duplicate or near-duplicate quotes
    
    Args:
        md_content (str): The markdown table content as a string
        
    Returns:
        str: Processed markdown table
    """
    # Split content into header and rows
    lines = md_content.strip().split('\n')
    
    # Preserve the title and header rows
    title = lines[0]
    headers = lines[1:3]  # This includes the header row and the separator row
    
    # Process only the data rows (skip headers and separator)
    data_rows = [row for row in lines[3:] if row.strip() and '|' in row]
    
    # Function to extract stakeholder name from a row
    def get_stakeholder(row):
        return row.split('|')[1].strip()
    
    # Function to extract quote from a row
    def get_quote(row):
        return row.split('|')[2].strip()
    
    # Function to calculate similarity between two quotes
    def quote_similarity(quote1, quote2):
        from difflib import SequenceMatcher
        return SequenceMatcher(None, quote1.lower(), quote2.lower()).ratio()
    
    # Remove duplicates while preserving order
    seen_quotes = {}
    unique_rows = []
    
    for row in data_rows:
        quote = get_quote(row)
        is_duplicate = False
        
        # Check against existing quotes for similarity
        for existing_quote in seen_quotes:
            if quote_similarity(quote, existing_quote) > 0.8:  # 80% similarity threshold
                is_duplicate = True
                break
                
        if not is_duplicate:
            seen_quotes[quote] = True
            unique_rows.append(row)
    
    # Sort unique rows by stakeholder name
    sorted_rows = sorted(unique_rows, key=get_stakeholder)
    
    # Reconstruct the markdown table
    processed_table = '\n'.join([title] + headers + sorted_rows)
    
    return processed_table

def translate_content(content: str, source_language: str, target_language: str) -> str:
    """
    Translate content from source language to target language using ChatGPT.
    
    Args:
        content (str): Content to translate
        source_language (str): Source language (detected automatically if not specified)
        target_language (str): Target language to translate into
        
    Returns:
        str: Translated content
    """
    try:
        # Skip translation if target language is English
        if target_language.lower() == "english":
            return content

        # Initialize translation chatbot
        translator = ChatGPT(
            system_prompt=f"""You are a professional translator specializing in {target_language} translations. 
Your task is to translate content while maintaining the original formatting, including markdown syntax, 
titles, headings, and any special characters.""",
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=5000,
        )

        # Keep the instructions separate from the content to translate
        question = f"""Translate the following text into {target_language}, maintaining all formatting exactly as is.
Here is the content to translate:
{content}"""

        translated = translator.ask(question)
        return translated

    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return content  # Return original content if translation fails

def convert_md_to_pdf(input_file: str, output_file: str = None, css_file: str = 'template/CompactCSSTemplate.css') -> str:
    """
    Convert a Markdown file to PDF using pypandoc.
    
    Args:
        input_file (str): Path to the input Markdown file
        output_file (str): Path for the output PDF file (optional)
        css_file (str): Path to the CSS template file
        
    Returns:
        str: Path to the generated PDF file
    """
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.pdf'))
        
    try:
        # Read the CSS file
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
            
        # Create a temporary CSS file with the content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.css', delete=False) as temp_css:
            temp_css.write(css_content)
            temp_css_path = temp_css.name
            
        # Configure pandoc options
        extra_args = [
            '--pdf-engine=pdflatex',
            f'--css={temp_css_path}',
            '--toc',
            '--toc-depth=3',
            '-V', 'geometry:margin=2.5cm',
            '-V', 'documentclass=article',
            '-V', 'fontsize=11pt',
            '--highlight-style=tango'
        ]
        
        # Convert markdown to PDF
        pypandoc.convert_file(
            input_file,
            'pdf',
            outputfile=output_file,
            extra_args=extra_args
        )
        
        # Clean up temporary CSS file
        Path(temp_css_path).unlink()
        
        logging.info(f"Successfully converted {input_file} to {output_file}")
        return output_file
        
    except Exception as e:
        logging.error(f"Error converting file to PDF: {str(e)}")
        raise

def create_markdown_anchor(category_name: str) -> str:
    """
    Creates a properly formatted markdown anchor from a category name.
    
    Args:
        category_name (str): The name of the category
        
    Returns:
        str: A properly formatted markdown anchor
    """
    # Convert to lowercase
    anchor = category_name.lower()
    # Replace spaces and special characters with hyphens
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    # Replace spaces with hyphens
    anchor = re.sub(r'\s+', '-', anchor)
    # Remove any duplicate hyphens
    anchor = re.sub(r'-+', '-', anchor)
    # Remove leading/trailing hyphens
    anchor = anchor.strip('-')
    
    return anchor

def find_extrema_points(df, min_days_separation=45):
    """
    Find local minima and maxima in the sentiment moving average with minimum separation,
    excluding the first 60 days of data.
    
    Args:
        df: DataFrame with 'date' and 'sentiment score' columns
        min_days_separation: Minimum days between extrema points
    
    Returns:
        tuple: Lists of (maxima_dates, maxima_values), (minima_dates, minima_values)
    """
    
    # Calculate 30-day moving average if not already present
    if 'moving_avg' not in df.columns:
        df = df.copy()
        df['moving_avg'] = df['sentiment score'].rolling(window=30).mean()
    
    # Sort by date and exclude first 60 days
    df = df.sort_values('date')
    start_date = df['date'].min()
    exclusion_date = start_date + pd.Timedelta(days=60)
    df_filtered = df[df['date'] > exclusion_date].copy()
    
    # Convert moving average to numpy array, handling NaN values
    values = df_filtered['moving_avg'].fillna(method='bfill').fillna(method='ffill').values
    
    # Find peaks (maxima) and valleys (minima)
    min_separation = min_days_separation  # minimum separation in data points
    maxima_idx, _ = find_peaks(values, distance=min_separation)
    minima_idx, _ = find_peaks(-values, distance=min_separation)
    
    # Get corresponding dates and values
    maxima_dates = df_filtered['date'].iloc[maxima_idx]
    maxima_values = values[maxima_idx]
    minima_dates = df_filtered['date'].iloc[minima_idx]
    minima_values = values[minima_idx]
    
    # Sort by value to get top extrema
    maxima = list(zip(maxima_dates, maxima_values))
    minima = list(zip(minima_dates, minima_values))
    
    maxima.sort(key=lambda x: x[1], reverse=True)
    minima.sort(key=lambda x: x[1])
    
    return maxima[:2], minima[:2]  # Return top 2 maxima and minima

def analyze_sentiment_period(df, start_date, end_date, company_name, is_peak=True):
    """
    Analyze articles and sentiment for a specific period with content size limits.
    
    Args:
        df: DataFrame with articles data
        start_date: Period start date
        end_date: Period end date
        company_name: Name of the company
        is_peak: Boolean indicating if this is a peak (True) or dip (False)
    
    Returns:
        str: Analysis of the period
    """
    # Filter articles for the period
    mask = (df['date'] >= start_date - pd.Timedelta(days=40)) & (df['date'] <= end_date)
    period_df = df[mask].copy()
    
    # Prepare article information with content limits
    articles_info = []
    total_chars = 0
    max_article_chars = 5000
    max_total_chars = 500000
    
    for _, row in period_df.iterrows():
        # Skip if we've exceeded total character limit
        if total_chars >= max_total_chars:
            break
            
        # Truncate content if needed
        content = row.get('content', '')
        if content:
            content = content[:max_article_chars]
        
        # Update total character count
        total_chars += len(content)
        
        article_info = {
            'date': row['date'].strftime('%Y-%m-%d'),
            'title': row.get('title', 'No title'),
            'sentiment': row['sentiment score'],
            'content': content,
            'media_outlet': row.get('media_outlet', 'Unknown')
        }
        articles_info.append(article_info)
    
    # Create prompt for the chatbot
    period_type = "peak" if is_peak else "dip"
    period_sign = "positive" if is_peak else "negative"
    chatbot = ChatGPT(
        system_prompt=f"""You are an expert media analyst focused on analyzing sentiment changes in media coverage. 
        Your task is to explain a significant {period_type} in sentiment regarding {company_name} by analyzing relevant articles from the period. Since the {period_type} represents a {period_sign} sentiment shift, I expect you retrieve and analyse articles that contributed to this {period_sign} sentiment change.
        Provide a concise one-paragraph analysis highlighting the key events or narratives that drove the sentiment trend.""",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=2000
    )
    
    question = f"""
    Analyze the media coverage of {company_name} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}, 
    which represents a sentiment {period_type}. Based on the provided articles, explain what events or narratives drove this trend.
    Focus on key stories, key events narrated in the media about {company_name}, which could explain the {period_type} trend, and their impact on public perception. Since we are investigating the reasons for a sentiment {period_type}, we want to observe which {period_sign} events contributed to that trend. Provide tangible facts and examples from the coverage. Reference specific media outlets and dates where relevant.
    
    Articles for analysis:
    {json.dumps(articles_info, indent=2)}
    
    Provide a comprehensive but not too long paragraph analysis that explains the sentiment trend during this period. Include tangible examples or facts from the coverage. Cite your sources into brackets: (media outlet, date)
    """
    
    try:
        response = chatbot.ask(question)
        return response if response else "Analysis could not be generated for this period."
    except Exception as e:
        print(f"Error analyzing period {start_date} to {end_date}: {str(e)}")
        return f"Error analyzing period: {str(e)}"

def generate_sentiment_analysis_section(df, company_name):
    """
    Generate comprehensive sentiment analysis section for the report.
    """
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate moving average
    df = df.copy()
    df['moving_avg'] = df['sentiment score'].rolling(window=30).mean()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Find extrema points
    maxima, minima = find_extrema_points(df)
    print(maxima, minima)
    
    # Generate analysis for each period
    analyses = []
    
    # Analyze periods around maxima
    for peak_date, peak_value in maxima:
        analysis = analyze_sentiment_period(
            df,
            peak_date - pd.Timedelta(days=40),
            peak_date,
            company_name,
            is_peak=True
        )
        analyses.append((peak_date, "Peak", analysis))
        print(analysis)
    
    # Analyze periods around minima
    for dip_date, dip_value in minima:
        analysis = analyze_sentiment_period(
            df,
            dip_date - pd.Timedelta(days=40),
            dip_date,
            company_name,
            is_peak=False
        )
        analyses.append((dip_date, "Dip", analysis))
    
    # Generate overall analysis
    chatbot = ChatGPT(
        system_prompt=f"""You are an expert media analyst. Summarize the overall sentiment trajectory for {company_name} 
        based on the previous period analyses.""",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1000
    )
    
    overall_analysis = chatbot.ask(f"Synthesize the following period analyses into an overall sentiment trajectory:\n{json.dumps([a[2] for a in analyses])}")
    
    # Format the sentiment analysis section
    sentiment_section = "\n### Sentiment Evolution Analysis\n\n"
    
    # Add identified extrema points
    sentiment_section += "#### Key Sentiment Points\n"
    sentiment_section += "**Peak Periods:**\n"
    for date, value in maxima:
        sentiment_section += f"- {date.strftime('%B %d, %Y')}: {value:.3f}\n"
    sentiment_section += "\n**Dip Periods:**\n"
    for date, value in minima:
        sentiment_section += f"- {date.strftime('%B %d, %Y')}: {value:.3f}\n"
    
    # Add chronological analyses
    analyses.sort(key=lambda x: x[0])
    for date, period_type, analysis in analyses:
        sentiment_section += f"\n#### {period_type} Period - {date.strftime('%B %Y')}\n{analysis}\n"
    
    sentiment_section += f"\n#### Overall Sentiment Trajectory\n{overall_analysis}\n"
    
    return sentiment_section

def find_category_extrema(df, category_name):
    """
    Find the maximum and minimum sentiment points for a specific category.
    
    Args:
        df: DataFrame with category data
        category_name: Name of the category to analyze
        
    Returns:
        tuple: (maximum_point, minimum_point) where each point is (date, value)
    """
    # Filter for the specific category
    category_df = df[df['category'] == category_name].copy()
    
    # Calculate 30-day moving average
    if 'moving_avg' not in category_df.columns:
        category_df['moving_avg'] = category_df['sentiment score'].rolling(window=30).mean()
    
    # Sort by date and exclude first 60 days
    category_df = category_df.sort_values('date')
    start_date = category_df['date'].min()
    exclusion_date = start_date + pd.Timedelta(days=60)
    df_filtered = category_df[category_df['date'] > exclusion_date].copy()
    
    # Find maximum and minimum points
    max_idx = df_filtered['moving_avg'].idxmax()
    min_idx = df_filtered['moving_avg'].idxmin()
    
    max_point = (df_filtered.loc[max_idx, 'date'], df_filtered.loc[max_idx, 'moving_avg'])
    min_point = (df_filtered.loc[min_idx, 'date'], df_filtered.loc[min_idx, 'moving_avg'])
    
    return max_point, min_point

def analyze_category_period(df, category, start_date, end_date, company_name, is_peak=True):
    """
    Analyze articles for a specific category and period.
    """
    # Filter for category and period
    mask = (
        (df['category'] == category) & 
        (df['date'] >= start_date - pd.Timedelta(days=40)) & 
        (df['date'] <= end_date)
    )
    period_df = df[mask].copy()
    
    # Prepare article information with content limits
    articles_info = []
    total_chars = 0
    max_article_chars = 5000
    max_total_chars = 500000
    
    for _, row in period_df.iterrows():
        if total_chars >= max_total_chars:
            break
            
        content = row.get('content', '')[:max_article_chars] if row.get('content') else ''
        total_chars += len(content)
        
        article_info = {
            'date': row['date'].strftime('%Y-%m-%d'),
            'title': row.get('title', 'No title'),
            'sentiment': row['sentiment score'],
            'content': content,
            'media_outlet': row.get('media_outlet', 'Unknown')
        }
        articles_info.append(article_info)
    
    # Create prompt for the chatbot
    period_type = "peak" if is_peak else "dip"
    period_sign = "positive" if is_peak else "negative"
    chatbot = ChatGPT(
        system_prompt=f"""
        You are an expert media analyst focused on analyzing sentiment changes in media coverage. 
        Your task is to explain a significant {period_type} in sentiment regarding {company_name}, specifically regarding {category}-related topics, by analyzing relevant articles from the period. Therefore, since the sentiment has been {period_sign}, you need to analyze the articles to explain what specific events or narratives within this category drove the sentiment trend.
        Explain in your output, with tanglible examples (media outlet, date) what specific events or narratives drove the {period_type}, {period_sign,} sentiment trend.
        Provide a complete but not too long one-paragraph analysis highlighting the key events or narratives that drove the sentiment trend, specifically with regards to {company_name} on {category}-related topics.
        """,
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=2000
    )
    
    question = f"""
    Analyze the media coverage of {company_name}'s {category}-related topics between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}, 
    a period of {period_type} sentiment. Based on the provided articles, explain what specific events or narratives within this category drove the sentiment trend.
    
    Articles for analysis:
    {json.dumps(articles_info, indent=2)}
    
    Provide a comprehensive but not too long paragraph analysis that explains explaining the {period_sign } sentiment trend for {category}-related coverage during this period, from the perspective of {company_name}. Include tangible examples or facts from the coverage. Cite your sources into brackets: (media outlet, date). Your explanation should be chrnological.
    """
    
    try:
        response = chatbot.ask(question)
        return response if response else f"Analysis could not be generated for {category} during this period."
    except Exception as e:
        print(f"Error analyzing {category} period {start_date} to {end_date}: {str(e)}")
        return f"Error analyzing period: {str(e)}"

def generate_category_sentiment_section(df, company_name):
    """
    Generate sentiment analysis for top 3 categories.
    """
    # Get top 3 categories by article count
    category_counts = df['category'].value_counts()
    print(category_counts)
    top_categories = category_counts.head(3).index.tolist()
    print(top_categories)
    
    # Initialize section content
    category_section = "\n### Category Sentiment Analysis\n\n"
    
    for category in top_categories:
        category_section += f"\n#### {category}\n"
        
        # Find extrema points for category
        try:
            max_point, min_point = find_category_extrema(df, category)
            print(max_point, min_point)
            
            # Add extrema points
            category_section += f"**Peak**: {max_point[0].strftime('%B %d, %Y')} (Score: {max_point[1]:.3f})\n"
            category_section += f"**Dip**: {min_point[0].strftime('%B %d, %Y')} (Score: {min_point[1]:.3f})\n\n"

            # Analyze peak period
            peak_analysis = analyze_category_period(
                df, category, 
                max_point[0] - pd.Timedelta(days=40),
                max_point[0],
                company_name, True
            )
            print(peak_analysis)
            category_section += f"**Peak Period Analysis:**\n{peak_analysis}\n\n"
            
            # Analyze dip period
            dip_analysis = analyze_category_period(
                df, category,
                min_point[0] - pd.Timedelta(days=40),
                min_point[0],
                company_name, False
            )
            print(dip_analysis)
            category_section += f"**Dip Period Analysis:**\n{dip_analysis}\n\n"
            
        except Exception as e:
            category_section += f"Error analyzing {category}: {str(e)}\n\n"
    
    return category_section

def setup_journalist_directories(news_folder_path: str, journalist_name: str) -> str:
    """
    Set up directory structure for journalist analysis based on the provided news folder path.
    
    Args:
        news_folder_path (str): Path to the folder containing news articles
        journalist_name (str): Name of the journalist being analyzed
        
    Returns:
        str: Path to the general folder for outputs
    """
    try:
        # Get the journalist's base directory by going up one level from news folder
        general_folder = os.path.dirname(news_folder_path)
        outputs_folder = os.path.join(general_folder, "Outputs")
        compiled_outputs = os.path.join(outputs_folder, "CompiledOutputs")
        
        # Create directories
        os.makedirs(compiled_outputs, exist_ok=True)
        
        logging.info(f"Created directory structure at {general_folder}")
        return general_folder
        
    except Exception as e:
        logging.error(f"Error creating directory structure: {str(e)}")
        raise
    
def preprocess_journalist_articles(journalist_name: str, articles: List[Dict], news_folder_path: str, 
                                 force_reprocess: bool = False) -> Tuple[List[Dict], str, bool]:
    try:
        logging.info(f"Starting article preprocessing for journalist {journalist_name}")
        print(f"Found {len(articles)} articles to process")
        
        # Set up directory structure
        general_folder = setup_journalist_directories(news_folder_path, journalist_name)
        print(f"Set up directory structure at: {general_folder}")
        
        preprocessed_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", "PreprocessedArticles.json")
        
        # Check if we have preprocessed articles saved
        if not force_reprocess and os.path.exists(preprocessed_path):
            logging.info("Loading previously preprocessed articles")
            print("Found existing preprocessed articles, loading...")
            # Load the entire JSON content.
            with open(preprocessed_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Extract the articles list if the JSON is a dictionary.
            if isinstance(data, dict) and "articles" in data:
                articles_sorted = data["articles"]
            else:
                articles_sorted = data  # Fallback, in case the file is already just a list.
            return articles_sorted, general_folder, True

        
        # Extract metadata and clean articles
        print("Extracting metadata...")
        articles = extract_metadata(articles)
        articles = clean_articles(articles)
        print(f"Cleaned and extracted metadata from {len(articles)} articles")

        # Extract hyperlinks
        logging.info("Extracting hyperlinks")
        articles = add_links_to_articles(articles)

        # Sort articles by date
        print("Sorting articles...")
        articles_sorted = sorted(articles, 
                               key=lambda x: datetime.strptime(x.get('date', '2024-01-01'), '%B %d, %Y'))

        # Save preprocessed articles
        print("Saving preprocessed articles...")
        save_data_to_json(articles_sorted, preprocessed_path)
        
        logging.info("Article preprocessing completed successfully")
        print("Preprocessing completed successfully")
        
        if not articles_sorted:
            print("Warning: No articles remained after preprocessing")
            return None, general_folder, False
            
        return articles_sorted, general_folder, True

    except Exception as e:
        logging.error(f"Error in article preprocessing: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Error during preprocessing: {str(e)}")
        print(traceback.format_exc())
        return None, None, False

def generate_introduction(profile_md, journalist_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            intro_bot = ChatGPT(
                model_name="chatgpt-4o-latest",
                temperature=0,
                max_tokens=1000,
                max_retries=3,  # Add retry logic to the ChatGPT instance
                retry_delay=2
            )
            
            intro_prompt = f"""
Based on the provided journalist profile, create a comprehensive introduction section.

Profile content:
{profile_md}

Write an engaging introduction that:
1. Provides an overview of {journalist_name}'s primary areas of expertise and focus
2. Highlights their most significant or impactful coverage topics
3. Identifies any overarching patterns or themes across their work
4. Notes their typical approach to reporting and story development

Keep the introduction to approximately 300-400 words, making it substantive but concise.
Use a professional, analytical tone while remaining engaging.
Base all observations strictly on the evidence from the analyzed articles.
Start your output with "## Introduction".
"""
            
            introduction = intro_bot.ask(intro_prompt)
            if introduction and introduction.strip():
                print("Successfully generated introduction")
                return introduction
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to generate introduction: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
                continue
            else:
                raise Exception(f"Failed to generate introduction after {max_retries} attempts: {str(e)}")
    
    raise Exception("Failed to generate valid introduction after all retry attempts")

#Function to create the base for all the outputs

def preprocess_articles(company_name, articles, industry_of_interest, region, force_reprocess=False):
    """
    Performs common preprocessing steps on articles that are required for all output types.
    
    Args:
        company_name (str): Name of the company being analyzed
        articles (list): List of article dictionaries
        industry_of_interest (str): Industry relevant to the analysis
        region (str): Geographic region of interest
        force_reprocess (bool): If True, force reprocessing even if checkpoint exists.
        
    Returns:
        tuple: (preprocessed_articles, general_folder, directories_created)
    """
    try:
        # Input validation (unchanged)
        if not company_name or not isinstance(company_name, str):
            raise ValueError("Company name is required and must be a string")
        if not articles or not isinstance(articles, list):
            raise ValueError("Articles must be a non-empty list")
            
        logging.info(f"Starting article preprocessing for {company_name}")
        
        # Get the base directory from the first article's path
        first_article = articles[0] if articles else None
        if not first_article or 'file_path' not in first_article:
            raise ValueError("No valid article paths found")
            
        article_path = first_article['file_path']
        company_folder = os.path.dirname(os.path.dirname(article_path))
        if not company_folder:
            raise ValueError("Could not determine base directory from article path")

        general_folder = company_folder
        print(f"general folder: {general_folder}")
        output_folder = os.path.join(company_folder, "Outputs")

        required_dirs = [
            output_folder,
            os.path.join(output_folder, "CompiledOutputs"),
            os.path.join(output_folder, "IndividualInsights"),
            os.path.join(output_folder, "TopicsSummaries")
        ]
        print(required_dirs)
        
        directories_created = True
        for directory in required_dirs:
            try:
                os.makedirs(directory, exist_ok=True)
                if not os.path.exists(directory):
                    logging.error(f"Failed to create directory: {directory}")
                    directories_created = False
                    break
            except Exception as e:
                logging.error(f"Error creating directory {directory}: {str(e)}")
                directories_created = False
                break
        
        if not directories_created:
            raise Exception("Failed to create necessary directories")
        
        # Define checkpoint path (you can change the file name as needed)
        checkpoint_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", "ArticlesList.json")
        if not force_reprocess and os.path.exists(checkpoint_path):
            logging.info("Checkpoint found. Loading preprocessed articles from file.")
            articles_sorted = load_data_from_json(checkpoint_path)
            return articles_sorted, general_folder, True

        # If checkpoint doesn't exist, process articles normally
        logging.info("Starting article filtering and metadata extraction")
        filtered_articles = filter_relevant_articles(articles, company_name, industry_of_interest, region)
        if not filtered_articles:
            logging.warning("No articles remained after filtering for relevance")
            return None, general_folder, True
            
        logging.info(f"Extracted metadata for {len(filtered_articles)} articles")
        processed_articles = extract_metadata(filtered_articles)
        cleaned_articles = clean_articles(processed_articles)

        logging.info("Generating article embeddings")
        article_embeddings = get_embeddings(cleaned_articles, embeddings_model)
        
        logging.info("Filtering duplicates")
        unique_articles = filter_duplicates(cleaned_articles, article_embeddings)
        
        logging.info("Extracting hyperlinks")
        unique_articles = add_links_to_articles(unique_articles)

        logging.info("Sorting articles")
        articles_sorted = sorted(unique_articles, key=lambda x: x.get('timestamp', 0))

        # Save checkpoint so subsequent runs can skip reprocessing
        save_data_to_json(articles_sorted, checkpoint_path)
        logging.info(f"Article preprocessing completed successfully with {len(articles_sorted)} articles")
        return articles_sorted, general_folder, directories_created

    except Exception as e:
        logging.error(f"Error in article preprocessing: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None, False

def setup_logging(general_folder: str = None) -> None:
    """
    Set up logging configuration with both file and console handlers.
    
    Args:
        general_folder (str): Base folder for outputs. If None, use current directory.
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    root_logger.handlers = []

    # Console Handler - INFO level and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler - DEBUG level and above
    if general_folder:
        # Change this line to use the company folder directly
        log_file = os.path.join(general_folder, "media_analysis.log")
    else:
        log_file = "media_analysis.log"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Log initial setup
    logging.info(f"Logging configured. Log file: {log_file}")

def log_function_call(func):
    """
    Decorator to log function calls with arguments and return values.
    """
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logging.debug(f"Entering {func_name} - Args: {args}, Kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logging.debug(f"Exiting {func_name} - Result: {type(result)}")
            return result
        except Exception as e:
            logging.error(f"Error in {func_name}: {str(e)}", exc_info=True)
            raise
    return wrapper

def parse_stakeholder_table(raw_data: str):
    """Parse the stakeholder table with better error handling"""
    stakeholders = {}
    lines = raw_data.split('\n')
    
    for line in lines:
        if '|' not in line or line.strip().startswith('#') or '---' in line:
            continue
            
        # Split and clean parts
        parts = [part.strip() for part in line.split('|') if part.strip()]
        if len(parts) < 4:  # Need at least name, role, quote, sentiment
            continue
            
        # Extract stakeholder name
        stakeholder_name = parts[0]
        if '[' in stakeholder_name:
            stakeholder_name = stakeholder_name[stakeholder_name.find('[')+1:stakeholder_name.find(']')]
        
        if not stakeholder_name or 'Stakeholder Name' in stakeholder_name:
            continue
            
        role = parts[1]
        quote = parts[2]
        # Handle both 4 and 5 column formats
        sentiment = parts[-1] if len(parts) > 4 else "Unknown"
        
        if stakeholder_name not in stakeholders:
            stakeholders[stakeholder_name] = {
                'roles': set(),
                'quotes': [],
                'sentiments': [],
                'context': []
            }
        
        stakeholders[stakeholder_name]['roles'].add(role)
        stakeholders[stakeholder_name]['quotes'].append(quote)
        stakeholders[stakeholder_name]['sentiments'].append(sentiment)
        # Add quote as context since article links are None
        stakeholders[stakeholder_name]['context'].append(f"In their role as {role}, they stated: {quote}")
    
    return stakeholders

def process_markdown_file(input_path: str, output_path: str = None) -> None:
    """
    Process citations in a markdown file and save the result.
    
    Args:
        input_path (str): Path to the input markdown file
        output_path (str, optional): Path where the processed file should be saved.
            If not provided, will append '_processed' to the input filename.
    
    Raises:
        FileNotFoundError: If the input file doesn't exist
        PermissionError: If there are permission issues with reading/writing files
        Exception: For other unexpected errors
    """
    try:
        # Initialize the citation processor
        processor = CitationProcessor()
        
        # Ensure input path exists
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process citations
        processed_content = processor.process_citations_in_markdown(content)
        
        # Determine output path if not provided
        if output_path is None:
            output_path = str(input_file.parent / f"{input_file.stem}_processed{input_file.suffix}")
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
            
        logging.info(f"Processed citations saved to: {output_path}")
        
    except FileNotFoundError as e:
        logging.error(f"Input file not found: {str(e)}")
        raise
    except PermissionError as e:
        logging.error(f"Permission error accessing files: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error processing markdown file: {str(e)}")
        raise

def get_coverage_categories(articles_sorted: List[Dict], journalist_name: str, general_folder: str, 
                          force_reprocess: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Get or generate main coverage categories and categorized articles.
    Now uses narrative_story field for consistency across functions.
    
    Args:
        articles_sorted: List of article dictionaries
        journalist_name: Name of the journalist
        general_folder: Base folder path
        force_reprocess: If True, regenerate categories even if they exist
        
    Returns:
        Tuple of (categories_data, categorized_articles)
    """
    try:
        categorized_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                      f"CategorizedArticles_{journalist_name}.json")

        # Check if we can use existing categorized data
        if not force_reprocess and os.path.exists(categorized_path):
            logging.info("Loading existing categorized articles")
            saved_data = load_data_from_json(categorized_path)
            
            # Handle different possible JSON structures
            categories_info = []
            articles_data = []
            
            if isinstance(saved_data, list):
                # If saved_data is a list, assume it's a list of articles
                articles_data = saved_data
                # Extract unique narratives
                unique_narratives = set(article.get('narrative_story', '') for article in articles_data if article.get('narrative_story'))
                categories_info = [{'category': cat, 'description': ''} for cat in unique_narratives]
            elif isinstance(saved_data, dict):
                # If saved_data is a dict, look for categories and articles
                categories_info = saved_data.get('categories', [])
                articles_data = saved_data.get('articles', [])
            
            # Verify we have valid narrative data
            if categories_info and articles_data:
                # Verify all articles have narrative stories and match saved categories
                valid_narratives = {cat['category'] for cat in categories_info}
                all_categorized = all('narrative_story' in article for article in articles_sorted)
                narratives_match = all(
                    article.get('narrative_story', '') in valid_narratives
                    for article in articles_sorted if 'narrative_story' in article
                )
                
                if all_categorized and narratives_match:
                    # Reconstruct categories_data structure
                    categories_data = []
                    for category_info in categories_info:
                        category = {
                            'category': category_info['category'],
                            'description': category_info.get('description', ''),
                            'articles': [
                                article for article in articles_sorted 
                                if article.get('narrative_story') == category_info['category']
                            ]
                        }
                        categories_data.append(category)
                    return categories_data, articles_sorted

        # Generate one-sentence descriptions if needed
        compiled_sentences = ""
        for article in articles_sorted:
            if 'one_sentence_description' not in article or force_reprocess:
                chatbot = ChatGPT(
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=350
                )
                
                response = chatbot.ask(f"""
                Please write a single sentence summarizing this article's main topic and {journalist_name}'s angle or approach.
                Focus on both the subject matter and how {journalist_name} covers it.
                
                Article: {article.get('content', '')}
                """)
                
                article['one_sentence_description'] = response
            
            compiled_sentences += article.get('one_sentence_description', '') + "\n"

        # Save processed articles with descriptions
        save_data_to_json({"articles": articles_sorted}, categorized_path)


        # Generate narrative categories using AI
        chatbot = ChatGPT(
            model_name="models/gemini-1.5-pro",
            temperature=0,
            max_tokens=1000
        )
        
        categories_response = chatbot.ask(f"""
        Based on these article summaries, identify the main recurring stories or narrative threads in {journalist_name}'s articles.
        Create 5-8 distinct categories that represent specific stories, events, or series of connected events that appear across multiple articles.
        
        Article summaries:
        {compiled_sentences}
        
        For each category:
        1. Give it a name that describes the specific story/narrative (e.g., "Tech Company X Layoff Series" rather than just "Tech Industry")
        2. Explain what specific events, developments, or connected stories this narrative encompasses
        3. Focus on identifying stories that span multiple articles or connected events rather than broad topics
        
        Format as:
        CATEGORY: [Story/Narrative Name]
        DESCRIPTION: [Explanation of the specific story thread and how it develops across articles]
        """)
        
        # Parse categories response
        categories_data = []
        current_category = None
        current_description = None
        
        for line in categories_response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('CATEGORY:'):
                if current_category:
                    categories_data.append({
                        'category': current_category.strip(),
                        'description': current_description.strip() if current_description else '',
                        'articles': []
                    })
                current_category = line.replace('CATEGORY:', '').strip()
                current_description = None
            elif line.startswith('DESCRIPTION:'):
                current_description = line.replace('DESCRIPTION:', '').strip()
        
        # Add final category
        if current_category:
            categories_data.append({
                'category': current_category,
                'description': current_description if current_description else '',
                'articles': []
            })
        
        # Add "Other" category
        categories_data.append({
            'category': 'Other',
            'description': 'Articles that do not clearly align with specific narrative threads or recurring stories',
            'articles': []
        })
        
        # Categorize articles
        for article in articles_sorted:
            classification_prompt = "\n".join(
                f"{category['category']}: {category['description']}"
                for category in categories_data
            )
            
            chatbot = ChatGPT(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=200
            )
            
            classification = chatbot.ask(f"""
            Given these coverage categories for the journalist {journalist_name}:
            {classification_prompt}
            
            Classify this article into one of these categories. Choose the most appropriate category based on the descriptions provided.
            
            Article content:
            {article.get('content', '')}
            
            Only output the exact category name that best matches this article.
            """)

            classification = classification.replace('**', '')
            article['narrative_story'] = classification

        save_data_to_json(articles_sorted, categorized_path)

        return categories_data, articles_sorted
        
    except Exception as e:
        logging.error(f"Error in get_coverage_categories: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def create_custom_colormap(n_colors):
    """Create a custom colormap with professional colors."""
    # Professional color palette
    colors = [
        '#2E86AB',  # Blue
        '#A23B72',  # Purple
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#3B1F2B',  # Dark purple
        '#4FB477',  # Green
        '#5C4742',  # Brown
        '#6B818C',  # Gray blue
        '#7E1946',  # Wine
        '#2F9599'   # Teal
    ]
    
    # If we need more colors than in our base palette, create intermediary colors
    if n_colors > len(colors):
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
        return [custom_cmap(i/n_colors) for i in range(n_colors)]
    
    return colors[:n_colors]

def get_text_color(background_color):
    """Determine if text should be black or white based on background color."""
    # Convert color to RGB if it's not already
    if isinstance(background_color, str):
        from matplotlib.colors import to_rgb
        background_color = to_rgb(background_color)
    
    # Calculate luminance (perceived brightness)
    luminance = (0.299 * background_color[0] + 
                0.587 * background_color[1] + 
                0.114 * background_color[2])
    
    # Return white for dark backgrounds, black for light backgrounds
    return 'white' if luminance < 0.5 else 'black'

def create_professional_pie(data, title, figsize=(10, 7)):
    """
    Create a professional-looking pie chart with legend.
    
    Args:
        data: Data to plot
        title: Chart title
        figsize: Tuple of (width, height) for figure size, defaults to (10, 7)
    """
    # Create figure and axis with passed figsize
    fig = plt.figure(figsize=figsize)
    
    colors = create_custom_colormap(len(data))
    
    # Create pie chart with adjusted pctdistance for more centered text
    patches, texts, autotexts = plt.pie(
        data.values,
        colors=colors,
        autopct='%1.1f%%',
        pctdistance=0.65,  # Moved closer to center
        wedgeprops={
            'edgecolor': 'white',
            'linewidth': 2,
            'antialiased': True
        }
    )
    
    # Adjust text colors based on background
    for autotext, patch in zip(autotexts, patches):
        background_color = patch.get_facecolor()
        autotext.set_color(get_text_color(background_color))
        
        if float(autotext.get_text().strip('%')) < 3:
            autotext.set_text('')
        else:
            autotext.set_fontweight('bold')
    
    plt.legend(
        patches,
        data.index,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )
    
    fig.suptitle(title, 
                 x=0.5,
                 y=0.95,
                 fontsize=14, 
                 fontweight='bold',
                 ha='center')
    
    plt.axis('equal')
    plt.tight_layout()
    
    return fig

def create_multiple_pie_charts(data_dict, main_title, rows=2, cols=3):
    """Create multiple pie charts in a grid layout with consistent styling."""
    fig_width = 6 * cols + 2
    fig_height = 5 * rows
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(rows, cols + 1)
    
    fig.suptitle(main_title, 
                 x=0.5,
                 y=0.95,
                 fontsize=16, 
                 fontweight='bold',
                 ha='center')
    
    all_categories = set()
    for data in data_dict.values():
        all_categories.update(data.index)
    
    color_map = dict(zip(
        sorted(all_categories),
        create_custom_colormap(len(all_categories))
    ))
    
    for idx, (title, data) in enumerate(data_dict.items(), 1):
        row = (idx - 1) // cols
        col = (idx - 1) % cols
        ax = fig.add_subplot(gs[row, col])
        
        colors = [color_map[cat] for cat in data.index]
        
        patches, texts, autotexts = ax.pie(
            data,
            colors=colors,
            autopct='%1.1f%%',
            pctdistance=0.65,  # Moved closer to center
            wedgeprops={
                'edgecolor': 'white',
                'linewidth': 1.5,
                'antialiased': True
            }
        )
        
        # Adjust text colors based on background
        for autotext, patch in zip(autotexts, patches):
            background_color = patch.get_facecolor()
            autotext.set_color(get_text_color(background_color))
            
            if float(autotext.get_text().strip('%')) < 3:
                autotext.set_text('')
            else:
                autotext.set_fontweight('bold')
        
        ax.set_title(f"{title}\n({sum(data)} articles)", pad=20)
    
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[cat])
        for cat in sorted(all_categories)
    ]
    
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis('off')
    
    legend = legend_ax.legend(
        legend_elements,
        sorted(all_categories),
        title="Categories",
        loc='center left',
        bbox_to_anchor=(0, 0.5),
        frameon=False
    )
    
    plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.4)
    
    return fig

def determine_main_categories(articles_sorted: List[Dict], journalist_name: str, general_folder: str) -> Dict[str, str]:
    """
    Determine main categories for a journalist's articles using AI analysis.
    
    Args:
        articles_sorted (List[Dict]): List of sorted articles to analyze
        journalist_name (str): Name of the journalist
        general_folder (str): Base path for output files
    
    Returns:
        Dict[str, str]: Dictionary mapping category names to their descriptions
    """
    try:
        logging.info(f"Starting main category determination for {journalist_name}")
        
        # Limit to at most 75 articles by random selection if necessary
        if len(articles_sorted) > 75:
            import random
            articles_sorted = random.sample(articles_sorted, 75)
            logging.info("Articles list exceeded 75 items; randomly selected 75 articles for analysis.")
        
        # Setup output path
        categorized_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                      f"CategorizedArticles_{journalist_name}.json")
        
        # Compile articles content for analysis
        compiled_content = "\n\n".join([
            f"Title: {article.get('title', 'Unknown')}\n"
            f"Date: {article.get('date', 'Unknown')}\n"
            f"Content: {article.get('content', '')}"
            for article in articles_sorted
        ])
        
        # Initialize AI model
        chatbot = ChatGPT(
            model_name="chatgpt-4o-latest",
            temperature=0,
            max_tokens=500
        )
        
        # Generate category prompt
        main_categories_prompt = f"""
Based on these articles by {journalist_name}, identify the main high-level categories of coverage.
Create 6-10 distinct main categories (e.g., Politics, Business, Technology).

Articles content:
{compiled_content}

Output Format:
**CATEGORY:** [Category Name]
**DESCRIPTION:** [Brief description of what this category encompasses]

Keep categories broad but meaningful. Each should be clearly distinct from others.
"""
        
        # Get AI response
        main_categories_response = chatbot.ask(main_categories_prompt)
        logging.debug(f"Category response received: {main_categories_response[:200]}...")
        
        # Parse main categories
        main_categories = {}
        current_category = None
        current_description = None
        
        for line in main_categories_response.split('\n'):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            if line.startswith('**CATEGORY:**'):
                if current_category is not None:
                    main_categories[current_category] = current_description
                current_category = line.replace('**CATEGORY:**', '').strip()
                current_description = None
            elif line.startswith('**DESCRIPTION:**'):
                current_description = line.replace('**DESCRIPTION:**', '').strip()
        
        # Add the last category if exists
        if current_category is not None:
            main_categories[current_category] = current_description
        
        # Validate categories
        if not main_categories:
            raise ValueError("No categories were extracted from the AI response")
        
        logging.info(f"Successfully determined {len(main_categories)} main categories")
        return main_categories
        
    except Exception as e:
        logging.error(f"Error in determine_main_categories: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def classify_single_article(article: Dict, main_categories: Dict[str, str]) -> Dict:
    """
    Classify a single article with category, subcategory, and specific topic.
    If the article already contains classification data, it will be skipped.
    
    Args:
        article (Dict): Article to classify.
        main_categories (Dict[str, str]): Available main categories.
    
    Returns:
        Dict: Updated article with classification information.
    """
    try:
        # Check if classification already exists
        if all(article.get(key) for key in ['main_category', 'subcategory', 'specific_topic']):
            logging.info("Classification already exists for this article. Skipping classification.")
            return article

        content = article.get('content', '')
        
        # Initialize AI model for category classification
        chatbot = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=200
        )
        
        # Determine main category
        category_prompt = f"""
Classify this article into one of the following main categories:
{json.dumps(main_categories, indent=2)}

Article:
Content: {content}

Output only the category name that best matches this article and nothing else. Choose from: {', '.join(main_categories.keys())}
"""
        main_category = chatbot.ask(category_prompt).strip()
        article['main_category'] = main_category
        
        # Determine subcategory
        subcategory_prompt = f"""
For an article in the main category "{main_category}", determine the most appropriate subcategory.
Subcategories should be more specific than the main category but still broad enough to group related articles.

Example subcategories for different main categories:
- Politics: Elections, Legislation, International Relations
- Technology: AI/ML, Cybersecurity, Consumer Tech
- Business: Startups, Market Analysis, Corporate Strategy

Article:
Content: {content}

Output only the subcategory name that best describes this article's focus.
"""
        article['subcategory'] = chatbot.ask(subcategory_prompt).strip()
        
        # Determine specific topic
        topic_prompt = f"""
For this article in category "{main_category}" and subcategory "{article['subcategory']}", 
determine the specific topic or story being covered.

The specific topic should be highly precise and descriptive of the exact subject matter.
Examples:
- "2024 Presidential Primary Debates" (not just "Elections")
- "OpenAI Leadership Crisis" (not just "AI Companies")
- "Tesla Q4 Earnings Report" (not just "Financial Results")

Article:
Content: {content}

Output a brief but specific topic description (5-8 words maximum) that precisely identifies what this article covers.
"""
        article['specific_topic'] = chatbot.ask(topic_prompt).strip()
        
        return article
        
    except Exception as e:
        logging.error(f"Error classifying article: {str(e)}")
        article['classification_error'] = str(e)
        return article

def classify_articles(journalist_name: str, articles_sorted: List[Dict], main_categories: Dict[str, str], 
                     general_folder: str) -> List[Dict]:
    """
    Classify all articles with categories, subcategories, and specific topics.
    If an article is already classified, its values are preserved.
    
    Args:
        journalist_name (str): Name of the journalist.
        articles_sorted (List[Dict]): List of articles to classify.
        main_categories (Dict[str, str]): Available main categories.
        general_folder (str): Base path for output files.
    
    Returns:
        List[Dict]: Updated list of articles with classification information.
    """
    try:
        logging.info("Starting article classification")
        categorized_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                        f"CategorizedArticles_{journalist_name}.json")
        
        classified_articles = []
        total_articles = len(articles_sorted)
        
        # Verify the data structure
        if total_articles > 0 and isinstance(articles_sorted[0], str):
            logging.warning("Articles are in string format, attempting to parse from JSON")
            try:
                articles_sorted = [json.loads(article) if isinstance(article, str) else article 
                                 for article in articles_sorted]
            except json.JSONDecodeError:
                logging.error("Could not parse articles as JSON strings")
                return []
        
        for i, article in enumerate(articles_sorted, 1):
            logging.info(f"Classifying article {i}/{total_articles}")
            try:
                # Ensure article is a dictionary and has content
                if not isinstance(article, dict):
                    logging.warning(f"Article {i} is not a dictionary, skipping")
                    continue
                
                if not article.get('content'):
                    logging.warning(f"No content found for article {i}, skipping")
                    continue
                
                # Classify article (skips if classification already exists)
                classified_article = classify_single_article(article, main_categories)
                classified_articles.append(classified_article)
                
            except Exception as e:
                logging.error(f"Error classifying article {i}: {str(e)}")
                if isinstance(article, dict):
                    article = article.copy()
                    article['classification_error'] = str(e)
                    classified_articles.append(article)
                
            if i % 10 == 0:
                save_data_to_json({"categories": [], "articles": classified_articles}, categorized_path)

        save_data_to_json(classified_articles, categorized_path)

        
        logging.info("Article classification completed")
        return classified_articles
        
    except Exception as e:
        logging.error(f"Error in classify_articles: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def generate_categorization_markdown(articles_sorted: List[Dict], journalist_name: str, 
                               general_folder: str) -> str:
    """
    Generate a markdown summary of article categorization.
    
    Args:
        articles_sorted (List[Dict]): List of categorized articles
        journalist_name (str): Name of the journalist
        general_folder (str): Base path for output files
    
    Returns:
        str: Generated markdown content
    """
    try:
        logging.info(f"Generating categorization markdown for {journalist_name}")
        
        # Initialize markdown content
        categorization_md = f"""# Article Categorization for {journalist_name}

## Overview
This analysis provides a hierarchical categorization of {len(articles_sorted)} articles by {journalist_name}.

## Category Hierarchy
"""
        
        # Group articles by category and subcategory
        category_structure = defaultdict(lambda: defaultdict(list))
        for article in articles_sorted:
            # Handle missing subcategory gracefully
            subcategory = article.get('subcategory', 'Uncategorized')
            category = article.get('category', 'Uncategorized')
            category_structure[category][subcategory].append(article)
        
        # Build structured summary
        for category, subcategories in category_structure.items():
            categorization_md += f"\n### {category}\n"
            
            for subcategory, articles in subcategories.items():
                categorization_md += f"\n#### {subcategory} ({len(articles)} articles)\n"
                
                for article in articles:
                    date = article.get('date', 'Unknown')
                    link = article.get('link', '#')
                    title = article.get('title', 'Unknown')
                    topic = article.get('specific_topic', 'Unknown')
                    
                    categorization_md += f"- [{title}]({link}) ({date}): {topic}\n"
        
        # Add entity information if present
        for article in articles_sorted:
            if article.get('organizations'):
                categorization_md += "\n\n#### Organizations Mentioned:\n"
                for org in article['organizations']:
                    categorization_md += f"- **{org['name']}** ({org['type']}): {org['description']}\n"
            
            if article.get('people'):
                categorization_md += "\n#### People Mentioned:\n"
                for person in article['people']:
                    categorization_md += f"- **{person['name']}** ({person['role']}): {person['context']}\n"
        
        # Save markdown summary
        summary_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                  f"Categories_{journalist_name}.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(categorization_md)
        
        logging.info("Categorization markdown generated successfully")
        return categorization_md
        
    except Exception as e:
        logging.error(f"Error generating categorization markdown: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def save_to_json(data, filepath):
    """
    Safely save data to JSON file with proper encoding and timestamp handling.
    
    Args:
        data: Data to save
        filepath (str): Path to save the JSON file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=json_serial, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logging.error(f"Error saving to JSON: {str(e)}")
        raise

def extract_organizations(article: Dict, journalist_name: str) -> List[Dict]:
    """
    Extract organizations with enhanced error handling and validation.
    If the article already contains valid organizations, skip extraction.
    """
    try:
        # Validate existing organizations, don't just check if the key exists
        if article.get('organizations') and isinstance(article['organizations'], list):
            valid_orgs = [
                org for org in article['organizations'] 
                if isinstance(org, dict) and org.get('name') and len(org.get('name').strip()) > 0
            ]
            
            # Only reuse existing organizations if they're valid
            if valid_orgs:
                logging.info(f"Valid organizations already extracted for article: {article.get('title', 'Unknown Title')}. Skipping.")
                return valid_orgs
            else:
                logging.warning(f"Article had organizations but they were invalid. Re-extracting.")
        
        # Debug incoming article
        logging.info(f"Starting organization extraction for article type: {type(article)}")
        if isinstance(article, dict):
            logging.info(f"Article keys: {list(article.keys())}")
        else:
            logging.error(f"Invalid article type: {type(article)}")
            return []

        # Get content and title safely
        content = article.get('content', '')
        title = article.get('title', 'Unknown Title')
        logging.info(f"Processing article: {title}")

        if not content:
            logging.warning(f"No content found for article: {title}")
            return []

        chatbot = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=1500
        )
        
        org_prompt = f"""
Extract a list of all organizations, companies, institutions, and other entities mentioned in this article. You must exclude all real people from your output, it should only list organizations, companies, institutions, excluding people.

Requirements:
1. Include all types of organizations (companies, government agencies, NGOs, etc.)
2. Only include entities that are actually discussed, not just mentioned in passing
3. Remove any duplicates
4. Include full official names where possible.
5. Do not include broad terms like "government" or "companies" without specific identifiers or a specific organisation.
6. DO NOT include people.
7. Limit the Organisation name (**ENTITY**) to a maximum of 3 to 4 words.
8. The entity name MUST NOT be empty - if you can't identify the name, skip that entity entirely.

Article:
Title: {title}
Content: {content}

Output Format:
**ENTITY**: [Organization Name]
**TYPE**: [Type of organization - e.g., Company, Government Agency, NGO, etc.]
**DESCRIPTION**: [Brief description of how the entity is discussed in the article]

Example Output:
**ENTITY**: Tesla, Inc.
**TYPE**: Company
**DESCRIPTION**: Main subject of article, discussing Q4 earnings report

Only include entities that are meaningful to the article's content. Return "None" in case there are no relevant organisations to be mentioned.
You must exclude {journalist_name} from your output.
"""
        org_response = chatbot.ask(org_prompt)
        
        # Parse organizations into structured format
        organizations = []
        current_org = {}
        
        if org_response.strip().lower() == "none":
            return []
        
        for line in org_response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('**ENTITY**:'):
                # If we have a complete organization, add it to the list
                if current_org and all(k in current_org for k in ['name', 'type', 'description']):
                    # Only add it if the name is not empty
                    if current_org['name'] and len(current_org['name'].strip()) > 0:
                        organizations.append(current_org.copy())
                
                # Get the new entity name
                entity_name = line.replace('**ENTITY**:', '').strip()
                
                # Only create a new org if the name is not empty
                if entity_name:
                    current_org = {'name': entity_name}
                else:
                    current_org = {}  # Reset without adding a name
                    
            elif line.startswith('**TYPE**:'):
                if current_org:  # Only add type if we have a valid org
                    current_org['type'] = line.replace('**TYPE**:', '').strip()
            elif line.startswith('**DESCRIPTION**:'):
                if current_org:  # Only add description if we have a valid org
                    current_org['description'] = line.replace('**DESCRIPTION**:', '').strip()
        
        # Add last organization if complete and name is not empty
        if current_org and all(k in current_org for k in ['name', 'type', 'description']):
            if current_org['name'] and len(current_org['name'].strip()) > 0:
                organizations.append(current_org.copy())
        
        # Validate all organizations have non-empty names
        valid_orgs = [
            org for org in organizations 
            if isinstance(org, dict) and org.get('name') and len(org.get('name').strip()) > 0
        ]
        
        if len(valid_orgs) < len(organizations):
            logging.warning(f"Filtered out {len(organizations) - len(valid_orgs)} invalid organizations")
            
        return valid_orgs
        
    except Exception as e:
        logging.error(f"Error extracting organizations from article '{article.get('title', 'Unknown Title')}': {str(e)}")
        return []

def extract_people(article: Dict, journalist_name: str) -> List[Dict]:
    """
    Extract people mentioned in an article using AI analysis with enhanced validation.
    If the article already contains valid people data, skip extraction.
    """
    try:
        # Validate existing people data, don't just check if the key exists
        if article.get('people') and isinstance(article['people'], list):
            valid_people = [
                person for person in article['people'] 
                if isinstance(person, dict) and person.get('name') and len(person.get('name').strip()) > 0
            ]
            
            # Only reuse existing people if they're valid
            if valid_people:
                logging.info(f"Valid people already extracted for article: {article.get('title', 'Unknown Title')}. Skipping.")
                return valid_people
            else:
                logging.warning(f"Article had people but they were invalid. Re-extracting.")

        chatbot = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=1500
        )
        content = article.get('content', '')
        title = article.get('title', '')
        
        people_prompt = f"""
Extract a list of all real people mentioned in this article.

Requirements:
1. Include only real individuals, not fictional characters or generic references
2. Only include people who are actually discussed, not just mentioned in passing
3. Remove any duplicates
4. Include full names where available
5. Do not include the article's author
6. Do not include generic references like "employees" or "officials" without specific names
7. DO NOT include non-people entities like institutions, companies, NGOs, etc...
8. The person name MUST NOT be empty - if you can't identify the name, skip that person entirely.

Article:
Title: {title}
Content: {content}

Output Format:
**PERSON**: [Full Name]
**ROLE**: [Person's role or position]
**CONTEXT**: [Brief description of how the person is discussed in the article]

Example Output:
**PERSON**: Elon Musk
**ROLE**: CEO of Tesla and SpaceX
**CONTEXT**: Primary subject of article, discussing company strategy

Only include people who play a meaningful role in the article's content. Return "None" if no relevant people are mentioned.
You must exclude {journalist_name} from your output.
"""
        
        people_response = chatbot.ask(people_prompt)
        
        # Parse people into structured format
        people = []
        current_person = {}
        
        if people_response.strip().lower() == "none":
            return []
        
        for line in people_response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('**PERSON**:'):
                # If we have a complete person, add it to the list
                if current_person and all(k in current_person for k in ['name', 'role', 'context']):
                    # Only add it if the name is not empty
                    if current_person['name'] and len(current_person['name'].strip()) > 0:
                        people.append(current_person.copy())
                
                # Get the new person name
                person_name = line.replace('**PERSON**:', '').strip()
                
                # Only create a new person if the name is not empty
                if person_name:
                    current_person = {'name': person_name}
                else:
                    current_person = {}  # Reset without adding a name
                    
            elif line.startswith('**ROLE**:'):
                if current_person:  # Only add role if we have a valid person
                    current_person['role'] = line.replace('**ROLE**:', '').strip()
            elif line.startswith('**CONTEXT**:'):
                if current_person:  # Only add context if we have a valid person
                    current_person['context'] = line.replace('**CONTEXT**:', '').strip()
        
        # Add last person if complete and name is not empty
        if current_person and all(k in current_person for k in ['name', 'role', 'context']):
            if current_person['name'] and len(current_person['name'].strip()) > 0:
                people.append(current_person.copy())
        
        # Validate all people have non-empty names
        valid_people = [
            person for person in people 
            if isinstance(person, dict) and person.get('name') and len(person.get('name').strip()) > 0
        ]
        
        if len(valid_people) < len(people):
            logging.warning(f"Filtered out {len(people) - len(valid_people)} invalid people")
            
        return valid_people
        
    except Exception as e:
        logging.error(f"Error extracting people from article '{title}': {str(e)}")
        return []

def extract_entities(articles_sorted: List[Dict], journalist_name: str, general_folder: str) -> List[Dict]:
    """
    Extract organizations and people from all articles with enhanced validation.
    Ensures all extracted entities have valid data structures.
    """
    try:
        logging.info(f"Starting entity extraction for {journalist_name}'s articles")
        
        # Debug incoming data
        logging.info(f"Type of articles_sorted: {type(articles_sorted)}")
        if isinstance(articles_sorted, pd.DataFrame):
            logging.info("Converting DataFrame to records")
            articles_sorted = articles_sorted.to_dict('records')
        
        if len(articles_sorted) > 0:
            logging.info(f"Type of first article: {type(articles_sorted[0])}")
            if isinstance(articles_sorted[0], dict):
                logging.info(f"First article keys: {list(articles_sorted[0].keys())}")
        
        categorized_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                      f"CategorizedArticles_{journalist_name}.json")
        
        processed_articles = []
        total_articles = len(articles_sorted)
        
        for i, article in enumerate(articles_sorted, 1):
            logging.info(f"Processing article {i}/{total_articles}")
            
            # Convert string to dict if necessary
            if isinstance(article, str):
                try:
                    article = json.loads(article)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse article {i} as JSON string")
                    continue
            
            if not isinstance(article, dict):
                logging.error(f"Article {i} is not a dictionary: {type(article)}")
                continue
                
            try:
                # Create a copy of the article to modify
                processed_article = article.copy()
                logging.info(f"Processing article with title: {processed_article.get('title', 'No title')}")
                
                # Extract organizations with validation
                organizations = extract_organizations(processed_article, journalist_name)
                # Ensure we only store valid organizations
                valid_orgs = [
                    org for org in organizations
                    if isinstance(org, dict) and 
                    org.get('name') and 
                    len(org.get('name', '').strip()) > 0
                ]
                
                processed_article['organizations'] = valid_orgs
                logging.info(f"Extracted {len(valid_orgs)} valid organizations")
                
                # Extract people with validation
                people = extract_people(processed_article, journalist_name)
                # Ensure we only store valid people
                valid_people = [
                    person for person in people
                    if isinstance(person, dict) and 
                    person.get('name') and 
                    len(person.get('name', '').strip()) > 0
                ]
                
                processed_article['people'] = valid_people
                logging.info(f"Extracted {len(valid_people)} valid people")
                
                processed_articles.append(processed_article)
                
            except Exception as e:
                logging.error(f"Error processing article {i}: {str(e)}")
                if isinstance(article, dict):
                    article['processing_error'] = str(e)
                    # Ensure article has valid organizations and people arrays even on error
                    if 'organizations' not in article:
                        article['organizations'] = []
                    if 'people' not in article:
                        article['people'] = []
                    processed_articles.append(article)
            
            # Save progress periodically
            if i % 5 == 0:
                try:
                    save_data_to_json(processed_articles, categorized_path)

                    logging.info(f"Saved progress after {i} articles")
                except Exception as e:
                    logging.error(f"Error saving progress: {str(e)}")
        
        # Validate all processed articles before final save
        for article in processed_articles:
            # Ensure organizations is a list of valid dicts with names
            if not isinstance(article.get('organizations'), list):
                article['organizations'] = []
            else:
                article['organizations'] = [
                    org for org in article['organizations']
                    if isinstance(org, dict) and org.get('name')
                ]
                
            # Ensure people is a list of valid dicts with names
            if not isinstance(article.get('people'), list):
                article['people'] = []
            else:
                article['people'] = [
                    person for person in article['people']
                    if isinstance(person, dict) and person.get('name')
                ]
        
        # Final save
        try:
            with open(categorized_path, 'w') as f:
                json.dump(processed_articles, f, indent=2)
            logging.info("Entity extraction completed")
        except Exception as e:
            logging.error(f"Error in final save: {str(e)}")
        
        return processed_articles
        
    except Exception as e:
        logging.error(f"Error in extract_entities: {str(e)}")
        logging.error(traceback.format_exc())
        # Return a valid but empty dataset in case of complete failure
        for article in articles_sorted:
            if isinstance(article, dict):
                article['organizations'] = article.get('organizations', [])
                article['people'] = article.get('people', [])
        return articles_sorted
    
def analyze_entity_sentiment(entity: Dict, article: Dict, entity_type: str) -> Dict:
    """
    Analyze sentiment for an organization or person in an article.
    If the entity already has sentiment analysis data ('tone' and 'sentiment_score'),
    skip re-analysis.
    
    Args:
        entity (Dict): Entity (organization or person) to analyze.
        article (Dict): Article containing the entity.
        entity_type (str): Type of entity ('organization' or 'person').
    
    Returns:
        Dict: Updated entity with sentiment information.
    """
    try:
        # If sentiment analysis has already been performed, skip reprocessing.
        if entity.get('tone') and entity.get('sentiment_score') is not None:
            logging.info(f"Sentiment already analyzed for entity: {entity.get('name', 'unknown')}. Skipping.")
            return entity

        chatbot = ChatGPT(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=100
        )
        
        content = article.get('content', '')
        title = article.get('title', '')
        
        # Determine context field based on entity type
        context_field = 'description' if entity_type == 'organization' else 'context'
        entity_name = entity['name']
        entity_context = entity.get(context_field, '')
        
        # Generate tone prompt based on entity type
        if entity_type == 'organization':
            tone_prompt = f"""
Analyze how {entity_name} is portrayed in this article. Consider the language used, context, and framing by the journalist.

Article:
Title: {title}
Content: {content}

Organization Context: {entity_context}

Determine if the portrayal of {entity_name} is "Positive", "Neutral", or "Negative" in the article.
Output ONLY ONE of these three words and nothing else.
"""
        else:  # person
            tone_prompt = f"""
Analyze how {entity_name} is portrayed in this article. Consider the language used, context, and framing by the journalist.

Article:
Title: {title}
Content: {content}

Person's Context: {entity_context}

Determine if {entity_name}'s portrayal in the provided article is "Positive", "Neutral", or "Negative".
Output ONLY ONE of these three words and nothing else.
"""
        
        # Get tone classification
        entity['tone'] = chatbot.ask(tone_prompt).strip()
        logging.info(f"Tone for {entity_name}: {entity['tone']}")
        
        # Generate sentiment prompt
        sentiment_prompt = f"""
Rate the sentiment towards {entity_name} in this article on a scale from -5 to 5, where:

-5: Extremely negative portrayal (severe criticism, scandal, major failure)
-4: Very negative portrayal (significant problems, strong criticism)
-3: Negative portrayal (clear problems or criticism)
-2: Somewhat negative portrayal (mild criticism or concerns)
-1: Slightly negative portrayal (minor issues or concerns)
0: Neutral portrayal (balanced or purely factual)
1: Slightly positive portrayal (minor achievements or praise)
2: Somewhat positive portrayal (moderate success or praise)
3: Positive portrayal (clear success or strong positive aspects)
4: Very positive portrayal (significant achievements, strong praise)
5: Extremely positive portrayal (exceptional success, highest praise)

Article:
Title: {title}
Content: {content}

{entity_type.capitalize()} Context: {entity_context}

Output ONLY a number from -5 to 5 and nothing else.
"""
        
        sentiment_response = chatbot.ask(sentiment_prompt).strip()
        logging.info(f"Sentiment response for {entity_name}: {sentiment_response}")
        try:
            entity['sentiment_score'] = int(sentiment_response)
        except ValueError:
            logging.error(f"Invalid sentiment score for {entity_name}: {sentiment_response}")
            entity['sentiment_score'] = 0
            
        return entity
        
    except Exception as e:
        logging.error(f"Error analyzing sentiment for {entity.get('name', 'unknown')}: {str(e)}")
        entity['tone'] = 'Neutral'
        entity['sentiment_score'] = 0
        return entity

def analyze_article_sentiments(article: Dict) -> Dict:
    """
    Analyze sentiments for all entities in an article.
    
    Args:
        article (Dict): Article to analyze.
    
    Returns:
        Dict: Updated article with sentiment information.
    """
    try:
        # Analyze organizations
        if 'organizations' in article:
            article['organizations'] = [
                analyze_entity_sentiment(org, article, 'organization')
                for org in article['organizations']
            ]
        
        # Analyze people
        if 'people' in article:
            article['people'] = [
                analyze_entity_sentiment(person, article, 'person')
                for person in article['people']
            ]
        
        return article
        
    except Exception as e:
        logging.error(f"Error in analyze_article_sentiments: {str(e)}")
        return article

def analyze_all_sentiments(articles_sorted: List[Dict], journalist_name: str, general_folder: str) -> List[Dict]:
    """
    Analyze sentiments for all entities in all articles.
    
    Args:
        articles_sorted (List[Dict]): List of articles to analyze.
        journalist_name (str): Name of the journalist.
        general_folder (str): Base path for output files.
    
    Returns:
        List[Dict]: Updated list of articles with sentiment information.
    """
    try:
        logging.info(f"Starting sentiment analysis for {journalist_name}'s articles")
        categorized_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                      f"CategorizedArticles_{journalist_name}.json")
        
        # Convert DataFrame to dict records if needed
        if isinstance(articles_sorted, pd.DataFrame):
            articles_sorted = articles_sorted.to_dict('records')
            
        # Convert Timestamp objects to string format if necessary
        for article in articles_sorted:
            if isinstance(article.get('date'), pd.Timestamp):
                article['date'] = article['date'].strftime('%Y-%m-%d')
            if isinstance(article.get('timestamp'), pd.Timestamp):
                article['timestamp'] = article['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        total_articles = len(articles_sorted)
        processed_articles = []
        
        for i, article in enumerate(articles_sorted, 1):
            logging.info(f"Analyzing sentiments for article {i}/{total_articles}")
            try:
                processed_article = analyze_article_sentiments(article)
                processed_articles.append(processed_article)
                
                # Save progress periodically
                if i % 5 == 0:
                    try:
                        save_data_to_json(processed_articles, categorized_path)

                    except Exception as save_error:
                        logging.error(f"Error saving progress: {str(save_error)}")
                        
            except Exception as e:
                logging.error(f"Error processing article {i}: {str(e)}")
                processed_articles.append(article)
        
        # Final save
        try:
            with open(categorized_path, 'w', encoding='utf-8') as f:
                json.dump(processed_articles, f, indent=2, ensure_ascii=False)
            logging.info("Sentiment analysis completed")
        except Exception as save_error:
            logging.error(f"Error in final save: {str(save_error)}")
        
        return processed_articles
        
    except Exception as e:
        logging.error(f"Error in analyze_all_sentiments: {str(e)}")
        logging.error(traceback.format_exc())
        return articles_sorted

def update_markdown_with_sentiments(articles_sorted: List[Dict], categorization_md: str,
                                  journalist_name: str, general_folder: str) -> str:
    """
    Update markdown content with sentiment information.
    
    Args:
        articles_sorted (List[Dict]): List of analyzed articles
        categorization_md (str): Existing markdown content
        journalist_name (str): Name of the journalist
        general_folder (str): Base path for output files
    
    Returns:
        str: Updated markdown content
    """
    try:
        logging.info("Updating markdown with sentiment information")
        
        # Remove existing entity sections
        categorization_md = categorization_md.replace("#### Organizations Mentioned:", "")
        categorization_md = categorization_md.replace("#### People Mentioned:", "")
        
        # Add updated entity sections with sentiment
        for article in articles_sorted:
            # Add organizations section
            if article.get('organizations'):
                categorization_md += "\n\n#### Organizations Mentioned:\n"
                for org in article['organizations']:
                    sentiment_emoji = "🟢" if org['tone'] == "Positive" else "🔴" if org['tone'] == "Negative" else "⚪"
                    categorization_md += (f"- {sentiment_emoji} **{org['name']}** ({org['type']}) - "
                                       f"Sentiment: {org['sentiment_score']} - {org['description']}\n")
            
            # Add people section
            if article.get('people'):
                categorization_md += "\n#### People Mentioned:\n"
                for person in article['people']:
                    sentiment_emoji = "🟢" if person['tone'] == "Positive" else "🔴" if person['tone'] == "Negative" else "⚪"
                    categorization_md += (f"- {sentiment_emoji} **{person['name']}** ({person['role']}) - "
                                       f"Sentiment: {person['sentiment_score']} - {person['context']}\n")
        
        # Save updated markdown
        summary_path = os.path.join(general_folder, "Outputs", "CompiledOutputs", 
                                  f"Categories_{journalist_name}.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(categorization_md)
        
        logging.info("Markdown updated successfully")
        return categorization_md
        
    except Exception as e:
        logging.error(f"Error updating markdown with sentiments: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def generate_top_journalists_analysis(df, company_name):
    """
    Generate analysis of top 3 journalists' coverage and sentiment towards the company using a single chatbot.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the articles data
        company_name (str): Name of the company being analyzed
        
    Returns:
        str: Markdown formatted analysis of top journalists
    """
    # Get top 3 journalists
    top_authors = df[df['author_name'] != 'Anonymous']
    top_3_authors = top_authors['author_name'].value_counts().head(3)
    
    # Prepare data for analysis
    analysis_data = []
    for author in top_3_authors.index:
        author_articles = df[df['author_name'] == author]
        analysis_data.append({
            'author': author,
            'articles': [
                {
                    'content': row['content'],
                    'tone': row['tone'],
                    'sentiment_score': row['sentiment score']
                }
                for _, row in author_articles.iterrows()
            ]
        })
    
    # Prepare prompt for the analysis
    analysis_prompt = f"""
As a media and sentiment analysis expert, analyze the coverage of {company_name} by the following journalists based on their articles. Follow these instructions:

1. Write a single paragraph text, without header, to describe both the respective coverage of these journalists with regards to {company_name} and an explanation of their respective sentiment towards {company_name}.
2. We want to get a grasp of these top 3 journalists' main narratives with regards to {company_name} 
3. We want to understand their sentiment or stance towards {company_name}
4. Be concise and fact-based.

Here is the data for each journalist:
    """
    
    for data in analysis_data:
        analysis_prompt += f"\n\nJournalist: {data['author']}\nArticles:\n"
        for article in data['articles']:
            analysis_prompt += f"- Content: {article['content']}\n"
            analysis_prompt += f"  Tone: {article['tone']}\n"
            analysis_prompt += f"  Sentiment Score: {article['sentiment_score']}\n"
    
    # Get analysis from chatbot
    chatbot = ChatGPT(
        system_prompt="You are a media analysis expert specializing in both content analysis and sentiment analysis of news coverage. Provide clear, concise analysis that connects content choices with sentiment patterns.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500
    )
    analysis = chatbot.ask(analysis_prompt)

    # Get remaining journalists (excluding top 3 and Anonymous)
    remaining_authors = set(top_authors['author_name']) - set(top_3_authors.index)
    
    # Calculate average sentiment for remaining journalists
    remaining_sentiments = {}
    for author in remaining_authors:
        author_articles = df[df['author_name'] == author]
        avg_sentiment = author_articles['sentiment score'].mean()
        remaining_sentiments[author] = avg_sentiment
    
    # Get 2 most positive and 2 most negative journalists
    sorted_sentiments = sorted(remaining_sentiments.items(), key=lambda x: x[1])
    most_negative = sorted_sentiments[:2]  # First 2 are most negative
    most_positive = sorted_sentiments[-2:][::-1]  # Last 2 are most positive, reverse to get highest first
    
    # Prepare data for sentiment extremes analysis
    extreme_analysis_data = []
    
    # Add most negative journalists
    for i, (author, sentiment) in enumerate(most_negative):
        author_articles = df[df['author_name'] == author]
        extreme_analysis_data.append({
            'author': author,
            'average_sentiment': sentiment,
            'rank': f"{i+1} most negative",
            'articles': [
                {
                    'content': row['content'],
                    'tone': row['tone'],
                    'sentiment_score': row['sentiment score']
                }
                for _, row in author_articles.iterrows()
            ]
        })
    
    # Add most positive journalists
    for i, (author, sentiment) in enumerate(most_positive):
        author_articles = df[df['author_name'] == author]
        extreme_analysis_data.append({
            'author': author,
            'average_sentiment': sentiment,
            'rank': f"{i+1} most positive",
            'articles': [
                {
                    'content': row['content'],
                    'tone': row['tone'],
                    'sentiment_score': row['sentiment score']
                }
                for _, row in author_articles.iterrows()
            ]
        })
    
    # Second chatbot - Analysis of sentiment extremes
    extreme_analysis_prompt = f"""
As a sentiment analysis specialist, analyze the coverage of {company_name} by these journalists who show the most extreme sentiment scores in their reporting with regards to {company_name}. You will be fed with the two most positive and the two most negative journalists and their corresponding article content. These journalists represent the most positive and most negative coverage outside the top 3 most frequent writers.

For each journalist, please:
- Analyze their narrative approach with regards to {company_name}.
- Describe their sentiment with regards to {company_name}, your output should explain why was this journalist judged to be so negative or so positive.
- For the most positive journalist, aim to explain why they are judged more positive, with regards to {company_name} than the others. For the most negative ones, explain why they are judged more negative than the others, with regards to {company_name}.
- Be concise and factual.
- Format your output as plain text, structured into two paragraphs (one for the more positive journalists and one for the most negative ones). Do not include headers. Do not mention that they are the two most positive or negative journalist, instead say: "Among the most positive/negative journalists", for instance.

Here is the data for analysis:
    """
    
    # First add negative journalists
    extreme_analysis_prompt += "\n\nA. TWO MOST NEGATIVE JOURNALISTS:\n"
    for data in extreme_analysis_data[:2]:
        extreme_analysis_prompt += f"\n{data['rank']} Journalist: {data['author']}\n"
        extreme_analysis_prompt += f"Average Sentiment Score: {data['average_sentiment']:.2f}\nArticles:\n"
        for article in data['articles']:
            extreme_analysis_prompt += f"- Content: {article['content']}\n"
            extreme_analysis_prompt += f"  Tone: {article['tone']}\n"
            extreme_analysis_prompt += f"  Sentiment Score: {article['sentiment_score']}\n"
            
    # Then add positive journalists
    extreme_analysis_prompt += "\n\nB. TWO MOST POSITIVE JOURNALISTS:\n"
    for data in extreme_analysis_data[2:]:
        extreme_analysis_prompt += f"\n{data['rank']} Journalist: {data['author']}\n"
        extreme_analysis_prompt += f"Average Sentiment Score: {data['average_sentiment']:.2f}\nArticles:\n"
        for article in data['articles']:
            extreme_analysis_prompt += f"- Content: {article['content']}\n"
            extreme_analysis_prompt += f"  Tone: {article['tone']}\n"
            extreme_analysis_prompt += f"  Sentiment Score: {article['sentiment_score']}\n"
    
    # Get extreme sentiment analysis from second chatbot
    extreme_chatbot = ChatGPT(
        system_prompt="You are a sentiment analysis specialist focusing on identifying and explaining extreme positive and negative coverage patterns in media reporting. Your analysis should clearly connect specific content elements to sentiment outcomes.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500
    )
    extreme_analysis = extreme_chatbot.ask(extreme_analysis_prompt)
    
    # Combine analyses into markdown content
    markdown_content = f"""\n
### Analysis of Top Journalists' Coverage

{analysis}

{extreme_analysis}
\n\n
"""
    print(markdown_content)
    
    return markdown_content

def parse_ranking_output(ranking_text: str) -> list:
    """
    Parse the ranking output text into a structured format.
    
    Args:
        ranking_text (str): Raw text containing rankings
        
    Returns:
        list: List of dictionaries with rank, category, and explanation
    """
    rankings = []
    # Split the text into individual ranking entries
    entries = ranking_text.split('\nRank')
    
    for entry in entries:
        if not entry.strip():
            continue
            
        # If entry doesn't start with 'Rank', add it back
        if not entry.startswith('Rank'):
            entry = 'Rank' + entry
            
        # Extract rank
        rank_match = entry.split(':', 1)
        if len(rank_match) != 2:
            continue
            
        rank_str, remainder = rank_match
        rank = int(rank_str.replace('Rank', '').strip())
        
        # Split the remainder at the last occurrence of ' - ' before any period
        parts = remainder.strip().split(' - ')
        
        # The category is the first part
        category = parts[0].strip()
        
        # The explanation is everything else joined back together
        explanation = ' - '.join(parts[1:]).strip() if len(parts) > 1 else ""
        
        rankings.append({
            'rank': rank,
            'category': category,
            'explanation': explanation
        })
    
    return rankings

def generate_markdown_table(rankings: list) -> str:
    """
    Generate a Markdown table from a list of ranking dictionaries.
    
    Args:
        rankings (list): List of dictionaries containing 'rank', 'category', and 'explanation' keys
        
    Returns:
        str: Formatted markdown table
    """
    # Initialize table headers
    table = "| Rank | Category | Explanation |\n"
    table += "|------|----------|-------------|\n"
    
    for item in rankings:
        # Get raw values and ensure they're strings
        rank = str(item['rank'])
        category = item['category'].strip()
        explanation = item['explanation'].strip()
        
        # Escape any vertical bars in the text with HTML entities
        category = category.replace('|', '&#124;')
        explanation = explanation.replace('|', '&#124;')
        
        # Add the row to the table
        table += f"| {rank} | {category} | {explanation} |\n"
    
    return table

def generate_publication_timeline_chart(df: pd.DataFrame) -> str:
    """
    Generate a professional line chart showing the number of articles published on a weekly basis.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least a 'date' column in datetime format.
        
    Returns:
        str: Base64-encoded PNG image of the publication timeline chart.
    """
    # Ensure the 'date' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Group articles by week using pd.Grouper with frequency 'W'
    weekly = df.groupby(pd.Grouper(key='date', freq='W')).size().reset_index(name='articles')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(weekly['date'], weekly['articles'], marker='o', linestyle='-', color='#2a9d8f')
    plt.title("Weekly Articles Publication Timeline", fontsize=16)
    plt.xlabel("Week", fontsize=14)
    plt.ylabel("Number of Articles", fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save and return the plot as a base64 string using the provided function
    image_base64 = save_plot_base64()
    plt.close()
    return image_base64

def generate_publication_timeline_section(df: pd.DataFrame, company_name: str, language: str) -> str:
    # Generate the weekly timeline chart image
    publication_timeline_chart = generate_publication_timeline_chart(df)
    timeline_header = "The chart below illustrates when the articles were published on a weekly basis. This view helps identify periods with increased media coverage over broader time spans."
    
    # Translate if needed and remove any '#' characters
    if language.lower() != "english":
        timeline_header = translate_content(timeline_header, "English", language).replace("#", "")
    timeline_md = f"""
## Publication Timeline
{timeline_header}

![Publication Timeline](data:image/png;base64,{publication_timeline_chart})
"""
    # Group by week for peak analysis
    weekly = df.groupby(pd.Grouper(key='date', freq='W')).size().reset_index(name='articles')
    top_peaks = weekly.nlargest(2, 'articles')
    
    # Instantiate ChatGPT for peak analysis (ensure your ChatGPT class is properly defined)
    chatbot = ChatGPT(
        system_prompt="You are a helpful assistant analyzing media coverage peaks.",
        model_name="chatgpt-4o-latest",
        temperature=0,
        max_tokens=1500,
    )
    
    peaks_analysis_md = "\n### Analysis of Coverage Peaks\n"
    for _, row in top_peaks.iterrows():
        week_date = row['date']
        num_articles = row['articles']
        # Filter articles for the week (using a week range)
        week_start = week_date - pd.Timedelta(days=6)
        articles_in_week = df[(df['date'] >= week_start) & (df['date'] <= week_date)]
        combined_text = "\n\n".join(articles_in_week['content'].tolist())
        excerpt = combined_text[:1000]  # Limit the excerpt to 1000 characters
        
        prompt = f"""
During the week ending {week_date.strftime('%d %B %Y')}, there was a significant spike in media coverage for {company_name} with {num_articles} articles published.
Below are excerpts from some of the articles published during this week:
{excerpt}

Based on these excerpts, please explain what events, topics, or issues might have contributed to this spike in coverage. Your explanation should help a reader understand why media attention for {company_name} increased during this period.
        """
        try:
            explanation = chatbot.ask(prompt)
        except Exception as e:
            explanation = f"Error generating explanation: {str(e)}"
        peaks_analysis_md += f"\n**Week Ending {week_date.strftime('%d %B %Y')} (Articles: {num_articles}):**\n\n{explanation}\n"
        if language.lower() != "english":
            peaks_analysis_md = translate_content(peaks_analysis_md, "auto", language)
    
    return timeline_md + peaks_analysis_md