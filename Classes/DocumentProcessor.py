import os
import pandas as pd
from difflib import SequenceMatcher
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from difflib import SequenceMatcher
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict
import re
import logging
from dataclasses import dataclass
from dateutil import parser
import unicodedata

# Class to process document inputs
class DocumentProcessor:
    def __init__(self, min_length: int = 1000, max_length: int = 25500, similarity_threshold: float = 0.8):
        """
        Initialize the document processor with configurable parameters.
        
        Args:
            min_length (int): Minimum character length for valid documents
            max_length (int): Maximum character length for valid documents
            similarity_threshold (float): Threshold for detecting duplicate content
        """
        self.min_length = min_length
        self.max_length = max_length
        self.similarity_threshold = similarity_threshold
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts"""
        return SequenceMatcher(None, text1.strip(), text2.strip()).ratio()

    def _get_files(self, folder_path: str) -> List[str]:
        """Get list of files from folder, excluding system files"""
        if not os.path.exists(folder_path):
            self.logger.error(f"Folder path does not exist: {folder_path}")
            return []
        
        files = [f"{folder_path}/{file}" for file in os.listdir(folder_path)]
        return [f for f in files if f.lower().endswith('.pdf')]

    def _extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """Extract text content from a PDF file using PyPDF2"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return None

    def _extract_text_from_docx(self, file_path: str) -> Optional[str]:
        """Extract text content from a DOCX file using python-docx"""
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return None

    def process_pdf_folder(self, folder_path: str) -> List[Dict]:
        """Process all PDFs in a folder."""
        if not os.path.exists(folder_path):
            self.logger.error(f"PDF folder path does not exist: {folder_path}")
            return []

        files = self._get_files(folder_path)
        self.logger.info(f"Found {len(files)} files in folder")

        articles = [{'file_path': file, 'position': idx + 1} for idx, file in enumerate(files)]

        # Process files using PyPDF2
        processed_documents = []
        for file in files:
            try:
                pdf_text = self._extract_text_from_pdf(file)

                if pdf_text is None or not self.min_length <= len(pdf_text) <= self.max_length:
                    self.logger.info(
                        f"Removing file {file} (text extraction failed or length out of range)"
                    )
                    continue

                pdf_text = pdf_text.replace("William Masquelier", " ")  # Optional cleanup
                processed_documents.append({'file_path': file, 'content': pdf_text})
            except Exception as e:
                self.logger.error(f"Error processing PDF {file}: {e}")

        self.logger.info(f"Processed {len(processed_documents)} valid PDF files")
        return processed_documents

    def process_docx(self, docx_path: str, separator: str = "--") -> List[Dict]:
        """
        Process a DOCX file containing multiple articles.
        
        Args:
            docx_path (str): Path to the DOCX file
            separator (str): Separator used between articles
            
        Returns:
            List[Dict]: List of processed articles
        """
        if not os.path.exists(docx_path):
            self.logger.error(f"DOCX file does not exist: {docx_path}")
            return []

        try:
            # Extract text from DOCX
            docx_text = self._extract_text_from_docx(docx_path)
            if not docx_text:
                self.logger.error(f"Failed to extract text from DOCX {docx_path}")
                return []

            # Split into articles based on separator
            raw_sections = docx_text.split(separator)
            articles = []

            for idx, section in enumerate(raw_sections, 1):
                content = section.strip()
                if not self.min_length <= len(content) <= self.max_length:
                    continue

                is_duplicate = any(
                    self._text_similarity(content, existing['content']) > self.similarity_threshold 
                    for existing in articles
                )

                if not is_duplicate:
                    articles.append({
                        'file_path': docx_path,
                        'content': content,
                        'position': idx
                    })

            self.logger.info(f"Processed {len(articles)} articles from DOCX")
            return articles

        except Exception as e:
            self.logger.error(f"Error processing DOCX {docx_path}: {e}")
            return []

    def process_documents(self, 
                         pdf_folder_path: Optional[str] = None, 
                         docx_file_path: Optional[str] = None, 
                         docx_separator: str = "--") -> List[Dict]:
        """
        Main processing function that handles both PDF folder and DOCX file inputs.
        
        Args:
            pdf_folder_path (str, optional): Path to folder containing PDFs
            docx_file_path (str, optional): Path to DOCX file
            docx_separator (str): Separator for DOCX processing
            
        Returns:
            List[Dict]: Combined list of processed articles from both sources
        """
        all_articles = []
        position = 1

        if pdf_folder_path:
            pdf_articles = self.process_pdf_folder(pdf_folder_path)
            for article in pdf_articles:
                article['reordered_position'] = position
                position += 1
            all_articles.extend(pdf_articles)

        if docx_file_path:
            docx_articles = self.process_docx(docx_file_path, docx_separator)
            for article in docx_articles:
                article['reordered_position'] = position
                position += 1
            all_articles.extend(docx_articles)

        self.logger.info(f"Total processed articles: {len(all_articles)}")
        return all_articles

@dataclass
class Citation:
    media_outlet: str
    author: Optional[str]
    date: str
    title: Optional[str] = None
    link: Optional[str] = None
    
    def __eq__(self, other):
        if not isinstance(other, Citation):
            return False
        return (self.media_outlet.lower() == other.media_outlet.lower() and
                self.author == other.author and
                self.date == other.date)
    
    def __hash__(self):
        return hash((self.media_outlet.lower(), self.author, self.date))

class CitationProcessor:
    def __init__(self):
        # Months in various languages (English, French, Spanish, Dutch, Italian, German)
        self.months = {
            'january|janvier|enero|januari|gennaio|januar',
            'february|février|febrero|februari|febbraio|februar',
            'march|mars|marzo|maart|marzo|märz',
            'april|avril|abril|april|aprile|april',
            'may|mai|mayo|mei|maggio|mai',
            'june|juin|junio|juni|giugno|juni',
            'july|juillet|julio|juli|luglio|juli',
            'august|août|agosto|augustus|agosto|august',
            'september|septembre|septiembre|september|settembre|september',
            'october|octobre|octubre|oktober|ottobre|oktober',
            'november|novembre|noviembre|november|novembre|november',
            'december|décembre|diciembre|december|dicembre|dezember'
        }
        self.month_pattern = '|'.join(self.months)
        
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing accents and converting to lowercase."""
        return unicodedata.normalize('NFKD', text.lower()).encode('ASCII', 'ignore').decode('utf-8')

    def is_valid_date(self, text: str) -> bool:
        """Check if text contains a valid date format in multiple languages."""
        # Normalize text for matching
        norm_text = self.normalize_text(text)
        
        # Check for month names in various languages
        if any(re.search(rf'\b{month}\b', norm_text) for month in self.month_pattern.split('|')):
            return True
            
        # Try parsing with dateutil as fallback
        try:
            parser.parse(text)
            return True
        except:
            return False

    def is_valid_citation(self, text: str) -> bool:
        """
        Determine if bracketed text is a valid citation.
        """
        # Common exclusion patterns
        if any(pattern in text for pattern in ['![', '[#', 'http', '.md', '.pdf']):
            return False
            
        # Split parts and check format
        parts = [p.strip() for p in text.split(',')]
        
        # Must have at least media outlet and date
        if len(parts) < 2:
            return False
            
        # Last part should be a date
        if not self.is_valid_date(parts[-1]):
            return False
            
        return True

    def parse_citation(self, citation_text: str) -> Optional[Citation]:
        """
        Parse citation text into structured Citation object.
        """
        try:
            parts = [p.strip() for p in citation_text.split(',')]
            
            if len(parts) == 2:  # [Media outlet, date]
                return Citation(
                    media_outlet=parts[0],
                    author=None,
                    date=parts[1]
                )
            elif len(parts) >= 3:  # [Media outlet, Author, date]
                return Citation(
                    media_outlet=parts[0],
                    author=parts[1],
                    date=parts[2]
                )
            return None
        except Exception as e:
            logging.debug(f"Error parsing citation: {str(e)}")
            return None

    def process_citations_in_markdown(self, markdown_content: str) -> str:
        """
        Process citations in markdown content and generate bibliography.
        """
        try:
            references = OrderedDict()
            
            def process_citation_match(match):
                full_match = match.group(0)
                citation_text = full_match.strip('[]')
                
                # Check if this is a valid citation
                if not self.is_valid_citation(citation_text):
                    return full_match
                    
                # Handle multiple citations separated by semicolon
                citations = [c.strip() for c in citation_text.split(';')]
                ref_numbers = []
                
                for citation in citations:
                    parsed_citation = self.parse_citation(citation)
                    if parsed_citation:
                        if citation not in references:
                            references[citation] = len(references) + 1
                        ref_numbers.append(str(references[citation]))
                
                if not ref_numbers:
                    return full_match
                    
                # Create reference string
                ref_string = ','.join(ref_numbers)
                return f'<sup style="line-height: 0; vertical-align: super; font-size: smaller;">' \
                       f'<a href="#ref-{ref_string}" style="text-decoration: none;">[{ref_string}]</a></sup>'
            
            # Process citations
            citation_pattern = r'\[[^\]]+?\]'
            processed_content = re.sub(citation_pattern, process_citation_match, markdown_content)
            
            # Add references section
            if references:
                processed_content += "\n\n---\n\n## References\n\n"
                for citation, ref_num in references.items():
                    processed_content += f'<div id="ref-{ref_num}" style="margin-bottom: 0.5em;">' \
                                      f'{ref_num}. {citation}</div>\n'
            
            return processed_content
            
        except Exception as e:
            logging.error(f"Error processing citations: {str(e)}")
            return markdown_content

