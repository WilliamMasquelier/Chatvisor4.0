import functools
import logging
import time
import os
import inspect
from datetime import datetime

class TokenEstimator:
    """Utility class to estimate token counts based on text length."""
    
    @staticmethod
    def estimate_tokens(text):
        """Estimate the number of tokens in a text string.
        
        This is a rough approximation based on GPT tokenization statistics.
        On average, 1 token is approximately 4 characters of English text.
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        if not text:
            return 0
            
        # Simple estimate: ~4 characters per token for English text
        return len(text) // 4

class TrackerIntegration:
    """Class to manage integration of the program summary tracker with existing classes."""
    
    def __init__(self, tracker):
        """Initialize with a ProgramSummaryTracker instance."""
        self.tracker = tracker
        self.logger = logging.getLogger("TrackerIntegration")
    
    def patch_chatgpt_class(self):
        """
        Dynamically patch the ChatGPT class to track metrics without modifying the source code.
        """
        try:
            from Classes.SimplifiedChatbots import ChatGPT

            # Store the original ask method
            original_ask = ChatGPT.ask

            # Store a reference to the tracker
            tracker = self.tracker

            # Create a wrapper that will track metrics
            @functools.wraps(original_ask)
            def tracked_ask(self, question):
                start_time = time.time()

                # Call the original method with the correct signature
                response = original_ask(self, question)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Estimate token counts
                system_prompt_tokens = TokenEstimator.estimate_tokens(getattr(self, "system_prompt", ""))
                question_tokens = TokenEstimator.estimate_tokens(question)
                response_tokens = TokenEstimator.estimate_tokens(response)

                # Get calling function name if available
                import inspect
                calling_frame = inspect.currentframe().f_back
                if calling_frame:
                    calling_function = calling_frame.f_code.co_name
                else:
                    calling_function = "unknown"

                # Use the tracker from the closure
                tracker.record_chatbot_call(
                    model_name=getattr(self, "model_name", "unknown"),
                    system_prompt_length=system_prompt_tokens,
                    prompt_length=question_tokens,
                    response_length=response_tokens,
                    execution_time=execution_time,
                    function_name=calling_function
                )

                return response

            # Replace the original method with our tracked version
            ChatGPT.ask = tracked_ask

            self.logger.info("ChatGPT class patched successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to patch ChatGPT class: {str(e)}")
            return False

    def patch_big_summarizer_gpt_class(self):
        """
        Dynamically patch the BigSummarizerGPT class to track metrics.
        """
        try:
            from Classes.SimplifiedChatbots import BigSummarizerGPT

            # Store the original ask method
            original_ask = BigSummarizerGPT.ask

            # Store a reference to the tracker
            tracker = self.tracker

            # Create a wrapper that will track metrics
            @functools.wraps(original_ask)
            def tracked_ask(self, question, document_path, max_chunk_size=20000, dict_max_token=15000, max_retries=3, retry_delay=2):
                start_time = time.time()

                # Call the original method with the correct signature and all parameters
                response = original_ask(self, question, document_path, max_chunk_size, dict_max_token, max_retries, retry_delay)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Estimate token counts
                system_prompt_tokens = TokenEstimator.estimate_tokens(getattr(self, "system_prompt", ""))
                question_tokens = TokenEstimator.estimate_tokens(question)
                response_tokens = TokenEstimator.estimate_tokens(response)

                # Add document tokens if applicable
                doc_tokens = 0
                if document_path and os.path.exists(document_path):
                    try:
                        # Estimate tokens based on file size
                        file_size = os.path.getsize(document_path)
                        # Rough approximation: 1KB ≈ 200 tokens
                        doc_tokens = file_size / 5
                    except Exception as e:
                        self.logger.warning(f"Could not estimate document tokens: {str(e)}")

                # Get calling function name if available
                import inspect
                calling_frame = inspect.currentframe().f_back
                if calling_frame:
                    calling_function = calling_frame.f_code.co_name
                else:
                    calling_function = "unknown"

                # Use the tracker from the closure
                tracker.record_chatbot_call(
                    model_name=getattr(self, "model_name", "unknown"),
                    system_prompt_length=system_prompt_tokens,
                    prompt_length=question_tokens + doc_tokens,
                    response_length=response_tokens,
                    execution_time=execution_time,
                    function_name=calling_function
                )

                return response

            # Replace the original method with our tracked version
            BigSummarizerGPT.ask = tracked_ask

            self.logger.info("BigSummarizerGPT class patched successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to patch BigSummarizerGPT class: {str(e)}")
            return False
    
    def patch_visualization_functions(self):
        """
        Patch visualization functions to track chart generation metrics.
        """
        try:
            # Import Helpers module where visualization functions are defined
            import Utils.Helpers as Helpers
            
            # List of visualization function names to patch
            visualization_functions = [
                'create_bar_chart_compiled_insights',
                'create_professional_pie',
                'create_multiple_pie_charts',
                'create_stacked_bar_chart',
                'generate_media_outlet_pie_chart',
                'generate_media_outlet_tone_chart',
                'generate_overall_sentiment_trend',
                'generate_sentiment_trends_by_category',
                'generate_articles_per_category',
                'generate_category_tone_chart',
                'generate_top_journalists_chart'
            ]
            
            patched_count = 0
            
            # Create wrapper for visualization functions
            def create_tracked_visualization_function(original_func):
                @functools.wraps(original_func)
                def tracked_func(*args, **kwargs):
                    start_time = time.time()
                    
                    # Call original function
                    result = original_func(*args, **kwargs)
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Record the operation
                    self.tracker.record_operation(
                        operation_name="chart_generation",
                        details={"function": original_func.__name__},
                        execution_time=execution_time
                    )
                    
                    return result
                
                return tracked_func
            
            # Patch each visualization function
            for func_name in visualization_functions:
                if hasattr(Helpers, func_name):
                    original_func = getattr(Helpers, func_name)
                    setattr(Helpers, func_name, create_tracked_visualization_function(original_func))
                    patched_count += 1
            
            self.logger.info(f"Patched {patched_count} visualization functions")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to patch visualization functions: {str(e)}")
            return False
    
    def patch_document_processor(self):
        """
        Patch DocumentProcessor to track document processing metrics.
        """
        try:
            from Classes.DocumentProcessor import DocumentProcessor
            
            # Store original process_documents method
            original_process = DocumentProcessor.process_documents
            
            # Store a reference to the tracker
            tracker = self.tracker
            
            @functools.wraps(original_process)
            def tracked_process_documents(self, pdf_folder_path=None, docx_file_path=None, docx_separator="--"):
                start_time = time.time()
                
                # Call the original method
                articles = original_process(self, pdf_folder_path, docx_file_path, docx_separator)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Use the tracker from the closure
                tracker.record_operation(
                    operation_name="process_article",
                    details={"article_count": len(articles) if articles else 0},
                    execution_time=execution_time
                )
                
                # Update article count
                tracker.articles_processed += len(articles) if articles else 0
                
                return articles
            
            # Replace the original method with our tracked version
            DocumentProcessor.process_documents = tracked_process_documents
            
            self.logger.info("DocumentProcessor patched successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to patch DocumentProcessor: {str(e)}")
            return False
    
    def patch_all(self):
        """
        Patch all classes and functions with tracking capabilities.
        
        Returns:
            bool: True if all patches succeeded, False otherwise
        """
        # Track patch successes
        patches = [
            self.patch_chatgpt_class(),
            self.patch_big_summarizer_gpt_class(),
            self.patch_visualization_functions(),
            self.patch_document_processor()
        ]
        
        # Return True only if all patches succeeded
        return all(patches)