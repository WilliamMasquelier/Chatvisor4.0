import os
import json
import time
import logging
import traceback
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Classes.TrackerIntegration import TrackerIntegration

class ProgramSummaryTracker:
    """
    A class to track and report on the execution metrics of the media analysis program.
    Records chatbot usage, token counts, execution times, and other relevant metrics.
    """
    
    def __init__(self, output_dir=None, entity_name=None, entity_type=None):
        """
        Initialize the tracker with an output directory for reports.
        
        Args:
            output_dir (str, optional): Base directory where summary reports will be saved.
                                        If None, uses the current working directory.
            entity_name (str, optional): Name of the journalist or company being analyzed.
            entity_type (str, optional): Type of analysis - "journalist" or "company".
        """
        self.start_time = time.time()
        self.output_dir = output_dir or os.getcwd()
        self.entity_name = entity_name
        self.entity_type = entity_type
        
        # Create the summary directory in the appropriate location
        if entity_name and entity_type:
            # For journalist or company analysis, use the entity-specific directory
            if entity_type.lower() == "journalist":
                self.summary_dir = os.path.join(self.output_dir, "KnowledgeBase", "JournalistProfile", 
                                              entity_name, "ProgramSummaries")
            else:  # company analysis
                self.summary_dir = os.path.join(self.output_dir, "KnowledgeBase", "CompanyProfile", 
                                              entity_name, "ProgramSummaries")
        else:
            # If no entity specified, create in the base output directory
            self.summary_dir = os.path.join(self.output_dir, "ProgramSummaries")
        
        # Create the directory if it doesn't exist
        os.makedirs(self.summary_dir, exist_ok=True)
        
        # Initialize counters and trackers
        self.chatbot_calls = []
        self.model_usage = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "execution_time": 0})
        self.function_calls = defaultdict(lambda: {"calls": 0, "execution_time": 0})
        self.articles_processed = 0
        self.entities_extracted = {"organizations": 0, "people": 0}
        self.operations_performed = []
        self.sentiment_analyses = 0
        self.charts_generated = 0
        
        # Set up logging for the tracker
        self.logger = logging.getLogger("ProgramSummaryTracker")
        self.logger.setLevel(logging.INFO)
        
        # Add a file handler for the tracker's logs
        log_path = os.path.join(self.summary_dir, "tracker.log")
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Program Summary Tracker initialized. Saving reports to: {self.summary_dir}")
        
        # Apply patches to track usage
        self.integration = TrackerIntegration(self)
        success = self.integration.patch_all()
        if success:
            self.logger.info("All classes and functions patched successfully")
        else:
            self.logger.warning("Some patches failed to apply. See logs for details.")
        
    def record_chatbot_call(self, model_name, system_prompt_length=0, prompt_length=0, 
                           response_length=0, execution_time=0, function_name=None):
        """
        Record a chatbot API call with relevant metrics.
        
        Args:
            model_name (str): Name of the model used (e.g., "gpt-4o-latest")
            system_prompt_length (int): Length of the system prompt in tokens
            prompt_length (int): Length of the user prompt in tokens
            response_length (int): Length of the model response in tokens
            execution_time (float): Time taken for the API call
            function_name (str): Name of the function making the call
        """
        # Record the call details
        call_info = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "system_prompt_tokens": system_prompt_length,
            "prompt_tokens": prompt_length,
            "response_tokens": response_length,
            "total_input_tokens": system_prompt_length + prompt_length,
            "execution_time": execution_time,
            "function": function_name
        }
        
        self.chatbot_calls.append(call_info)
        
        # Update model-specific counters
        self.model_usage[model_name]["calls"] += 1
        self.model_usage[model_name]["input_tokens"] += system_prompt_length + prompt_length
        self.model_usage[model_name]["output_tokens"] += response_length
        self.model_usage[model_name]["execution_time"] += execution_time
        
        if function_name:
            self.function_calls[function_name]["calls"] += 1
            self.function_calls[function_name]["execution_time"] += execution_time
        
        self.logger.info(f"Recorded chatbot call: {model_name}, tokens: {system_prompt_length + prompt_length} in, {response_length} out")
        
    def record_operation(self, operation_name, details=None, execution_time=0):
        """
        Record a program operation with relevant details.
        
        Args:
            operation_name (str): Name of the operation (e.g., "generate_insights")
            details (dict): Additional details about the operation
            execution_time (float): Time taken to execute the operation
        """
        operation_info = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "details": details or {},
            "execution_time": execution_time
        }
        
        self.operations_performed.append(operation_info)
        
        if operation_name == "sentiment_analysis":
            self.sentiment_analyses += 1
        elif operation_name == "chart_generation":
            self.charts_generated += 1
        elif operation_name == "process_article":
            if details and "article_count" in details:
                self.articles_processed += details["article_count"]
        elif operation_name == "extract_entities":
            if details:
                if "organizations" in details:
                    self.entities_extracted["organizations"] += details.get("organizations", 0)
                if "people" in details:
                    self.entities_extracted["people"] += details.get("people", 0)
        
        self.logger.info(f"Recorded operation: {operation_name}")
        
    def _create_model_usage_chart(self, timestamp):
        """
        Create a chart showing model usage statistics.
        
        Args:
            timestamp (str): Timestamp for file naming
            
        Returns:
            str or None: Base filename of the chart if created, None otherwise
        """
        try:
            # Prepare models usage data for visualization
            models_data = []
            for model, stats in self.model_usage.items():
                models_data.append({
                    "model": model,
                    "calls": stats["calls"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "total_tokens": stats["input_tokens"] + stats["output_tokens"]
                })
            
            if not models_data:
                return None
                
            # Create DataFrame and sort
            models_df = pd.DataFrame(models_data)
            if models_df.empty:
                return None
            
            models_df = models_df.sort_values("total_tokens", ascending=False)
            
            # Create the chart
            plt.figure(figsize=(12, 6))
            
            # Create bars for input and output tokens
            x = np.arange(len(models_df))
            width = 0.35
            
            plt.bar(x - width/2, models_df["input_tokens"], width, label="Input Tokens")
            plt.bar(x + width/2, models_df["output_tokens"], width, label="Output Tokens")
            
            plt.xlabel("Model")
            plt.ylabel("Tokens")
            plt.title("Token Usage by Model")
            plt.xticks(x, models_df["model"], rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            
            # Save chart
            chart_filename = f"model_usage_{timestamp}.png"
            chart_path = os.path.join(self.summary_dir, chart_filename)
            plt.savefig(chart_path)
            plt.close()
            
            return chart_filename
            
        except Exception as e:
            self.logger.error(f"Error creating model usage chart: {str(e)}")
            return None
    
    def _create_function_calls_chart(self, timestamp):
        """
        Create a chart showing function call statistics.
        
        Args:
            timestamp (str): Timestamp for file naming
            
        Returns:
            str or None: Base filename of the chart if created, None otherwise
        """
        try:
            if not self.function_calls:
                return None
                
            # Create DataFrame for function calls
            functions_df = pd.DataFrame([
                {"function": func, "calls": stats["calls"]}
                for func, stats in self.function_calls.items()
            ])
            
            if functions_df.empty:
                return None
                
            # Sort and take top 15
            functions_df = functions_df.sort_values("calls", ascending=False).head(15)
            
            # Create the chart
            plt.figure(figsize=(12, 6))
            plt.bar(functions_df["function"], functions_df["calls"], color='darkblue')
            plt.xlabel("Function")
            plt.ylabel("Number of Calls")
            plt.title("Top Functions by Chatbot Usage")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Save chart
            chart_filename = f"function_calls_{timestamp}.png"
            chart_path = os.path.join(self.summary_dir, chart_filename)
            plt.savefig(chart_path)
            plt.close()
            
            return chart_filename
            
        except Exception as e:
            self.logger.error(f"Error creating function calls chart: {str(e)}")
            return None
    
    def _create_operations_timeline_chart(self, timestamp):
        """
        Create a chart showing operations over time.
        
        Args:
            timestamp (str): Timestamp for file naming
            
        Returns:
            str or None: Base filename of the chart if created, None otherwise
        """
        try:
            if not self.operations_performed:
                return None
                
            # Group operations by type
            op_types = [op["operation"] for op in self.operations_performed]
            op_counts = Counter(op_types)
            
            # Create the chart
            plt.figure(figsize=(12, 6))
            plt.bar(op_counts.keys(), op_counts.values(), color='darkgreen')
            plt.xlabel("Operation Type")
            plt.ylabel("Count")
            plt.title("Operations Performed")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Save chart
            chart_filename = f"operations_{timestamp}.png"
            chart_path = os.path.join(self.summary_dir, chart_filename)
            plt.savefig(chart_path)
            plt.close()
            
            return chart_filename
            
        except Exception as e:
            self.logger.error(f"Error creating operations timeline chart: {str(e)}")
            return None
    
    def generate_report(self):
        """
        Generate a comprehensive report of program execution metrics.
        
        Returns:
            str: Path to the generated report file
        """
        try:
            # Calculate execution time
            execution_time = time.time() - self.start_time
            
            # Create report filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"ProgramSummary_{timestamp}.md"
            report_path = os.path.join(self.summary_dir, report_filename)
            
            # Calculate summary metrics
            total_chatbot_calls = len(self.chatbot_calls)
            total_input_tokens = sum(call["total_input_tokens"] for call in self.chatbot_calls)
            total_output_tokens = sum(call["response_tokens"] for call in self.chatbot_calls)
            total_tokens = total_input_tokens + total_output_tokens
            
            # Generate charts
            model_chart = self._create_model_usage_chart(timestamp)
            function_chart = self._create_function_calls_chart(timestamp)
            operations_chart = self._create_operations_timeline_chart(timestamp)
            
            # Generate the markdown report
            with open(report_path, "w", encoding="utf-8") as report_file:
                report_file.write(f"# Media Analysis Program Execution Summary\n\n")
                report_file.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                report_file.write("## Overview\n\n")
                report_file.write(f"* **Total Execution Time:** {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n")
                report_file.write(f"* **Articles Processed:** {self.articles_processed}\n")
                report_file.write(f"* **Charts Generated:** {self.charts_generated}\n")
                report_file.write(f"* **Sentiment Analyses Performed:** {self.sentiment_analyses}\n")
                report_file.write(f"* **Entities Extracted:** {sum(self.entities_extracted.values())} (Organizations: {self.entities_extracted['organizations']}, People: {self.entities_extracted['people']})\n\n")
                
                report_file.write("## AI Model Usage\n\n")
                report_file.write(f"* **Total AI Chatbot Calls:** {total_chatbot_calls}\n")
                report_file.write(f"* **Total Input Tokens:** {total_input_tokens:,}\n")
                report_file.write(f"* **Total Output Tokens:** {total_output_tokens:,}\n")
                report_file.write(f"* **Total Tokens Processed:** {total_tokens:,}\n\n")
                
                # Add estimated cost analysis (using approximate costs)
                gpt4_input_tokens = sum(call["total_input_tokens"] for call in self.chatbot_calls if "gpt-4" in call["model"].lower())
                gpt4_output_tokens = sum(call["response_tokens"] for call in self.chatbot_calls if "gpt-4" in call["model"].lower())
                gpt35_input_tokens = sum(call["total_input_tokens"] for call in self.chatbot_calls if "gpt-3.5" in call["model"].lower())
                gpt35_output_tokens = sum(call["response_tokens"] for call in self.chatbot_calls if "gpt-3.5" in call["model"].lower())
                
                # Approximate costs per 1000 tokens
                gpt4_input_cost = 0.03  # $0.03 per 1000 tokens
                gpt4_output_cost = 0.06  # $0.06 per 1000 tokens
                gpt35_input_cost = 0.0015  # $0.0015 per 1000 tokens
                gpt35_output_cost = 0.002  # $0.002 per 1000 tokens
                
                estimated_cost = (
                    (gpt4_input_tokens / 1000 * gpt4_input_cost) +
                    (gpt4_output_tokens / 1000 * gpt4_output_cost) +
                    (gpt35_input_tokens / 1000 * gpt35_input_cost) +
                    (gpt35_output_tokens / 1000 * gpt35_output_cost)
                )
                
                report_file.write("### Estimated API Costs\n\n")
                report_file.write(f"* **Estimated Total Cost:** ${estimated_cost:.2f}\n")
                report_file.write(f"* **GPT-4 Series Models:** ${((gpt4_input_tokens / 1000 * gpt4_input_cost) + (gpt4_output_tokens / 1000 * gpt4_output_cost)):.2f}\n")
                report_file.write(f"* **GPT-3.5 Series Models:** ${((gpt35_input_tokens / 1000 * gpt35_input_cost) + (gpt35_output_tokens / 1000 * gpt35_output_cost)):.2f}\n\n")
                
                if model_chart:
                    report_file.write("### Model-Specific Usage\n\n")
                    
                    # Create a markdown table for model usage
                    report_file.write("| Model | Calls | Input Tokens | Output Tokens | Total Tokens |\n")
                    report_file.write("|-------|-------|--------------|---------------|-------------|\n")
                    
                    for model, stats in sorted(self.model_usage.items(), key=lambda x: x[1]["input_tokens"] + x[1]["output_tokens"], reverse=True):
                        report_file.write(f"| {model} | {stats['calls']} | {stats['input_tokens']:,} | {stats['output_tokens']:,} | {(stats['input_tokens'] + stats['output_tokens']):,} |\n")
                    
                    report_file.write(f"\n![Model Token Usage]({model_chart})\n\n")
                
                if function_chart:
                    report_file.write("## Function Analysis\n\n")
                    
                    # Create a markdown table for top functions
                    sorted_functions = sorted(
                        [(func, stats["calls"]) for func, stats in self.function_calls.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    report_file.write("### Top 15 Functions by Chatbot Usage\n\n")
                    report_file.write("| Function | Chatbot Calls |\n")
                    report_file.write("|----------|---------------|\n")
                    
                    for func, calls in sorted_functions[:15]:
                        report_file.write(f"| {func} | {calls} |\n")
                    
                    report_file.write(f"\n![Function Calls]({function_chart})\n\n")
                
                if operations_chart:
                    report_file.write("## Operations Analysis\n\n")
                    report_file.write(f"![Operations Performed]({operations_chart})\n\n")
                
                report_file.write("## Operation Timeline\n\n")
                report_file.write("| Timestamp | Operation | Execution Time (s) |\n")
                report_file.write("|-----------|-----------|-------------------|\n")
                
                # Get the last 20 operations (or fewer if we don't have that many)
                recent_ops = self.operations_performed[-20:] if len(self.operations_performed) > 20 else self.operations_performed
                
                for op in recent_ops:
                    ts = datetime.fromisoformat(op["timestamp"]).strftime("%H:%M:%S")
                    report_file.write(f"| {ts} | {op['operation']} | {op.get('execution_time', 0):.2f} |\n")
                
                report_file.write("\n## Performance Analysis\n\n")
                
                # Calculate average response times by model
                report_file.write("### Average Response Times by Model\n\n")
                report_file.write("| Model | Average Response Time (s) |\n")
                report_file.write("|-------|--------------------------|\n")
                
                for model, stats in sorted(self.model_usage.items(), key=lambda x: x[0]):
                    avg_time = stats["execution_time"] / stats["calls"] if stats["calls"] > 0 else 0
                    report_file.write(f"| {model} | {avg_time:.2f} |\n")
                
                report_file.write("\n## Additional Notes\n\n")
                report_file.write("* This report provides a summary of the program execution metrics and resource usage.\n")
                report_file.write("* Token counts are estimated based on standard approximations.\n")
                report_file.write("* For detailed logs, refer to the tracker.log file in the ProgramSummaries directory.\n")
                report_file.write("* Estimated costs are based on approximate OpenAI pricing and may not reflect actual billing.\n")
                report_file.write("* Performance figures are measured on this specific run and may vary based on system load and network conditions.\n")
            
            self.logger.info(f"Generated summary report: {report_path}")
            return report_path
    
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
        
    def save_raw_data(self):
        """
        Save the raw tracking data to a JSON file for later analysis.
        
        Returns:
            str: Path to the saved JSON file
        """
        try:
            # Create data filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"ProgramSummaryData_{timestamp}.json"
            data_path = os.path.join(self.summary_dir, data_filename)
            
            # Prepare data for serialization
            data = {
                "execution_time": time.time() - self.start_time,
                "chatbot_calls": self.chatbot_calls,
                "model_usage": {model: dict(stats) for model, stats in self.model_usage.items()},
                "function_calls": dict(self.function_calls),
                "articles_processed": self.articles_processed,
                "entities_extracted": self.entities_extracted,
                "operations_performed": self.operations_performed,
                "sentiment_analyses": self.sentiment_analyses,
                "charts_generated": self.charts_generated
            }
            
            # Save to JSON
            with open(data_path, "w", encoding="utf-8") as data_file:
                json.dump(data, data_file, indent=4)
            
            self.logger.info(f"Saved raw tracking data: {data_path}")
            return data_path
            
        except Exception as e:
            self.logger.error(f"Error saving raw data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None