import argparse
import gradio as gr
from Classes.DocumentProcessor import DocumentProcessor, CitationProcessor
from Classes.ProgramSummaryTracker import ProgramSummaryTracker
from Utils.Helpers import *
from Utils.Outputs import *
import logging
import traceback
import logging
import json

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def gradio_journalist_list(company_name, file_folder, docx_file_path, industry_of_interest, 
                           region, journalist_name, language, force_reprocess,
                           generate_journalist_list, generate_insights, 
                           generate_analysis, generate_topics, 
                           generate_analytics, generate_quotes,
                           generate_consolidated_quotes, generate_journalist_article_list, 
                           generate_journalist_coverage_profile,
                           generate_topic_analysis, generate_journalist_analysis,  # NEW parameter
                           topic_focus=None):
    try:
        # Set up base folder and logging (unchanged)
        if file_folder:
            base_folder = os.path.dirname(os.path.dirname(file_folder))
        elif docx_file_path:
            base_folder = os.path.dirname(os.path.dirname(docx_file_path))
        else:
            base_folder = os.getcwd()
        
        setup_logging(base_folder)
        
        # Determine if this is a journalist analysis (based on provided journalist_name and options)
        is_journalist_analysis = bool(journalist_name and (generate_journalist_coverage_profile or generate_topic_analysis or generate_journalist_article_list or generate_journalist_analysis))

        # Initialize the program summary tracker with the appropriate entity info
        if is_journalist_analysis:
            tracker = ProgramSummaryTracker(
                output_dir=base_folder,
                entity_name=journalist_name,
                entity_type="journalist"
            )
        else:
            tracker = ProgramSummaryTracker(
                output_dir=base_folder,
                entity_name=company_name,
                entity_type="company"
            )

        # Validate input paths, process documents, etc.
        pdf_folder = file_folder.strip() if file_folder else None
        docx_path = docx_file_path.strip() if docx_file_path else None
        
        if not check_input_paths(pdf_folder, docx_path):
            error_msg = "No valid input paths provided. Please check your folder/file paths."
            return [error_msg] * 11  # Now 11 outputs

        processor = DocumentProcessor(
            min_length=1000,
            max_length=25500,
            similarity_threshold=0.8
        )

        articles = processor.process_documents(
            pdf_folder_path=file_folder if file_folder.strip() else None,
            docx_file_path=docx_file_path if docx_file_path.strip() else None,
            docx_separator="--"
        )

        if not articles:
            error_msg = "No valid articles found after processing documents"
            return [error_msg] * 11

        if is_journalist_analysis:
            logging.info(f"Starting journalist analysis for {journalist_name}")
            articles_sorted, general_folder, success = preprocess_journalist_articles(
                journalist_name=journalist_name,
                articles=articles,
                news_folder_path=file_folder,
                force_reprocess=force_reprocess
            )
            if not success:
                error_msg = "Failed to preprocess journalist articles"
                return [error_msg] * 11

            # For journalist analysis mode, we keep some default placeholder outputs for the media analysis slots.
            results = ["Journalist analysis mode"] * 7  # first 7 outputs (placeholders)

            journalist_profile_result = "Journalist profile not requested"
            topic_analysis_result = "Topic analysis not requested"
            article_list_result = "Articles list not requested"
            journalist_analysis_result = "Journalist analysis not requested"  # NEW output

            # Generate article list if requested
            if generate_journalist_article_list:
                try:
                    logging.info("Generating journalist article list")
                    from Utils.Outputs import generate_journalist_article_list
                    article_list_result = generate_journalist_article_list(
                        articles_sorted=articles_sorted,
                        journalist_name=journalist_name,
                        general_folder=general_folder,
                        language=language
                    )
                except Exception as e:
                    logging.error(f"Error generating journalist article list: {str(e)}")
                    article_list_result = f"Error generating article list: {str(e)}"

            # Generate journalist profile if requested
            if generate_journalist_coverage_profile:
                try:
                    logging.info("Generating journalist profile")
                    journalist_profile_result = generate_journalist_profile(
                        articles_sorted=articles_sorted,
                        journalist_name=journalist_name,
                        news_folder_path=file_folder,
                        language=language
                    )
                except Exception as e:
                    logging.error(f"Error generating journalist profile: {str(e)}")
                    journalist_profile_result = f"Error generating journalist profile: {str(e)}"

            # Generate topic analysis if requested
            if generate_topic_analysis and topic_focus and topic_focus.strip():
                try:
                    logging.info(f"Generating topic analysis for {topic_focus}")
                    topic_analysis_result = analyze_journalist_topic_coverage(
                        articles_sorted=articles_sorted,
                        journalist_name=journalist_name,
                        topic_focus=topic_focus,
                        general_folder=general_folder,
                        language=language
                    )
                except Exception as e:
                    logging.error(f"Error in topic analysis: {str(e)}")
                    topic_analysis_result = f"Error generating topic analysis: {str(e)}"

            # NEW: Generate comprehensive journalist analysis if requested
            if generate_journalist_analysis:
                try:
                    logging.info("Generating comprehensive journalist analysis")
                    journalist_analysis_result = generate_journalist_analysis_output(
                        articles_sorted=articles_sorted,
                        journalist_name=journalist_name,
                        general_folder=general_folder,
                        language=language
                    )
                except Exception as e:
                    logging.error(f"Error generating journalist analysis: {str(e)}")
                    journalist_analysis_result = f"Error generating journalist analysis: {str(e)}"
            
            # Prepare results for journalist analysis
            results = results + [journalist_profile_result, topic_analysis_result, article_list_result, journalist_analysis_result]
            
            # Before returning, generate the program summary
            try:
                summary_path = tracker.generate_report()
                logging.info(f"Program summary report generated: {summary_path}")
                
                # Save raw data for possible future analysis
                tracker.save_raw_data()
                
                # Add summary information to the journalist_analysis_result field
                if isinstance(results[-1], str):
                    report_name = os.path.basename(summary_path)
                    
                    # Create a relative path for better display
                    summary_dir = os.path.dirname(summary_path)
                    rel_path = os.path.relpath(summary_dir, base_folder)
                    display_path = os.path.join(rel_path, report_name)
                    
                    results[-1] += f"\n\n---\n\n## Program Analysis Summary\n\nA detailed execution report has been generated: {display_path}\n\n"
                    results[-1] += f"* **Total AI Model Calls:** {len(tracker.chatbot_calls)}\n"
                    results[-1] += f"* **Articles Processed:** {tracker.articles_processed}\n" 
                    results[-1] += f"* **Total Execution Time:** {(time.time() - tracker.start_time)/60:.2f} minutes\n"
                    
                    # Add used models if available
                    if tracker.model_usage:
                        results[-1] += f"* **Models Used:** {', '.join(tracker.model_usage.keys())}\n"
                    
                    # Add token usage totals
                    total_input = sum(stats["input_tokens"] for stats in tracker.model_usage.values())
                    total_output = sum(stats["output_tokens"] for stats in tracker.model_usage.values())
                    results[-1] += f"* **Total Tokens:** {total_input + total_output:,} (Input: {total_input:,}, Output: {total_output:,})\n"
            except Exception as e:
                logging.error(f"Error generating program summary: {str(e)}")

            # Return all 11 outputs for journalist analysis
            return results

        else:
            # Company analysis branch
            logging.info("Starting company analysis")
            try:
                articles_sorted, general_folder, directories_created = preprocess_articles(
                    company_name=company_name,
                    articles=articles,
                    industry_of_interest=industry_of_interest,
                    region=region
                )

                if not directories_created:
                    error_msg = "Failed to create necessary directories"
                    return [error_msg] * 11

                md_content = "Articles' reference list not requested"
                summary_insights = "Summary insights not requested"
                combined_analysis = "Comprehensive issues analysis not requested"
                topics_summaries = "Topic summaries not requested"
                media_analytics = "Media analytics not requested"
                stakeholder_quotes = "Stakeholder quotes analysis not requested"
                consolidated_quotes = "Consolidated stakeholder analysis not requested"
                journalist_profile = "Journalist profile not requested"
                topic_analysis_result = "Topic analysis not requested"
                journalist_articles_list = "Articles list not requested"
                # Since this is not journalist analysis, the new output is not applicable.
                journalist_analysis_result = "Journalist analytics not requested"

                if generate_journalist_list:
                    try:
                        logging.info("Generating Articles' reference list")
                        md_content = generate_journalist_list_output(
                            articles_sorted=articles_sorted,
                            company_name=company_name,
                            general_folder=general_folder,
                            language=language
                        )
                    except Exception as e:
                        logging.error(f"Error generating articles list: {str(e)}")
                        md_content = f"Error generating articles list: {str(e)}"

                if generate_insights:
                    try:
                        logging.info("Generating insights")
                        summary_insights = generate_insights_output(
                            articles_sorted=articles_sorted,
                            company_name=company_name,
                            general_folder=general_folder,
                            industry_of_interest=industry_of_interest,
                            region=region,
                            language=language
                        )
                    except Exception as e:
                        logging.error(f"Error generating insights: {str(e)}")
                        summary_insights = f"Error generating insights: {str(e)}"

                if generate_analysis:
                    try:
                        logging.info("Generating comprehensive analysis")
                        combined_analysis = generate_issue_analysis_output(
                            articles_sorted=articles_sorted,
                            company_name=company_name,
                            general_folder=general_folder,
                            industry_of_interest=industry_of_interest,
                            region=region,
                            language=language
                        )
                    except Exception as e:
                        logging.error(f"Error generating analysis: {str(e)}")
                        combined_analysis = f"Error generating analysis: {str(e)}"

                if generate_topics:
                    try:
                        logging.info("Generating topic summaries")
                        topics_summaries = generate_topics_output(
                            articles_sorted=articles_sorted,
                            company_name=company_name,
                            general_folder=general_folder,
                            industry_of_interest=industry_of_interest,
                            region=region,
                            language=language
                        )
                    except Exception as e:
                        logging.error(f"Error generating topics: {str(e)}")
                        topics_summaries = f"Error generating topics: {str(e)}"

                if generate_analytics:
                    try:
                        logging.info("Generating media analytics")
                        media_analytics = generate_analytics_output(
                            articles_sorted=articles_sorted,
                            company_name=company_name,
                            general_folder=general_folder,
                            industry_of_interest=industry_of_interest,
                            region=region,
                            language=language
                        )
                    except Exception as e:
                        logging.error(f"Error generating analytics: {str(e)}")
                        media_analytics = f"Error generating analytics: {str(e)}"

                if generate_quotes:
                    try:
                        logging.info("Generating stakeholder analysis")
                        stakeholder_quotes = generate_stakeholder_quotes(
                            articles_sorted=articles_sorted,
                            company_name=company_name,
                            general_folder=general_folder,
                            language=language
                        )
                    except Exception as e:
                        logging.error(f"Error generating stakeholder quotes: {str(e)}")
                        stakeholder_quotes = f"Error generating stakeholder quotes: {str(e)}"

                if generate_consolidated_quotes:
                    try:
                        logging.info("Generating consolidated stakeholder analysis")
                        consolidated_quotes = generate_consolidated_stakeholder_analysis(
                            company_name=company_name,
                            articles=articles_sorted,
                            general_folder=general_folder,
                            language=language
                        )
                    except Exception as e:
                        logging.error(f"Error generating consolidated quotes: {str(e)}")
                        consolidated_quotes = f"Error generating consolidated quotes: {str(e)}"

                # Prepare results for company analysis
                results = [
                    md_content,
                    summary_insights,
                    combined_analysis,
                    topics_summaries,
                    media_analytics,
                    stakeholder_quotes,
                    consolidated_quotes,
                    journalist_profile,
                    topic_analysis_result,
                    journalist_articles_list,
                    journalist_analysis_result  # NEW output (not applicable)
                ]
                
                # Before returning, generate the program summary
                try:
                    summary_path = tracker.generate_report()
                    logging.info(f"Program summary report generated: {summary_path}")
                    
                    # Save raw data for possible future analysis
                    tracker.save_raw_data()
                    
                    # Add summary information to the journalist_analysis_result field
                    if isinstance(results[-1], str):
                        report_name = os.path.basename(summary_path)
                        
                        # Create a relative path for better display
                        summary_dir = os.path.dirname(summary_path)
                        rel_path = os.path.relpath(summary_dir, base_folder)
                        display_path = os.path.join(rel_path, report_name)
                        
                        results[-1] += f"\n\n---\n\n## Program Analysis Summary\n\nA detailed execution report has been generated: {display_path}\n\n"
                        results[-1] += f"* **Total AI Model Calls:** {len(tracker.chatbot_calls)}\n"
                        results[-1] += f"* **Articles Processed:** {tracker.articles_processed}\n" 
                        results[-1] += f"* **Total Execution Time:** {(time.time() - tracker.start_time)/60:.2f} minutes\n"
                        
                        # Add used models if available
                        if tracker.model_usage:
                            results[-1] += f"* **Models Used:** {', '.join(tracker.model_usage.keys())}\n"
                        
                        # Add token usage totals
                        total_input = sum(stats["input_tokens"] for stats in tracker.model_usage.values())
                        total_output = sum(stats["output_tokens"] for stats in tracker.model_usage.values())
                        results[-1] += f"* **Total Tokens:** {total_input + total_output:,} (Input: {total_input:,}, Output: {total_output:,})\n"
                except Exception as e:
                    logging.error(f"Error generating program summary: {str(e)}")
                
                return results

            except Exception as e:
                error_msg = f"Error in company analysis: {str(e)}"
                logging.error(error_msg)
                logging.error(traceback.format_exc())
                return [error_msg] * 11

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}\nPlease check the app.log file for more details."
        logging.error(f"Error in Gradio interface: {str(e)}")
        logging.error(traceback.format_exc())
        return [error_msg] * 11

def update_interface(analysis_type):
    if analysis_type == "Media Coverage Analysis":
        return {
            company_name: gr.update(visible=True),
            industry: gr.update(visible=True),
            region: gr.update(visible=True),
            journalist_name: gr.update(visible=False),
            topic_focus: gr.update(visible=False),
            generate_journalist_article_list: gr.update(visible=False),  # Fixed name here
            generate_journalist_coverage_profile: gr.update(visible=False),
            generate_topic_analysis: gr.update(visible=False),
            checkbox_container_media: gr.update(visible=True),
            checkbox_container_journalist: gr.update(visible=False),
            output_container_media: gr.update(visible=True),
            journalist_articles_list_output: gr.update(visible=False),
            output_container_journalist: gr.update(visible=False)
        }
    else:  # Journalist Analysis
        return {
            company_name: gr.update(visible=False),
            industry: gr.update(visible=False),
            region: gr.update(visible=False),
            journalist_name: gr.update(visible=True),
            topic_focus: gr.update(visible=True),
            generate_journalist_article_list: gr.update(visible=True),  # Fixed name here
            generate_journalist_coverage_profile: gr.update(visible=True),
            generate_topic_analysis: gr.update(visible=True),
            checkbox_container_media: gr.update(visible=False),
            checkbox_container_journalist: gr.update(visible=True),
            output_container_media: gr.update(visible=False),
            journalist_articles_list_output: gr.update(visible=True),
            output_container_journalist: gr.update(visible=True)
        }


with gr.Blocks(title="Media and Journalist Analysis Tool") as iface:
    gr.Markdown("# Media and Journalist Analysis Tool")
    
    # Analysis Type Selection
    analysis_type = gr.Radio(
        choices=["Media Coverage Analysis", "Journalist Analysis"],
        label="Select Analysis Type",
        value="Media Coverage Analysis"
    )
    
    # Common Inputs
    with gr.Row():
        file_folder = gr.Textbox(label="PDF File Folder Path (Optional)", placeholder="Enter path to PDF folder")
        docx_file = gr.Textbox(label="DOCX File Path (Optional)", placeholder="Enter path to DOCX file")
    
    language = gr.Dropdown(
        label="Output Language",
        choices=["English", "French", "German", "Spanish", "Italian", "Dutch"],
        value="English"
    )
    
    force_reprocess = gr.Checkbox(label="Force Reprocess", value=False)
    
    # Media Coverage Analysis Inputs
    with gr.Group() as media_inputs:
        company_name = gr.Textbox(label="Company Name", placeholder="Enter company name")
        industry = gr.Textbox(label="Industry of Interest", placeholder="Enter industry")
        region = gr.Textbox(label="Region", placeholder="Enter region")
    
    # Journalist Analysis Inputs
    with gr.Group() as journalist_inputs:
        journalist_name = gr.Textbox(
            label="Journalist Name",
            placeholder="Enter journalist name",
            visible=False
        )
        topic_focus = gr.Textbox(
            label="Topic Focus",
            placeholder="Enter specific topic to analyze (e.g., 'climate change', 'economic policy')",
            visible=False
        )
    
    # Media Coverage Analysis Checkboxes
    with gr.Group() as checkbox_container_media:
        generate_journalist_list = gr.Checkbox(label="Generate Articles' reference list", value=True)
        generate_insights = gr.Checkbox(label="Generate Insights", value=False)
        generate_analysis = gr.Checkbox(label="Generate Comprehensive Issues Analysis", value=False)
        generate_topics = gr.Checkbox(label="Generate Topic Summaries", value=False)
        generate_analytics = gr.Checkbox(label="Generate Media Analytics", value=False)
        generate_quotes = gr.Checkbox(label="Generate Stakeholder Quotes Analysis", value=False)
        generate_consolidated_quotes = gr.Checkbox(label="Generate Consolidated Stakeholder Analysis", value=False)
    
    # Journalist Analysis Options
    with gr.Group() as checkbox_container_journalist:
        gr.Markdown("### Select Analysis Type(s)")
        generate_journalist_article_list = gr.Checkbox(  # Define the component here
            label="Generate Articles List",
            info="Generate a list of all articles written by the journalist",
            value=True,
            visible=False
        )
        generate_journalist_coverage_profile = gr.Checkbox(
            label="Generate Full Journalist Coverage Profile",
            info="Analyze the journalist's overall coverage patterns and style",
            value=False,
            visible=False
        )
        generate_topic_analysis = gr.Checkbox(
            label="Generate Topic-Focused Analysis",
            info="Analyze the journalist's coverage of a specific topic",
            value=False,
            visible=False
        )
        generate_journalist_analysis = gr.Checkbox(
            label="Generate Journalist Analytics Report",
            info="Generate a comprehensive analytics report including visualizations and AI-powered insights",
            value=False,
            visible=True
        )
    
    # Media Coverage Analysis Outputs
    with gr.Group() as output_container_media:
        journalist_list_output = gr.Textbox(label="Articles' reference list")
        insights_output = gr.Textbox(label="Summary Insights")
        analysis_output = gr.Textbox(label="Comprehensive Issues Analysis")
        topics_output = gr.Textbox(label="Topics Summaries")
        analytics_output = gr.Textbox(label="Media Analytics Report")
        stakeholder_output = gr.Textbox(label="Stakeholder Quotes Analysis")
        consolidated_stakeholder_output = gr.Textbox(label="Consolidated Stakeholder Analysis")
    
    # Journalist Analysis Outputs
    with gr.Group() as output_container_journalist:
        journalist_articles_list_output = gr.Textbox(  # Add this new line
            label="Journalist's Articles List",
            visible=False
        )
        journalist_profile_output = gr.Textbox(
            label="Journalist Profile Analysis",
            visible=False
        )
        topic_analysis_output = gr.Textbox(
            label="Topic-Focused Analysis",
            visible=False
        )
        journalist_analysis_output = gr.Textbox(
        label="Journalist Analytics Report",
        visible=True
    )
    
    # Submit Button
    submit_btn = gr.Button("Generate Analysis")
    
    # Connect interface update function
    # Connect interface update function
    analysis_type.change(
        fn=update_interface,
        inputs=[analysis_type],
        outputs=[
            company_name, industry, region, journalist_name, topic_focus,
            generate_journalist_article_list,  # Fixed name here
            generate_journalist_coverage_profile, generate_topic_analysis,
            checkbox_container_media, checkbox_container_journalist,
            output_container_media, journalist_articles_list_output,
            output_container_journalist
        ]
    )
    
    # Connect the submit button to the main processing function
    submit_btn.click(
        fn=gradio_journalist_list,
        inputs=[
            company_name,           # UI component references
            file_folder, 
            docx_file, 
            industry, 
            region,
            journalist_name, 
            language, 
            force_reprocess,
            generate_journalist_list,     # Checkbox for articles list
            generate_insights,
            generate_analysis, 
            generate_topics,
            generate_analytics, 
            generate_quotes,
            generate_consolidated_quotes, 
            generate_journalist_article_list,  
            generate_journalist_coverage_profile,
            generate_topic_analysis,
            generate_journalist_analysis,   # NEW checkbox input
            topic_focus
        ],
        outputs=[
            journalist_list_output,        # UI component references
            insights_output,
            analysis_output, 
            topics_output,
            analytics_output, 
            stakeholder_output,
            consolidated_stakeholder_output, 
            journalist_profile_output,
            topic_analysis_output, 
            journalist_articles_list_output,
            journalist_analysis_output      # NEW output textbox
        ]
    )


if __name__ == "__main__":
    iface.launch()