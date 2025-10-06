import pandas as pd
from openai import OpenAI
import os
import environ
from typing import Dict
import time
import json
import re

env = environ.Env()
env.read_env()

class JobDescriptionExtractor:
    """
    AI Agent to extract structured information from job descriptions
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the extractor with OpenAI API key
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
        """
        self.api_key = api_key or env('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def extract_job_info(self, description: str) -> Dict[str, str]:
        """
        Extract job description, responsibilities, and skills from raw description
        
        Args:
            description: Raw job description text
            
        Returns:
            Dictionary with extracted_description, responsibilities, and skills
        """
        if pd.isna(description) or not description.strip():
            return {
                'extracted_description': '',
                'responsibilities': '',
                'skills_experience': ''
            }
        
        prompt = f"""
            Analyze the following job posting and extract three specific pieces of information:

            1. **Job Description**: A brief summary of what the job is about (2-3 sentences)
            2. **Key Responsibilities**: List the main duties and responsibilities (bullet points or numbered list)
            3. **Required Skills and Experience**: List the required skills, qualifications, and experience (bullet points or numbered list)

            Job Posting:
            {description}

            Please provide the response in the following JSON format:
            {{
                "extracted_description": "brief job summary here",
                "responsibilities": "• responsibility 1\\n• responsibility 2\\n• responsibility 3",
                "skills_experience": "• skill 1\\n• skill 2\\n• skill 3"
            }}

            If any section is not found in the job posting, return an empty string for that field.
            """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that extracts structured information from job postings. Always respond with valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_completion_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
            
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from response")
            
            return {
                'extracted_description': result.get('extracted_description', ''),
                'responsibilities': result.get('responsibilities', ''),
                'skills_experience': result.get('skills_experience', '')
            }
            
        except Exception as e:
            print(f"Error: {str(e)[:100]}")
            return {
                'extracted_description': '',
                'responsibilities': '',
                'skills_experience': ''
            }
    
    def process_dataframe(self, df: pd.DataFrame, description_column: str = 'description', 
                         batch_delay: float = 1.0) -> pd.DataFrame:
        """
        Process entire dataframe and add extracted columns
        
        Args:
            df: Input dataframe with job descriptions
            description_column: Name of column containing descriptions
            batch_delay: Delay between API calls to avoid rate limits (seconds)
            
        Returns:
            DataFrame with three new columns added
        """
        if description_column not in df.columns:
            raise ValueError(f"Column '{description_column}' not found in dataframe")
        
        df['extracted_description'] = ''
        df['key_responsibilities'] = ''
        df['required_skills_experience'] = ''
        
        total_rows = len(df)
        successful = 0
        failed = 0
        
        for idx, row in df.iterrows():
            print(f"Processing row {idx + 1}/{total_rows}...", end=' ')
            
            description = row[description_column]
            extracted = self.extract_job_info(description)
            
            if extracted['extracted_description'] or extracted['responsibilities'] or extracted['skills_experience']:
                successful += 1
                print("✓")
            else:
                failed += 1
                print("✗")
            
            df.at[idx, 'extracted_description'] = extracted['extracted_description']
            df.at[idx, 'key_responsibilities'] = extracted['responsibilities']
            df.at[idx, 'required_skills_experience'] = extracted['skills_experience']
            
            if idx < total_rows - 1:
                time.sleep(batch_delay)
        
        print("\nProcessing complete!")
        print(f"Successful: {successful}/{total_rows}")
        print(f"Failed: {failed}/{total_rows}")
        
        return df


def main():
    """
    Main function to demonstrate usage
    """
    print("="*70)
    print("JOB DESCRIPTION EXTRACTOR AGENT")
    print("="*70)
    
    try:
        extractor = JobDescriptionExtractor()
        print("\n✓ OpenAI API key loaded successfully")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("Please set OPENAI_API_KEY in your .env file")
        return
    
    input_file = 'soc-analyst-indeed.csv'
    
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} job listings from {input_file}")
        print(f"\nColumns: {list(df.columns)[:5]}... (showing first 5)")
    except FileNotFoundError:
        print(f"\n✗ Error: File '{input_file}' not found")
        return
    
    print("\nStarting extraction process...\n")
    df_processed = extractor.process_dataframe(
        df, 
        description_column='description',
        batch_delay=1.0
    )
    
    output_file = 'socanalystindeed_processed.csv'
    df_processed.to_csv(
        output_file, 
        index=False, 
        encoding='utf-8-sig',  
        lineterminator='\n',
        quoting=1               
    )
    print(f"\n✓ Results saved to {output_file}")
    
    print("\n" + "="*70)
    print("SAMPLE RESULTS")
    print("="*70)
    
    for idx in range(min(2, len(df_processed))):
        row = df_processed.iloc[idx]
        print(f"\n--- Job {idx + 1} ---")
        print(f"Company: {row['company']}")
        print(f"Title: {row['title']}")
        print(f"Location: {row['location']}")
        
        print("\n📝 Extracted Description:")
        desc = row['extracted_description']
        print(f"{desc if desc else '(No description extracted)'}")
        
        print("\n📋 Responsibilities:")
        resp = row['key_responsibilities']
        print(f"{resp if resp else '(No responsibilities extracted)'}")
        
        print("\n🎯 Skills:")
        skills = row['required_skills_experience']
        print(f"{skills if skills else '(No skills extracted)'}")
        print("\n" + "-"*70)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()