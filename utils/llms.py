import openai
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import time

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""
 
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
 
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
 
            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1
 
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
 
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
 
                # Sleep for the delay
                time.sleep(delay)
 
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
 
    return wrapper

def annotate_texts_with_llm(texts, model, prompt, mapping_categories, sleep, temperature, OPENAI_API_KEY):
    @retry_with_exponential_backoff
    def completions_with_backoff(**kwargs):
        return openai.chat.completions.create(**kwargs)

    openai.api_key = OPENAI_API_KEY
    
    sample_texts = pd.DataFrame()
    sample_texts['Text'] = texts
    
    list_outputs = []
    
    for _,row in tqdm(sample_texts.iterrows(), desc = "Collecting LLM annotations", unit = " annotation", total = len(sample_texts)):
        entry = {}
        try:
            entry = {}
            input_text = prompt + row['Text'] + """
    
            Answer:"""
            response = completions_with_backoff(
                                          model=model, max_tokens = 5, temperature = temperature,
                                          messages=[
                                            {"role": "user", "content": input_text}
                                        ])
    
            entry['label_'+model] = response.choices[0].message.content
            time.sleep(sleep)
        except:
            entry['label_'+model] = np.nan
        
        list_outputs.append(entry)

    sample_texts['LLM_label'] = \
    pd.DataFrame(list_outputs)['label_'+model].values
    
    sample_texts['LLM_label'] = sample_texts['LLM_label'].apply(lambda x: x.strip(' ').strip('.'))
    sample_texts = sample_texts[sample_texts['LLM_label'].isin(mapping_categories.keys())]
    sample_texts['LLM_annotation'] = sample_texts['LLM_label'].apply(lambda x: mapping_categories[x])

    return sample_texts

def collect_llm_confidence(sample_texts, model, sleep, temperature, OPENAI_API_KEY):
    @retry_with_exponential_backoff
    def completions_with_backoff(**kwargs):
        return openai.chat.completions.create(**kwargs)
    
    openai.api_key = OPENAI_API_KEY
    
    prompt_part_1 = "How likely is it that the following text is "

    prompt_part_2 = """?\nOutput the probability only (a number between 0 and 1).
    
    Text: """

    list_outputs = []

    for _,row in tqdm(sample_texts.iterrows(), desc = "Collecting LLM confidence", unit = " annotation", total = len(sample_texts)):
    
        entry = {}
        try:
    
            input_text = prompt_part_1 + row['LLM_annotation']+ prompt_part_2+  row['Text'] + """\nProbability:"""
    
            response = completions_with_backoff(
                                          model=model, max_tokens = 5, temperature = temperature,
                                          messages=[
                                            {"role": "user", "content": input_text}
                                        ])
    
            entry['confidence_in_prediction'] = response.choices[0].message.content
            time.sleep(sleep)
        except:
            entry['confidence_in_prediction'] = np.nan
        
        list_outputs.append(entry)

    sample_texts['confidence_in_prediction'] = pd.DataFrame(list_outputs)['confidence_in_prediction'].values

    return sample_texts