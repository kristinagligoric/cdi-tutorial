import requests
import json
from datetime import datetime, timezone
import time
import zipfile
import io
import json
import random
import pandas as pd
from tqdm.notebook import tqdm

def create_qualtrics_survey(QUALTRICS_API_URL, QUALTRICS_API_KEY):
    survey_data = {
        "SurveyName": "CDI",
        "Language": "EN",
        "ProjectCategory": "CORE"
            }

    headers = {
        'x-api-token': QUALTRICS_API_KEY,
        'Content-Type': 'application/json'
    }
    
    response = requests.post(
        f'{QUALTRICS_API_URL}/survey-definitions', 
        headers=headers, 
        json=survey_data
    )

    if response.status_code != 200:
        print("Qualtrics API Error:", response.status_code, response.text)
        return None
    survey_id = response.json()['result']['SurveyID']
    return survey_id


# Activate Qualtrics Survey and get survey link
def activate_and_get_survey_link(survey_id, text_to_annotate, categories, annotation_instruction, QUALTRICS_API_URL, QUALTRICS_API_KEY):

    #add a question
    
    headers = {
        'x-api-token': QUALTRICS_API_KEY,
        'Content-Type': 'application/json'
    }

    url = f'{QUALTRICS_API_URL}/survey-definitions/{survey_id}/questions'

    choices = {str(idx + 1): {"Display": category} for idx, category in enumerate(categories)}
    
    data = {
    "QuestionText": annotation_instruction+"\n\n"+text_to_annotate,
    "DataExportTag": "Q1",
    "QuestionType": "MC",
    "Selector": "SAVR",
    "SubSelector": "TX",
    "Configuration": {
        "QuestionDescriptionOption": "UseText"
    },
    "Choices": choices
}
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code in [200, 201]:
        #print("Question added.")
        pass
    else:
        print("Error adding question:", response.status_code, response.text)


    #activate the survey

    activate_url = f'{QUALTRICS_API_URL}/surveys/{survey_id}'
    
    headers = {
        'x-api-token': QUALTRICS_API_KEY,
        'Content-Type': 'application/json'
    }
    
    activate_payload = {
        "isActive": True
    }
    
    response = requests.put(activate_url, json=activate_payload, headers=headers)
    
    if response.status_code == 200:
        #print("Survey activated.")
        pass
    else:
        print("Error activating survey:", response.status_code, response.text)

    #publish the survey
    
    publish_url = f'{QUALTRICS_API_URL}/survey-definitions/{survey_id}/versions'

    publish_payload = {
        "Description": "Initial version", "Published": True
    }
    
    response = requests.post(publish_url, json=publish_payload, headers=headers)
    
    if response.status_code in [200, 201]:
        #print("Survey published.")
        pass
    else:
        print("Error publishing survey:", response.status_code, response.text)


    #get the anonymous link
    QUALTRICS_SERVER = "https://" + QUALTRICS_API_URL.split("//")[1].split("/")[0]
        
    anonymous_link = f"{QUALTRICS_SERVER}/jfe/form/{survey_id}"
    
    #print("Anonymous Survey Link:", anonymous_link)
       
    
    return anonymous_link


def create_and_activate_surveys(texts_to_annotate, categories, annotation_instruction, QUALTRICS_API_URL, QUALTRICS_API_KEY):
    """
    Creates and activates a separate Qualtrics survey for each text with a progress bar.
    
    :param texts_to_annotate: List of texts to annotate.
    :return: Dictionary mapping texts to their survey links.
    """
    survey_links = {}

    # Use tqdm for a Jupyter-friendly progress bar
    for text in tqdm(texts_to_annotate, desc="Creating Annotation Tasks", unit="Annotation", colour="blue", leave=True, total = len(texts_to_annotate)):
        # Create a new survey
        survey_id = create_qualtrics_survey(QUALTRICS_API_URL, QUALTRICS_API_KEY)
        if not survey_id:
            print(f"Skipping text '{text}' due to survey creation failure.")
            continue

        # Activate survey with a single question
        survey_url = activate_and_get_survey_link(survey_id, text, categories, annotation_instruction, QUALTRICS_API_URL, QUALTRICS_API_KEY)
        
        # Store the survey link
        survey_links[text] = survey_url

    return survey_links


# Retrieve Qualtrics Survey Responses
def get_qualtrics_response(survey_id, QUALTRICS_API_URL, QUALTRICS_API_KEY):
    # Headers
    headers = {
        "X-API-TOKEN": QUALTRICS_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Start Data Export Job
    export_url = f"{QUALTRICS_API_URL}/surveys/{survey_id}/export-responses"
    export_payload = {"format": "json"}
    
    response = requests.post(export_url, headers=headers, json=export_payload)
    if response.status_code != 200:
        print("Error starting export:", response.json())
        exit()
    
    progress_id = response.json()["result"]["progressId"]
    #print(f"Export started: Progress ID = {progress_id}")
    
    # Check Export Progress
    progress_url = f"{QUALTRICS_API_URL}/surveys/{survey_id}/export-responses/{progress_id}"
    
    i = 0
    
    while True:
        progress_response = requests.get(progress_url, headers=headers)
        progress_data = progress_response.json()
    
        if progress_data["result"]["status"] == "complete":
            file_id = progress_data["result"]["fileId"]
            #print("Export complete! File ID:", file_id)
            break
        elif progress_data["result"]["status"] == "failed":
            print("Export failed:", progress_data)
            exit()
        if i==0:
            pass
        else:
            i+=1
        time.sleep(5)
    
    # Download the Data File
    download_url = f"{QUALTRICS_API_URL}/surveys/{survey_id}/export-responses/{file_id}/file"
    file_response = requests.get(download_url, headers=headers, stream=True)
    
    with zipfile.ZipFile(io.BytesIO(file_response.content), 'r') as zip_ref:
            zip_ref.extractall("qualtrics_responses")  # Extract to a folder
            extracted_files = zip_ref.namelist()  # List files in zip

    
    json_filename = f"qualtrics_responses/{extracted_files[0]}"  # First file in the zip
    with open(json_filename, "r", encoding="utf-8") as json_file:
        responses_data = json.load(json_file)

    return responses_data["responses"][0]["labels"]["QID1"]