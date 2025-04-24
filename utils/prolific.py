import requests
import json
from datetime import datetime, timezone
import time
import zipfile
import io
import json
import random
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from utils import qualtrics
from utils.qualtrics import get_qualtrics_response

def create_study(name, description, qualtrics_url, completion_code, places, reward, estimated_time,
                 max_time, HEADERS, BASE_URL):
    """
    Creates a Prolific study.

    :param name: Study title
    :param description: Study description
    :param qualtrics_url: Qualtrics survey link
    :param completion_code: Completion code for the study
    :param places: Number of participants
    :param reward: Payment per participant in GBP
    :param estimated_time: Estimated completion time in minutes
    :return: Study ID or error message
    """
    study_data = {
        "name": name,
        "description": description,
        "external_study_url": qualtrics_url+"?PROLIFIC_PID={{%PROLIFIC_PID%}}",
        "prolific_id_option": "url_parameters",  
        "completion_code": completion_code,
        "total_available_places": 1,
        "estimated_completion_time": estimated_time,
        "reward": reward,
        "maximum_allowed_time": max_time,
        "device_compatibility": ["desktop", "mobile"],  # Set devices
        "permitted_prolific_ids": None,  # No restrictions
        "eligibility_requirements": []
    }

    response = requests.post(f"{BASE_URL}/studies/", headers=HEADERS, json=study_data)

    if response.status_code == 201:
        study_info = response.json()
        #print(f"Study created successfully: {study_info['id']}")
        return study_info['id']
    else:
        print(f"Error creating study: {response.json()}")
        return None


def get_study_status(study_id, HEADERS, BASE_URL):
    """
    Fetches the status of a given study.

    :param study_id: Study ID
    :return: Study status
    """
    response = requests.get(f"{BASE_URL}/studies/{study_id}/", headers=HEADERS)

    if response.status_code == 200:
        study_status = response.json()
        print(f"Study Status: {study_status['status']}")
        return study_status['status']
    else:
        print(f"Error fetching study status: {response.json()}")
        return None


def publish_study(study_id, HEADERS, BASE_URL):
    """
    Publishes the study so participants can see it.

    :param study_id: Study ID
    :return: None
    """
    response = requests.post(
        f"{BASE_URL}/studies/{study_id}/transition/",
        headers=HEADERS,
        json={"action": "PUBLISH"}
    )
        
    #error is in response.json()
        

    
def get_study_submissions(study_id, HEADERS, BASE_URL):
    """
    Retrieves all submissions for a given study.
    
    :param study_id: Prolific study ID
    :return: List of submissions
    """


    response = requests.get(f"{BASE_URL}/studies/{study_id}/submissions/", headers=HEADERS)        

    if response.status_code == 200:
        submissions = response.json()
        return submissions
    else:
        print(f"Error fetching submissions: {response.json()}")
        return None

def approve_submissions(study_id, HEADERS, BASE_URL):
    """
    Approves all **AWAITING REVIEW** submissions for a given study.
    
    :param study_id: Prolific study ID
    """
    submissions = get_study_submissions(study_id, HEADERS, BASE_URL)

    if submissions['results']:
        submission = submissions['results'][0]
        if submission["status"] == "AWAITING REVIEW":  # Only approve pending ones
            submission_id = submission["id"]

            response = requests.post(
                f"{BASE_URL}/submissions/{submission_id}/transition/",
                headers=HEADERS,
                json={"action": "APPROVE"}
            )
    else:
        print("No submissions found or an error occurred.")


def run_prolific_annotation_pipeline(survey_links, name_prefix, description, reward, max_time, estimated_time, HEADERS, BASE_URL, QUALTRICS_API_URL, QUALTRICS_API_KEY, BATCH_TIMEOUT):
    """
    Wrapper to create, publish, monitor, approve, and collect responses for Prolific annotation tasks.

    :param survey_links: List of Qualtrics survey links.
    :param name_prefix: Prefix for study titles.
    :param description: Study description for Prolific.
    :param reward: Payment per participant in cents.
    :param estimated_time: Estimated completion time in minutes.
    :return: Dictionary mapping survey links to collected responses.
    """

    
    study_ids = {}
    responses = {}

  
    # Step 1: Create and publish studies
    for idx, survey_url in enumerate(tqdm(survey_links, desc="Publishing Annotation Tasks", unit="Annotation", total = len(survey_links) )):
        study_name = f"{name_prefix} - Survey {idx + 1}"
        completion_code = str(random.randint(1000, 9999))  # Unique completion code

        # Create the study
        study_id = create_study(
            name=study_name,
            description=description,
            qualtrics_url=survey_url,
            completion_code=completion_code,
            places=1,
            reward=reward,
            max_time = max_time,
            estimated_time=estimated_time,
            HEADERS = HEADERS,
            BASE_URL = BASE_URL
        )

        if study_id:
            study_ids[survey_url] = study_id
            publish_study(study_id, HEADERS, BASE_URL)
            #print(f"Study {study_id} published for {survey_url}")
        else:
            print(f"Failed to create study for {survey_url}")

    # Step 2: Monitor and approve submissions
    
    print("\nWaiting for annotations to be collected...")
    
    ready = False
    start_time = time.time()

    timeout = False
    
    while True:
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > BATCH_TIMEOUT:
            timeout = True
            break
        
        number_ready = 0
        status = []

        for survey_url, study_id in study_ids.items():
            submissions = get_study_submissions(study_id, HEADERS, BASE_URL)
    
            if submissions['results']:
                status.append(submissions['results'][0]["status"])
                awaiting_review = 1 if submissions['results'][0]["status"] == "AWAITING REVIEW" else 0
                approved = 1 if submissions['results'][0]["status"] == "APPROVED" else 0
                returned = 1 if submissions['results'][0]["status"] == "RETURNED" else 0
                number_ready += approved
                number_ready += awaiting_review  
                number_ready += returned
            else:
                status.append("SKIPPED")
                
            if number_ready == len(survey_links):
                ready = True
        # Create a progress bar
        
        print(f"{number_ready} out of {len(survey_links)} annotations collected so far.")
        print(f"Elapsed time: {elapsed_minutes:.2f} minutes.")

        if ready:
            break  # Exit loop when all are ready for review
    
        time.sleep(5*60)  # Check again in 5 minutes

    if timeout:
        print(f"Batch timeout. {number_ready} out of {len(survey_links)} annotations collected so far.")
        status = []
        number_ready = 0

        for survey_url, study_id in study_ids.items():
            submissions = get_study_submissions(study_id, HEADERS, BASE_URL)
    
            if submissions['results']:
                if submissions['results'][0]["status"]=="AWAITING REVIEW":
                    number_ready+=1
                    status.append("AWAITING REVIEW")
                else:
                    status.append("SKIPPED")
            else:
                status.append("SKIPPED")
                
        print(f"Moving on (batch timeout). {number_ready} annotations collected.")
        
    cnt=0
    for survey_url, study_id in study_ids.items():
        if status[cnt]=="AWAITING REVIEW":
            approve_submissions(study_id, HEADERS, BASE_URL)
        cnt+=1

    print("\nDownloading annotations...")


    # Step 3: Collect and return responses

    annotations = []
    for i in tqdm(range(len(survey_links)), desc="Downloading", unit="Annotation", total = len(survey_links) ):
        if status[i]=="AWAITING REVIEW":
            annotation = get_qualtrics_response(survey_links[i].split('/')[-1], QUALTRICS_API_URL, QUALTRICS_API_KEY)
            annotations.append(annotation)
        else:
            annotations.append(np.nan)
        
    return annotations