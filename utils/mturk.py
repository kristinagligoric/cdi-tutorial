import xmltodict
from tqdm.notebook import tqdm
import time
import pandas as pd
import boto3


def generate_task_template(task_title, annotation_instructions, text_instance):
	header = '''
	<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
	<HTMLContent><![CDATA[
	<!-- YOUR HTML BEGINS -->
	<!DOCTYPE html>
	<html>
	<head>
	<meta http-equiv='Content-Type' content='text/html; charset=UTF-8'/>
	<script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
	</head>
	<body>

	<form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'><input type='hidden' value='' name='assignmentId' id='assignmentId'/>
	'''


	footer = '''
	<h3>Please briefly explain why:</h3>
	<div>
	  <textarea name="explanation" placeholder="Type your explanation here" required></textarea>
	</div>


	<p><input type='submit' id='submitButton' value='Submit' /></p></form>
	<script language='Javascript'>turkSetAssignmentID();</script>
	</body></html>
	<!-- YOUR HTML ENDS -->
	]]>
	</HTMLContent>
	<FrameHeight>600</FrameHeight>
	</HTMLQuestion>
	'''

	#generate radio buttons dynamically, given the list with the options

	radio_buttons = '<div>\n'
	for option in annotation_instructions["options"]:
	    option_id = option.lower()
	    radio_buttons += f'  <input type="radio" id="{option_id}" name="{task_title}" value="{option}">\n'
	    radio_buttons += f'  <label for="{option_id}">{option}</label><br>\n'

	radio_buttons += '</div>'

	question = header + \
	    '\n<h2>' + annotation_instructions['question'] + '<h2>' + \
	    '\n<h3>' + text_instance + '<h3>' + \
	    radio_buttons + \
	    footer

	return question



def parse_results(worker_results):
	results = []
	if worker_results['NumResults'] > 0:
	    result = {}
	    for assignment in worker_results['Assignments']:
	        result["HIT_ID"] = assignment['HITId']
	        xml_doc = xmltodict.parse(assignment['Answer'])
	      
	        for answer_field in xml_doc['QuestionFormAnswers']['Answer']:
	            result[answer_field['QuestionIdentifier']] = answer_field['FreeText']

	    results.append(result)
	return results


def run_mturk_annotation_pipeline(sample_texts, annotation_instructions, task_title, task_description, task_reward,
                                 minimum_approval_rate, minimum_tasks_approved, aws_access_key_id, aws_secret_access_key):
    
    # Note: Here we're using the sandbox. To send to actual mturkers, just remove endpoint_url parameter
    
    mturk = boto3.client('mturk',
       aws_access_key_id = aws_access_key_id,
       aws_secret_access_key = aws_secret_access_key,
       region_name='us-east-1',
       endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )
    
    ## Step 1: Generate the annotation layout dynamically, and create tasks with the texts we want to annotate
    tasks = []
    for _,text in sample_texts.iterrows():
        tasks.append(generate_task_template(task_title, annotation_instructions, text['Text']))

    hit_ids = []

    print("Publishing anntotation tasks:")

    ## Create hits: A hit (Human Intelligence Task) is a single unit of work to complete
    for count in tqdm(range(len(tasks))):
        task = tasks[count]
        new_hit = mturk.create_hit(
            Title = task_title,
            Description = task_description,
            Keywords = 'text, quick, labeling',
            Reward = task_reward,
            MaxAssignments = 1, 
            LifetimeInSeconds = 172800,
            AssignmentDurationInSeconds = 600,
            AutoApprovalDelayInSeconds = 172800,
            Question = task,
            QualificationRequirements=[
                {
                    'QualificationTypeId': '000000000000000000L0',#Qualification for Worker_Approval_Percentage
                    'Comparator': 'GreaterThanOrEqualTo',
                    'IntegerValues': [minimum_approval_rate],
                    'RequiredToPreview': True
                },
                {
                    'QualificationTypeId': '00000000000000000040',#Qualification for NumberHITsApproved
                    'Comparator': 'GreaterThanOrEqualTo',
                    'IntegerValues': [minimum_tasks_approved],  # Minimum number of approved tasks (e.g., 1000)
                    'RequiredToPreview': True
                }]
        )
        if count ==0:
            print("You can preview the hits here:")
            print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
        
        #We will use hit ID to get results
        hit_ids.append(new_hit['HIT']['HITId'])
        
        # Modify the URL above when publishing HITs to the live marketplace.
        # Use: https://worker.mturk.com/mturk/preview?groupId=


    ## Step 2: Retrieve the results

    total_hits = len(hit_ids)
    completed_count = 0
    
    # Set up the progress bar
    progress_bar = tqdm(total=total_hits, desc="Completed HITs", unit="HIT")
    
    # Keep track of completed HITs
    completed_hits = set()
    
    while completed_count < total_hits:
        # Check the status of each HIT
        for hit_id in hit_ids:
            if hit_id in completed_hits:
                continue  # Skip already completed HITs
    
            # Get the assignment status for the HIT
            assignments = mturk.list_assignments_for_hit(HITId=hit_id)['Assignments']
            
            # Check if there are assignments and if the status is completed
            if assignments and assignments[0]['AssignmentStatus'] == 'Submitted':
                completed_hits.add(hit_id)
                completed_count += 1
                progress_bar.update(1)  # Update the progress bar for each completed HIT
    
        # Wait 3 seconds before the next check
        time.sleep(3)
    
    # Close the progress bar when done
    progress_bar.close()
    print("All HITs are completed.")


    results_list = []
    for task in hit_ids:
        if len(mturk.list_assignments_for_hit(HITId=task)['Assignments'])!=0:
            if mturk.list_assignments_for_hit(HITId=task)['Assignments'][0]['AssignmentStatus'] == 'Submitted':
                #collect the annpotations
                worker_results = mturk.list_assignments_for_hit(HITId=task, AssignmentStatuses=['Submitted'])
                #parse the response format
                results = parse_results(worker_results)
                results_list.append(results[0])
    results = pd.DataFrame(results_list)
    sample_texts['HIT_ID'] = hit_ids
    #join annotations with the initial dataset on hit id
    annotated_data = sample_texts.merge(results, on = 'HIT_ID')[['Text',task_title]]
    
    annotations = annotated_data[task_title]

    return annotations