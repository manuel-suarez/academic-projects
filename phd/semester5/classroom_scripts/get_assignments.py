from __future__ import print_function

import os.path
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/classroom.courses.readonly',
    'https://www.googleapis.com/auth/classroom.student-submissions.students.readonly'
]

# Get course ID
if len(sys.argv) == 1:
    print("Command argument course ID required")
    exit(-1)

course_id = sys.argv[1]
if course_id is None or course_id == "":
    print("Course ID required")
    exit(-1)

def main():
    """Show basic usage of the Classroom API.
    Prints the names of the first 10 courses the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the autorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('classroom', 'v1', credentials=creds)

        # Call the Classroom API
        # Use course ID
        results = (service.courses().courseWork().list(courseId=course_id, pageSize=10).execute())
        courseWork = results.get('courseWork', [])

        if not courseWork:
            print('No course work found.')
            return
        # Prints the names of the first 10 courses.
        print('Course work:')
        for work in courseWork:
            print(work['title'], work['id'])

    except HttpError as error:
        print('An error ocurred: %s' % error)

if __name__ == '__main__':
    main()