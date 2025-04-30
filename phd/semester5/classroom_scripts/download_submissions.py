from __future__ import print_function

import os
import io
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'  # Ignore change of SCOPE warnings
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/classroom.student-submissions.students.readonly',
    'https://www.googleapis.com/auth/classroom.courses.readonly',
    'https://www.googleapis.com/auth/classroom.rosters.readonly ',
]

# Get course ID
if len(sys.argv) < 4:
    print("Command arguments course ID, courseWork ID and store route required")
    exit(-1)

course_id = sys.argv[1]
if course_id is None or course_id == "":
    print("Course ID required")
    exit(-1)
course_work_id = sys.argv[2]
if course_work_id is None or course_work_id == "":
    print("CourseWork ID required")
    exit(-1)
destination = sys.argv[3]
if destination is None or destination == "":
    print("Destination dir required")
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
        gservice = build('drive', 'v3', credentials=creds)

        # Call the Classroom API
        # Use course ID
        results = (service.courses().courseWork().studentSubmissions().list(courseId=course_id, courseWorkId=course_work_id).execute())
        profiles = service.userProfiles()
        submissions = results.get('studentSubmissions', [])

        if not submissions:
            print('No submissions found.')
            return
        # Create destination dir
        if not os.path.exists(destination):
            os.makedirs(destination, exist_ok=True)

        print('Course work submissions:')
        for submission in submissions:
            profile = (profiles.get(userId=submission['userId']).execute()).get('name', [])
            print(submission['id'], submission['userId'], profile['fullName'], submission['assignmentSubmission'])
            attachments = submission['assignmentSubmission'].get('attachments')
            if attachments is None:
                # Descartamos estudiante si no subió nada a su tarea
                continue
            # Creamos directorio del estudiante
            studentName = profile['fullName']
            os.makedirs(os.path.join(destination, studentName), exist_ok=True)
            # Descargamos archivos
            for attachment in attachments:
                driveFile = attachment['driveFile']
                file_id = driveFile['id']
                file_name = driveFile['title']
                request = gservice.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print("Download %d%%." % int(status.progress() * 100))

                fh.seek(0)
                with open(os.path.join(destination, studentName, file_name), 'wb') as f:
                    f.write(fh.read())
                    f.close()

    except HttpError as error:
        print('An error ocurred: %s' % error)

if __name__ == '__main__':
    main()