import requests


def send_notification(message):
    """
    Send a notification message to Google Chat.

    Args:
        message (str): The message to be sent.
    """
    url = "https://chat.googleapis.com/v1/spaces/AAAAVklVDOw/messages?key=[chat key]&token=[chat token]"
    data = {"text": message}
    response = requests.post(url, json=data)
    if response.status_code != 200:
        print("Failed to send notification.")
    else:
        print("Notification sent successfully.")
