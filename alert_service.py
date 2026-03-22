import os
import smtplib
import threading
import time
from dotenv import load_dotenv
load_dotenv()
from email.message import EmailMessage
from datetime import datetime

EMAIL_USER = os.getenv("SMARTFALL_EMAIL_USER")
EMAIL_PASS = os.getenv("SMARTFALL_EMAIL_PASS")
EMAIL_TO = os.getenv("SMARTFALL_EMAIL_TO")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

_last_sent_time = 0
COOLDOWN_SECONDS = 60

def send_fall_alert(confidence=0.0, trigger_source="unknown", location="home", extra_notes=""):
	def task():
		global _last_sent_time

		now = time.time()
		if now - _last_sent_time < COOLDOWN_SECONDS:
			print("Alert suppressed due to cooldown")
			return
			
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		
		msg = EmailMessage()
		msg["Subject"] = "SmartFall Alert: Fall Detected"
		msg["From"] = EMAIL_USER
		msg["To"] = EMAIL_TO
		
		body = f"""FALL DETECTED
	
Time: {timestamp}
Location: {location}
Confidence: {confidence:.2f}
Trigger: {trigger_source}

Notes: {extra_notes if extra_notes else "No additional notes."}
 
Please check immediately.

""".strip()

		msg.set_content(body)
		
		try:
			with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
				smtp.login(EMAIL_USER, EMAIL_PASS)
				smtp.send_message(msg)
			
			_last_sent_time = now
			print("Alert email sent.")
		
		except Exception as e:
			print("Email failed:", e)
		
	threading.Thread(target=task).start()
		

