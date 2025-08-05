import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import HTTPException, BackgroundTasks
from app.utils.logger import logger
from app.config.settings import settings
import time 

# Function to send email using smtplib
def send_email(sender_email: str, receiver_email: str, subject: str, body: str, password: str):
    try:
        # Create the email content
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Set up the SMTP server and send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Upgrade connection to a secure encrypted SSL/TLS connection
            server.login(sender_email, password)  # Login with sender's email credentials
            server.sendmail(sender_email, receiver_email, msg.as_string())  # Send email
        logger.info(f"Email sent to {receiver_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise HTTPException(status_code=500, detail="Error sending verification email")


# Background task function to send verification email
def send_verification_email_background_task(background_tasks: BackgroundTasks, email: str, name: str, token: str):
    verification_link = f"{settings.FRONTEND_URL}/verify-email?token={token}"

    subject = "Please Verify Your Email"
    body = f"Hello {name},\n\nPlease click on the link below to verify your email address:\n{verification_link}"

    sender_email = settings.EMAIL_HOST_USER
    password = settings.EMAIL_HOST_PASSWORD  

    background_tasks.add_task(send_email, sender_email, email, subject, body, password)

# Background task function to send password reset email
def send_reset_email_background_task(background_tasks: BackgroundTasks, email: str, name: str, token: str):
    reset_link = f"{settings.FRONTEND_URL}/reset-password?token={token}"

    subject = "Reset Your Password"
    body = f"Hello {name},\n\nWe received a request to reset your password.\nPlease click the link below to proceed:\n{reset_link}\n\nIf you did not request this, you can ignore this email."

    sender_email = settings.EMAIL_HOST_USER
    password = settings.EMAIL_HOST_PASSWORD

    background_tasks.add_task(send_email, sender_email, email, subject, body, password)
