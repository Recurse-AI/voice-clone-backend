import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import HTTPException, BackgroundTasks
import logging
from app.config.settings import settings
import time

logger = logging.getLogger(__name__) 

# Function to send HTML email using smtplib
def send_email(sender_email: str, receiver_email: str, subject: str, body: str, password: str, is_html: bool = False, raise_on_error: bool = True):
    try:
        logger.info(f"📤 Sending email to {receiver_email} from {sender_email}")

        # Create the email content
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # Attach both plain text and HTML versions
        if is_html:
            # Create plain text version as fallback
            plain_text = body.replace('<br>', '\n').replace('<p>', '').replace('</p>', '\n')
            # Remove HTML tags for plain text version
            import re
            plain_text = re.sub('<[^<]+?>', '', plain_text)

            text_part = MIMEText(plain_text, 'plain', 'utf-8')
            html_part = MIMEText(body, 'html', 'utf-8')

            msg.attach(text_part)
            msg.attach(html_part)
        else:
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Set up the SMTP server for Zoho Mail
        with smtplib.SMTP('smtp.zoho.com', 587) as server:
            server.starttls()  # Upgrade connection to a secure encrypted SSL/TLS connection
            server.login(sender_email, password)  # Login with sender's email credentials
            server.sendmail(sender_email, receiver_email, msg.as_string())  # Send email

        logger.info(f"✅ Email sent successfully to {receiver_email}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"❌ SMTP Authentication failed: {e}")
        logger.error("Check EMAIL_HOST_USER and EMAIL_HOST_PASSWORD in environment variables")
        if raise_on_error:
            raise HTTPException(status_code=500, detail="SMTP authentication failed")
        return False
    except smtplib.SMTPConnectError as e:
        logger.error(f"❌ SMTP Connection failed: {e}")
        logger.error("Check network connectivity and Zoho SMTP settings")
        if raise_on_error:
            raise HTTPException(status_code=500, detail="SMTP connection failed")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to send email to {receiver_email}: {e}")
        if raise_on_error:
            raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")
        return False


def create_email_verification_template(name: str, verification_link: str) -> str:
    """Create beautiful HTML email template for email verification"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Verify Your Email - ClearVocals</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f5f5f5; }}
            .email-container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; }}
            .logo {{ color: #ffffff; font-size: 28px; font-weight: bold; margin-bottom: 10px; }}
            .header-subtitle {{ color: #e8eaff; font-size: 16px; }}
            .content {{ padding: 40px 30px; }}
            .greeting {{ font-size: 24px; color: #333333; margin-bottom: 20px; }}
            .message {{ font-size: 16px; color: #666666; line-height: 1.6; margin-bottom: 30px; }}
            .verify-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             color: #ffffff; text-decoration: none; padding: 15px 30px; border-radius: 8px; 
                             font-size: 16px; font-weight: bold; text-align: center; margin: 20px 0; 
                             box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); transition: all 0.3s ease; }}
            .verify-button:hover {{ transform: translateY(-2px); box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5); }}
            .alternative-link {{ margin-top: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }}
            .alternative-text {{ font-size: 14px; color: #666666; margin-bottom: 10px; }}
            .copy-link {{ word-break: break-all; color: #667eea; font-size: 12px; }}
            .security-note {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; 
                             border-radius: 8px; margin: 25px 0; }}
            .security-text {{ font-size: 14px; color: #856404; }}
            .footer {{ background-color: #f8f9fa; padding: 30px; text-align: center; border-top: 1px solid #e9ecef; }}
            .footer-text {{ font-size: 14px; color: #666666; margin-bottom: 10px; }}
            .social-links {{ margin-top: 15px; }}
            .social-link {{ display: inline-block; margin: 0 10px; color: #667eea; text-decoration: none; }}
            @media only screen and (max-width: 600px) {{
                .content {{ padding: 20px 15px; }}
                .header {{ padding: 30px 15px; }}
                .greeting {{ font-size: 20px; }}
                .verify-button {{ padding: 12px 24px; font-size: 14px; }}
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <!-- Header -->
            <div class="header">
                <div class="logo">🎤 ClearVocals</div>
                <div class="header-subtitle">AI-Powered Voice & Audio Platform</div>
            </div>
            
            <!-- Main Content -->
            <div class="content">
                <h1 class="greeting">Welcome, {name}! 👋</h1>
                
                <p class="message">
                    Thank you for registering with ClearVocals! Your account has been successfully created. 
                    To access all features and use our amazing AI voice tools, 
                    please verify your email address.
                </p>
                
                <!-- Call to Action Button -->
                <div style="text-align: center;">
                    <a href="{verification_link}" class="verify-button">
                        ✨ Verify Your Email
                    </a>
                </div>
                
                <!-- Alternative Link Section -->
                <div class="alternative-link">
                    <p class="alternative-text">
                        <strong>If the button doesn't work:</strong> Copy and paste the link below into your browser:
                    </p>
                    <p class="copy-link">{verification_link}</p>
                </div>
                
                <!-- Security Note -->
                <div class="security-note">
                    <p class="security-text">
                        🔒 <strong>Security Notice:</strong> This verification link is valid for 24 hours only. 
                        If you didn't expect this email, please ignore it.
                    </p>
                </div>
                
                <p class="message">
                    After email verification, you'll get access to:
                    <br>• 🎯 25 free credits
                    <br>• 🎵 Audio separation & dubbing 
                    <br>• 🗣️ AI voice cloning
                    <br>• 📹 Video processing tools
                </p>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p class="footer-text">
                    <strong>ClearVocals Team</strong><br>
                    Your AI-powered voice companion
                </p>
                
                <p class="footer-text">
                    Need help? <a href="mailto:support@clearvocals.io" style="color: #667eea;">support@clearvocals.io</a>
                </p>
                
                <div class="social-links">
                    <a href="#" class="social-link">Website</a> |
                    <a href="#" class="social-link">Support</a> |
                    <a href="#" class="social-link">Privacy Policy</a>
                </div>
                
                <p style="font-size: 12px; color: #999999; margin-top: 15px;">
                    © 2024 ClearVocals. All rights reserved.
                </p>
            </div>
        </div>
    </body>
    </html>
    """


# Background task function to send verification email
def send_verification_email_background_task(background_tasks: BackgroundTasks, email: str, name: str, token: str):
    verification_link = f"{settings.FRONTEND_URL}/auth/verify-email?token={token}"

    # Enhanced subject line
    subject = "🎤 Welcome to ClearVocals! Please Verify Your Email"
    
    # Create beautiful HTML email body
    html_body = create_email_verification_template(name, verification_link)

    sender_email = settings.EMAIL_HOST_USER
    password = settings.EMAIL_HOST_PASSWORD  

    # Send HTML email
    background_tasks.add_task(send_email, sender_email, email, subject, html_body, password, True)


def create_password_reset_template(name: str, reset_link: str) -> str:
    """Create beautiful HTML email template for password reset"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reset Your Password - ClearVocals</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f5f5f5; }}
            .email-container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; }}
            .header {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 40px 20px; text-align: center; }}
            .logo {{ color: #ffffff; font-size: 28px; font-weight: bold; margin-bottom: 10px; }}
            .header-subtitle {{ color: #ffe8e8; font-size: 16px; }}
            .content {{ padding: 40px 30px; }}
            .greeting {{ font-size: 24px; color: #333333; margin-bottom: 20px; }}
            .message {{ font-size: 16px; color: #666666; line-height: 1.6; margin-bottom: 30px; }}
            .reset-button {{ display: inline-block; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                           color: #ffffff; text-decoration: none; padding: 15px 30px; border-radius: 8px; 
                           font-size: 16px; font-weight: bold; text-align: center; margin: 20px 0; 
                           box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4); transition: all 0.3s ease; }}
            .reset-button:hover {{ transform: translateY(-2px); box-shadow: 0 6px 16px rgba(255, 107, 107, 0.5); }}
            .alternative-link {{ margin-top: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }}
            .alternative-text {{ font-size: 14px; color: #666666; margin-bottom: 10px; }}
            .copy-link {{ word-break: break-all; color: #ff6b6b; font-size: 12px; }}
            .security-note {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; 
                             border-radius: 8px; margin: 25px 0; }}
            .security-text {{ font-size: 14px; color: #856404; }}
            .footer {{ background-color: #f8f9fa; padding: 30px; text-align: center; border-top: 1px solid #e9ecef; }}
            .footer-text {{ font-size: 14px; color: #666666; margin-bottom: 10px; }}
            @media only screen and (max-width: 600px) {{
                .content {{ padding: 20px 15px; }}
                .header {{ padding: 30px 15px; }}
                .greeting {{ font-size: 20px; }}
                .reset-button {{ padding: 12px 24px; font-size: 14px; }}
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <!-- Header -->
            <div class="header">
                <div class="logo">🔒 ClearVocals</div>
                <div class="header-subtitle">Password Reset Request</div>
            </div>
            
            <!-- Main Content -->
            <div class="content">
                <h1 class="greeting">Hello, {name}! 🔐</h1>
                
                <p class="message">
                    We received a password reset request for your ClearVocals account. 
                    Click the button below to reset your password.
                </p>
                
                <!-- Call to Action Button -->
                <div style="text-align: center;">
                    <a href="{reset_link}" class="reset-button">
                        🔐 Reset Password
                    </a>
                </div>
                
                <!-- Alternative Link Section -->
                <div class="alternative-link">
                    <p class="alternative-text">
                        <strong>If the button doesn't work:</strong> Copy and paste the link below into your browser:
                    </p>
                    <p class="copy-link">{reset_link}</p>
                </div>
                
                <!-- Security Note -->
                <div class="security-note">
                    <p class="security-text">
                        ⚠️ <strong>Security Notice:</strong> This reset link is valid for 1 hour only. 
                        If you didn't request a password reset, please ignore this email and contact us.
                    </p>
                </div>
                
                <p class="message">
                    After resetting your password:
                    <br>• Choose a strong password
                    <br>• Avoid using passwords from other accounts  
                    <br>• Keep your account secure
                </p>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p class="footer-text">
                    <strong>ClearVocals Security Team</strong><br>
                    Your account security is our priority
                </p>
                
                <p class="footer-text">
                    Need help? <a href="mailto:support@clearvocals.io" style="color: #ff6b6b;">support@clearvocals.io</a>
                </p>
                
                <p style="font-size: 12px; color: #999999; margin-top: 15px;">
                    © 2024 ClearVocals. All rights reserved.
                </p>
            </div>
        </div>
    </body>
    </html>
    """


# Background task function to send password reset email
def send_reset_email_background_task(background_tasks: BackgroundTasks, email: str, name: str, token: str):
    reset_link = f"{settings.FRONTEND_URL}/auth/reset-password?token={token}"

    # Enhanced subject line
    subject = "🔐 ClearVocals Password Reset Request"
    
    # Create beautiful HTML email body
    html_body = create_password_reset_template(name, reset_link)

    sender_email = settings.EMAIL_HOST_USER
    password = settings.EMAIL_HOST_PASSWORD

    # Send HTML email
    background_tasks.add_task(send_email, sender_email, email, subject, html_body, password, True)


def create_job_completion_template(name: str, job_type: str, job_id: str, download_urls: dict) -> str:
    """Job completion email template with distinct colors for each job type"""
    job_titles = {"dub": "Video Dubbing", "separation": "Audio Separation", "clip": "Video Clips"}
    job_title = job_titles.get(job_type, "Job")
    
    emojis = {"dub": "🎬", "separation": "🎵", "clip": "✂️"}
    emoji = emojis.get(job_type, "✅")
    
    # Define color schemes for each job type
    colors = {
        "dub": {
            "gradient": "linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)",
            "light_bg": "#eff6ff",
            "border": "#bfdbfe",
            "text": "#1e3a8a",
            "link": "#3b82f6"
        },
        "separation": {
            "gradient": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
            "light_bg": "#f0fdf4",
            "border": "#bbf7d0",
            "text": "#065f46",
            "link": "#10b981"
        },
        "clip": {
            "gradient": "linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)",
            "light_bg": "#faf5ff",
            "border": "#e9d5ff",
            "text": "#5b21b6",
            "link": "#8b5cf6"
        }
    }
    
    color_scheme = colors.get(job_type, colors["dub"])
    
    # Create download links HTML
    download_links_html = ""
    if job_type == "dub":
        download_page_url = f"{settings.FRONTEND_URL}/workspace/dubbing/download/{job_id}"
        download_links_html = f'<p><a href="{download_page_url}" class="download-button">View Download Page</a></p>'
    elif job_type == "clip":
        clips_page_url = f"{settings.FRONTEND_URL}/workspace/clips/results/{job_id}"
        download_links_html = f'<p><a href="{clips_page_url}" class="download-button">View Your Clips</a></p>'
    else:
        if download_urls.get("separation_url"):
            download_links_html = f'<p><a href="{download_urls["separation_url"]}" class="download-button">Download Separated Audio</a></p>'
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{job_title} Completed - ClearVocals</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f9fafb; }}
            .email-container {{ max-width: 600px; margin: 20px auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; }}
            .header {{ background: {color_scheme["gradient"]}; padding: 32px 20px; text-align: center; }}
            .logo {{ color: #ffffff; font-size: 24px; font-weight: 600; }}
            .content {{ padding: 32px 30px; }}
            .greeting {{ font-size: 22px; color: #111827; margin-bottom: 16px; font-weight: 600; }}
            .message {{ font-size: 15px; color: #6b7280; line-height: 1.5; margin-bottom: 24px; }}
            .download-button {{ display: inline-block; background: {color_scheme["gradient"]}; 
                               color: #ffffff; text-decoration: none; padding: 14px 28px; border-radius: 6px; 
                               font-size: 15px; font-weight: 600; margin: 8px 0; }}
            .job-info {{ background-color: {color_scheme["light_bg"]}; border-left: 3px solid {color_scheme["link"]}; 
                        padding: 16px; border-radius: 4px; margin: 20px 0; }}
            .job-info-text {{ font-size: 13px; color: {color_scheme["text"]}; line-height: 1.6; }}
            .footer {{ background-color: #f9fafb; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb; }}
            .footer-text {{ font-size: 13px; color: #6b7280; margin-bottom: 8px; }}
            @media only screen and (max-width: 600px) {{
                .content {{ padding: 24px 20px; }}
                .header {{ padding: 24px 20px; }}
                .greeting {{ font-size: 20px; }}
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="header">
                <div class="logo">{emoji} ClearVocals</div>
            </div>
            
            <div class="content">
                <h1 class="greeting">Hi {name},</h1>
                
                <p class="message">
                    Your {job_title.lower()} has been completed successfully.
                </p>
                
                <div style="text-align: center;">
                    {download_links_html}
                </div>
                
                <div class="job-info">
                    <p class="job-info-text">
                        <strong>Job ID:</strong> {job_id}<br>
                        <strong>Type:</strong> {job_title}<br>
                        <strong>Status:</strong> Completed
                    </p>
                </div>
            </div>
            
            <div class="footer">
                <p class="footer-text">
                    <strong>ClearVocals Team</strong>
                </p>
                <p class="footer-text">
                    Need help? <a href="mailto:support@clearvocals.io" style="color: {color_scheme["link"]}; text-decoration: none;">support@clearvocals.io</a>
                </p>
                <p style="font-size: 12px; color: #9ca3af; margin-top: 12px;">
                    © 2024 ClearVocals. All rights reserved.
                </p>
            </div>
        </div>
    </body>
    </html>
    """


def send_job_completion_email_background_task(background_tasks: BackgroundTasks, email: str, name: str,
                                             job_type: str, job_id: str, download_urls: dict):
    """Send job completion notification email"""
    try:
        job_titles = {"dub": "Video Dubbing", "separation": "Audio Separation", "clip": "Video Clips"}
        job_title = job_titles.get(job_type, "Job")
        
        emojis = {"dub": "🎬", "separation": "🎵", "clip": "✂️"}
        emoji = emojis.get(job_type, "✅")

        subject = f"{emoji} Your {job_title} {'are' if job_type == 'clip' else 'is'} Ready - ClearVocals"

        html_body = create_job_completion_template(name, job_type, job_id, download_urls)

        sender_email = settings.EMAIL_HOST_USER
        password = settings.EMAIL_HOST_PASSWORD

        logger.info(f"📧 Preparing email: from={sender_email}, to={email}, job={job_id}")

        if not sender_email or not password:
            logger.warning(f"⚠️ Email credentials not configured - skipping email for job {job_id}")
            return

        background_tasks.add_task(send_email, sender_email, email, subject, html_body, password, True)
        logger.info(f"✅ Email task queued for job {job_id}")

    except Exception as e:
        logger.error(f"❌ Failed to prepare completion email for job {job_id}: {e}")
