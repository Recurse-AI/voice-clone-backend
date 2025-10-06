import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import HTTPException, BackgroundTasks
import logging
from app.config.settings import settings
import time

logger = logging.getLogger(__name__) 

def get_logo_url():
    """Get the official ClearVocals logo URL"""
    return "https://pub-e668f82c3ede4548869ac0a3acad4e7f.r2.dev/dub-uploads/test_acb34834/main_transparent_backgrund.png"

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
    logo_url = get_logo_url()
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
                <a href="https://clearvocals.ai" target="_blank" style="text-decoration: none;">
                    <div style="display: inline-block; background-color: #ffffff; border-radius: 20px; padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); border: 2px solid rgba(255, 255, 255, 0.3);">
                        <img src="{logo_url}" alt="ClearVocals Logo" style="height: 70px; display: block;">
                    </div>
                </a>
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
    logo_url = get_logo_url()
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
                <a href="https://clearvocals.ai" target="_blank" style="text-decoration: none;">
                    <div style="display: inline-block; background-color: #ffffff; border-radius: 20px; padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); border: 2px solid rgba(255, 255, 255, 0.3);">
                        <img src="{logo_url}" alt="ClearVocals Logo" style="height: 70px; display: block;">
                    </div>
                </a>
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
    """Enhanced job completion email template with modern design"""
    logo_url = get_logo_url()
    job_titles = {"dub": "Video Dubbing", "separation": "Audio Separation", "clip": "Video Clips"}
    job_title = job_titles.get(job_type, "Job")
    
    # emojis = {"dub": "🎬", "separation": "🎵", "clip": "✂️"}
    # emoji = emojis.get(job_type, "✅")
    
    colors = {
        "dub": {"gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "accent": "#667eea", "light": "#f3f4ff"},
        "separation": {"gradient": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", "accent": "#f093fb", "light": "#fff5f7"},
        "clip": {"gradient": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)", "accent": "#4facfe", "light": "#f0fbff"}
    }
    color = colors.get(job_type, colors["dub"])
    
    if job_type == "dub":
        download_page_url = f"{settings.FRONTEND_URL}/workspace/dubbing/download/{job_id}"
        download_links = f'<a href="{download_page_url}" class="cta-button">🎬 View & Download</a>'
    elif job_type == "clip":
        clips_page_url = f"{settings.FRONTEND_URL}/workspace/clips/results/{job_id}"
        download_links = f'<a href="{clips_page_url}" class="cta-button">✂️ View Your Clips</a>'
    else:
        url = download_urls.get("separation_url", "#")
        download_links = f'<a href="{url}" class="cta-button">🎵 Download Audio</a>'
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{job_title} Ready - ClearVocals</title>
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; padding: 40px 20px;">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.3);">
            <tr>
                <td style="background: {color["gradient"]}; padding: 50px 40px; text-align: center;">
                    <a href="https://clearvocals.ai" target="_blank" style="text-decoration: none;">
                        <div style="display: inline-block; background-color: #ffffff; border-radius: 24px; padding: 24px; margin-bottom: 25px; box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2); border: 3px solid rgba(255, 255, 255, 0.4);">
                            <img src="{logo_url}" alt="ClearVocals Logo" style="height: 80px; display: block;">
                        </div>
                    </a>
                    
                    <h1 style="color: #ffffff; font-size: 32px; font-weight: 700; margin: 0 0 12px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        Your {job_title} is Ready!
                    </h1>
                    <p style="color: rgba(255,255,255,0.95); font-size: 16px; margin: 0;">Processing Complete ✨</p>
                </td>
            </tr>
            
            <tr>
                <td style="padding: 45px 40px; background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);">
                    <div style="text-align: center; margin-bottom: 35px;">
                        <div style="display: inline-block; background: {color["light"]}; border-radius: 50%; width: 80px; height: 80px; line-height: 80px; margin-bottom: 20px;">
                            <span style="font-size: 40px;">✓</span>
                        </div>
                        <h2 style="color: #1a1a1a; font-size: 24px; font-weight: 600; margin: 0 0 12px 0;">Hi {name}! 👋</h2>
                        <p style="color: #666666; font-size: 16px; line-height: 1.6; margin: 0;">
                            Great news! Your {job_title.lower()} has been processed successfully and is ready to download.
                        </p>
                    </div>
                    
                    <div style="text-align: center; margin: 35px 0;">
                        {download_links}
                    </div>
                    
                    <div style="background: linear-gradient(135deg, {color["light"]} 0%, #ffffff 100%); border-radius: 12px; padding: 25px; margin: 30px 0; border: 2px solid {color["accent"]}20;">
                        <table width="100%" cellpadding="8" cellspacing="0">
                            <tr>
                                <td style="color: #666666; font-size: 14px; font-weight: 500;">📋 Job ID:</td>
                                <td style="color: #1a1a1a; font-size: 14px; font-weight: 600; text-align: right;">{job_id[:12]}...</td>
                            </tr>
                            <tr>
                                <td style="color: #666666; font-size: 14px; font-weight: 500;">🎯 Type:</td>
                                <td style="color: #1a1a1a; font-size: 14px; font-weight: 600; text-align: right;">{job_title}</td>
                            </tr>
                            <tr>
                                <td style="color: #666666; font-size: 14px; font-weight: 500;">✅ Status:</td>
                                <td style="color: #10b981; font-size: 14px; font-weight: 700; text-align: right;">COMPLETED</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div style="background: #f8f9fa; border-radius: 10px; padding: 20px; margin-top: 25px; text-align: center;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 12px 0;">
                            💡 <strong>Pro Tip:</strong> Download your files within 7 days for guaranteed availability.
                        </p>
                    </div>
                </td>
            </tr>
            
            <tr>
                <td style="background: #f8f9fa; padding: 30px 40px; border-top: 1px solid #e5e7eb;">
                    <table width="100%" cellpadding="0" cellspacing="0">
                        <tr>
                            <td style="text-align: center;">
                                <p style="color: #1a1a1a; font-size: 15px; font-weight: 600; margin: 0 0 8px 0;">🎤 ClearVocals</p>
                                <p style="color: #666666; font-size: 13px; margin: 0 0 15px 0;">AI-Powered Audio & Video Processing</p>
                                <p style="color: #666666; font-size: 13px; margin: 0;">
                                    Questions? <a href="mailto:support@clearvocals.io" style="color: {color["accent"]}; text-decoration: none; font-weight: 600;">Contact Support</a>
                                </p>
                                <p style="color: #999999; font-size: 12px; margin: 15px 0 0 0;">© 2024 ClearVocals. All rights reserved.</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <style>
            .cta-button {{
                display: inline-block;
                background: {color["gradient"]};
                color: #ffffff !important;
                text-decoration: none;
                padding: 16px 40px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 700;
                letter-spacing: 0.3px;
                box-shadow: 0 8px 20px {color["accent"]}40;
                transition: transform 0.2s;
            }}
            .cta-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 12px 28px {color["accent"]}60;
            }}
            @media only screen and (max-width: 600px) {{
                body {{ padding: 20px 10px !important; }}
                table {{ border-radius: 12px !important; }}
                td {{ padding: 30px 20px !important; }}
                h1 {{ font-size: 26px !important; }}
                h2 {{ font-size: 22px !important; }}
            }}
        </style>
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
