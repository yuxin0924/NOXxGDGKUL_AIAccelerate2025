import smtplib
import ssl
from email.message import EmailMessage

# --- Please configure your email information here ---

# Sender's email (Yahoo)
sender_email = "yao_philip@yahoo.com"

# Receiver's email (Outlook)
receiver_email = "kaixi.yao@outlook.com"

# Email subject
subject = "Greetings from a Python Script"

# Email body
body = """
Hello,
This is a test email sent automatically from a Python script.
"""

# !! Important Security Note !!
# You need to enter your Yahoo Mail "App Password" here.
# For security reasons, Yahoo does not allow using your regular login password directly.
#
# How to get an App Password:
# 1. Log in to your Yahoo Mail account.
# 2. Go to "Account Info" -> "Account Security".
# 3. Find the "App Password" option.
# 4. Generate a new password and paste it into the 'password' variable below.
#
password = "ulirkiciqhnrilet"

# --- Script Body (Usually no modification needed) ---

# Create an EmailMessage object
msg = EmailMessage()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject
msg.set_content(body)

# Yahoo Mail SMTP server settings
smtp_server = "smtp.mail.yahoo.com"
smtp_port = 587  # For TLS

# Create a secure SSL context
context = ssl.create_default_context()

print(f"Attempting to connect to {smtp_server}...")

try:
    # Using 'with' statement automatically handles opening and closing the connection
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)  # Upgrade to a secure TLS connection
        print("Connection successful, logging in...")
        
        server.login(sender_email, password)  # Log in with your email and App Password
        print("Login successful!")
        
        server.send_message(msg)  # Send the email
        
        print(f"Email successfully sent to {receiver_email}!")

except smtplib.SMTPAuthenticationError:
    print("Login failed.")
    print("Please check if your sender_email is correct and ensure you are using an 'App Password', not your regular login password.")
except smtplib.SMTPException as e:
    print(f"An error occurred while sending the email: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

