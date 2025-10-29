import smtplib
import ssl
from email.message import EmailMessage

# --- 请在这里配置您的邮件信息 ---

# 发件人邮箱 (Yahoo)
sender_email = "yao_philip@yahoo.com"

# 收件人邮箱 (Outlook)
receiver_email = "kaixi.yao@outlook.com"

# 邮件主题
subject = "来自 Python 脚本的问候"

# 邮件正文
body = """
你好，
这是一封通过 Python 脚本自动发送的测试邮件。
"""

# !! 重要的安全提示 !!
# 您需要在此处填写您的 Yahoo 邮箱的 "应用专用密码"
# 出于安全原因，Yahoo 不允许直接使用您的常规登录密码。
#
# 如何获取应用专用密码：
# 1. 登录您的 Yahoo 邮箱账户。
# 2. 前往 "账户信息" -> "账户安全"。
# 3. 找到 "应用专用密码" (App Password) 选项。
# 4. 生成一个新的密码，并将其复制粘贴到下面的 password 变量中。
#
password = "ulirkiciqhnrilet"

# --- 脚本正文 (通常不需要修改) ---

# 创建一个 EmailMessage 对象
msg = EmailMessage()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject
msg.set_content(body)

# Yahoo Mail (雅虎邮箱) 的 SMTP 服务器设置
smtp_server = "smtp.mail.yahoo.com"
smtp_port = 587  # 对于 TLS

# 创建安全的 SSL 上下文
context = ssl.create_default_context()

print(f"正在尝试连接到 {smtp_server}...")

try:
    # 使用 'with' 语句可以自动处理连接的打开和关闭
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)  # 升级到安全的 TLS 连接
        print("连接成功，正在登录...")
        
        server.login(sender_email, password)  # 使用您的邮箱和应用专用密码登录
        print("登录成功！")
        
        server.send_message(msg)  # 发送邮件
        
        print(f"邮件已成功发送至 {receiver_email}！")

except smtplib.SMTPAuthenticationError:
    print("登录失败。")
    print("请检查您的 sender_email 是否正确，并确保您使用的是 '应用专用密码'，而不是您的常规登录密码。")
except smtplib.SMTPException as e:
    print(f"发送邮件时发生错误: {e}")
except Exception as e:
    print(f"发生意外错误: {e}")
