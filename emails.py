import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
 
def send_email(body, toaddr="nrback16@earlham.edu", subject="AUTOMATIC TESTING ANALOGY FAILURE!"): 
    fromaddr = "analogyresearch@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = subject
    
    body = body + "\n ----------------------------------------- \n"
    body = body + "Please discard this email if you are not suppose to receive one. If you want to opt out of this subcription, please email David Barbella at barbeda@earlham.edu"
    msg.attach(MIMEText(body, 'plain'))
    

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "barbella19")
    text = msg.as_string()
    
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    print('There was an error in automatic testing. Error email send to ' + toaddr )
    
if __name__ == "__main__":
    send_email("Error 1999")