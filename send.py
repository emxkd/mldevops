#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import smtplib
server = smtplib.SMTP_SSL("smtp.gmail.com" , 465)
server.login("sender@mail.com", "sender_password")
server.sendmail("sender@mail.com"
                ,"receiver@mail.com"
                ,"msg you want to write")
server.quit()

