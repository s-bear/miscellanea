
from . import ansi
from .progress import Progress
import os

def find_files(paths, exts, recurse=True, quiet=False):
    """search paths for files ending with exts"""
    if recurse is True:
        recurse = float('inf')
    
    file_paths = []
    def search_dir(path,r):
        for f in os.listdir(path):
            f = os.path.join(path, f)
            if os.path.isdir(f) and r < recurse:
                search_dir(f, r+1)
            elif f.endswith(exts):
                file_paths.append(f)
    
    with Progress('Building file list',len(paths),quiet) as p:
        for p in paths:
            if os.path.isdir(p):
                with Progress('Searching {} for {} files'.format(p, exts),quiet=quiet):
                    search_dir(p,0)
            elif p.endswith(exts):
                file_paths.append(p)
    return file_paths

from smtplib import SMTP
from email.message import EmailMessage
class emailer:
    def __init__(self, host, port, user=None, password=None, useTLS=True):
        self.smtp = SMTP(host,port)
        self.useTLS = useTLS
        self.user = user
        self.password = password
    
    def send(self,to,subject,message,frm=None):
        if frm is None:
            frm = self.user
        msg = EmailMessage()
        msg.set_content(message)
        msg['To'] = to
        msg['From'] = frm
        msg['Subject'] = subject
        with self.smtp:
            self.smtp.connect()
            if self.useTLS:
                self.smtp.starttls()
            if self.user is not None and self.password is not None:
                self.smtp.login(self.user,self.password)
            return self.smtp.send_message(msg)

