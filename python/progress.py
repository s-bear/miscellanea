
from . import ansi
import os

class Progress:
    """ Class for printing progress bars and task status

    Class Attributes
    ----------------
    disp_width : int
        Width of the display, in characters--set automatically once
        using os.get_terminal_size().columns
    indent_str : str, default '|  '
        Indent string for nested Progress items. Must be same length
        as prefix_str.
    prefix_str : str, default '\_ '
        Prefix to indicate a child Progress item. Must be same length
        as indent_str.
    min_bar_width : int, default 10
        Minimum progress bar width, in characters.
    bar_chars : str, default '[#O~]'
        Characters used for drawing the progress bar.
        '[' and ']' are the start and end of the bar,
        '#' indicates new progress, 'O' indicates prior progress,
        '~' indicates skipped progress.
    ok_str : str, default 'OK'
    cont_str : str, default '...'
    fail_str : str, default 'FAIL'
        Printed to indicate various statuses. The right margin
        is sized to fit the longest of these strings with a space.
    
    Example
    -------
    with Progress('Task 1'):
        with Progress('Task 1.1') as p:
            pass
        with Progress('Task 1.2'):
            p.interrupt('a message')
    with Progress('Task 2', 10) as p:
        p.update(0) #start progress bar
        for i in range(5):
            time.sleep(1)
            p.update(i+1) #increment when done
        p.interrupt('another message')
        for i in range(5,8):
            time.sleep(1)
            p.update(i+1)

    Final output (bar width may vary):

    Task 1
    |  \_ Task 1.1 OK
    |  \_ Task 1.2 ...
    |  |  \_ a message
    |  \_ Task 1.2 OK
    Task 1 OK
    Task 2 [##########...       ]
    |  \_ another message
    Task 2 [OOOOOOOOOO######~~~~] OK

    """
    disp_width = os.get_terminal_size().columns
    indent_str = ansi.style('|  ',fg=6,dim=True)
    prefix_str = ansi.style('\\_ ',fg=6,dim=True)
    min_bar_width = 20
    bar_chars = '[#O~]'
    ok_str   = ansi.style('OK',fg=2, bold=True)
    fail_str = ansi.style('FAIL',fg=1, bold=True)
    cont_str = ansi.style('...',fg=4, bold=True)

    __depth = 0
    __stack = []

    def __init__(self, message, total=100, quiet=False):
        """__init__(message, total)
        Parameters
        ----------
        message : str
            Message printed before the progress bar
        total : number, default 100
            The total value of the full progress bar
        """
        self.msg = message

        self.depth = None

        self.total_progress = total
        self.current_progress = None
        self.bar_width = None
        self.current_chars = None
        
        self.interrupted = False

        self.avail_len = None
        #space + longest of the status strings:
        self.status_len = 1 + max(ansi.len(Progress.ok_str), ansi.len(Progress.fail_str), ansi.len(Progress.cont_str))
        #space + [ + min_bar_width + ] + status_len:
        self.min_bar_len = 3 + Progress.min_bar_width + self.status_len

        self.quiet = quiet

    def print(self,*args,**kwargs):
        if not self.quiet: print(*args,**kwargs)

    def indent(self,msg,depth=None,pfx=None,sep='\n',width=None, crop=False):
        """indent(msg, depth=None, pfx=None, sep='\n', width=None, crop=False)

        Line-wraps and indents a message.

        Parameters
        ----------
        msg : str
            The message to wrap and indent
        depth : int or None, default None
            Indent level. If None, defaults to (self.depth + 1).
        pfx : str or None, default None
            First line prefix. If None, defaults to Progress.prefix_str
        sep : str or None, default '\n'
            Line separator. If None, returns a list of strings.
        width : int or None, default None
            Line width. If None, defaults to (disp_width - status_len)
        crop : boolean, default False
            If True, remove characters from message so it fits on one line.
        """
        if pfx is None: pfx = Progress.prefix_str
        if width is None: width = Progress.disp_width - self.status_len
        if depth is None: depth = self.depth+1
        if depth > 0:
            pfx0 = Progress.indent_str*(depth-1) + pfx
        else:
            pfx0 = ''
        imsg = pfx0 + msg
        if ansi.len(imsg) > width:
            w = width - ansi.len(pfx0) #how many characters we get per line
            x = ansi.split(msg)
            code = ''
            if len(x) > 1 and len(x[0]) == 0:
                #the message starts with an ansi code block
                code = x[1] #get it

            if crop:
                #remove characters from middle
                w1 = (w-3)//2
                w2 = w - 3 - w1
                lines = [pfx0 + ansi.style(msg[0:w1]) + ansi.style(' ~ ',fg=3) + ansi.style(msg[-w2:],code=code)]
            else:
                #break apart for line wrap
                pfx1 = Progress.indent_str*depth
                lines = []
                for m in msg.split('\n'):
                    lines += [ansi.style(m[i:i+w],code=code) for i in range(0,ansi.len(m),w)]
                lines = [pfx0 + lines[0]] + [pfx1 + l for l in lines[1:]]
        else:
            lines = [imsg]
        if sep is None:
            return lines
        else:
            return sep.join(lines)

    def __enter__(self):
        #Depth management:
        self.depth = Progress.__depth
        if self.depth > 0: Progress.__stack[-1].interrupt() #let parent know we're starting a sub-task
        Progress.__stack.append(self)
        Progress.__depth += 1

        #wrap and indent our message
        lines = self.indent(self.msg, self.depth, sep=None)
        
        #how much space is available for a status message
        #or a progress bar at the end of the line?
        self.avail_len = Progress.disp_width - ansi.len(lines[-1])

        #print the lines
        self.print(*lines,sep='\n',end='',flush=True)
        return self

    def interrupt(self, message=None):
        """interrupt(message=None)
        Interrupts the current progress bar to show a message.

        Parameters
        ----------
        message : str or None, default None
            If not None, the message is indented and printed.
        """
        #newly interrupted?
        if not self.interrupted:
            s = ' ' + Progress.cont_str
            #currently drawing a progress bar?
            if self.current_chars is not None:
                dc = self.bar_width - self.current_chars
                if ansi.len(s) > dc:
                    s = ' '*dc + Progress.bar_chars[4]
                else:
                    s += ' '*(dc - ansi.len(s)) + Progress.bar_chars[4]
                s += ' ' + Progress.cont_str
            
            self.print(s, flush=True)
            self.interrupted = True
        if message is not None:
            self.print(self.indent(message), flush=True)

    def _continue(self):
        if self.interrupted:
            #continue by reprinting the message
            #if the message is longer than one line, crop it down
            msg = self.indent(self.msg, self.depth, crop=True)
            self.print(msg,end='')

            self.avail_len = Progress.disp_width - ansi.len(msg)
            self.bar_width = None
            self.current_chars = None
            self.interrupted = False
            return True
        return False

    def update(self, amt, fail=False):
        #print continuation message if necessary:
        do_flush = self._continue()

        s = ''
        #do we need to start a new progress bar?
        if self.bar_width is None:
            #will the bar & status message fit in the available space?
            self.bar_width = self.avail_len - (3 + self.status_len)
            if self.bar_width < Progress.min_bar_width:
                # No. go to a new line
                self.print(self.indent(''),end='')
            s += ' ' + ansi.style(Progress.bar_chars[0],bold=True)
        
        #characters per unit progress
        cpp = self.bar_width/self.total_progress

        #are we starting for the first time?
        if self.current_progress is None:
            self.current_chars = 0
        #otherwise, do we need to print prior progress?
        elif self.current_chars is None:
            self.current_chars = int(self.current_progress*cpp)
            s += ansi.style(Progress.bar_chars[2]*self.current_chars,fg=6)
        
        #record fresh progress, can't exceed the total
        self.current_progress = min(amt, self.total_progress)

        #how many characters do we need to print?
        if self.current_progress == self.total_progress:
            dc = self.bar_width - self.current_chars
        else:
            dc = int(self.current_progress*cpp) - self.current_chars
        #print them as necessary:
        if dc > 0:
            if fail: 
                s += Progress.bar_chars[3]*dc
            else:
                s += ansi.style(Progress.bar_chars[1]*dc,bold=True)
            self.current_chars += dc
            #close the bar after we print the last character
            if self.current_progress == self.total_progress:
                s += ansi.style(Progress.bar_chars[4],bold=True)

        if do_flush or s: self.print(s, end='',flush=True)

    def __exit__(self,etype,eval,trace):
        self._continue()
        
        if self.current_progress is not None:
            self.update(self.total_progress,True)
        
        if etype is None:
            msg = ' ' + Progress.ok_str
        else:
            msg = ' ' + Progress.fail_str
        
        self.print(msg, flush=True)
        #revert depth counter
        Progress.__stack.pop()
        Progress.__depth = self.depth
        #True to suppress exception
        return False
