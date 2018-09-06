#ansi.py

#This is free and unencumbered software released into the public domain.

#try the colorama package for displaying ansi coded text on windows

import re

"""ANSI color codes and helpful functions"""
#esc_re = re.compile(r'\x1B([@-_])')
CSI_RE = re.compile(r'((?:\x1B\[[0-?]*[ -/]*[@-~])+)')
ESC = '\x1B'
OSC = '\x1B]'
CSI = '\x1B['
RESET = '\x1B[0m'
BOLD = '\x1B[1m'
DIM = '\x1B[2m'
ITALIC = '\x1B[3m'
UNDERLINE = '\x1B[4m'
BLINK = '\x1B[5m'
FAST_BLINK = '\x1B[6m'
REVERSE = '\x1B[7m'
CONCEAL = '\x1B[8m'
STRIKE = '\x1B[9m'
NO_BOLD = '\x1B[21m'
NO_COLOR = '\x1B[22m'
NO_ITALIC = '\x1B[23m'
NO_UNDERLINE = '\x1B[24m'
NO_BLINK = '\x1B[25m'
NO_REVERSE = '\x1B[27m'
NO_CONCEAL = '\x1B[28m'
NO_STRIKE = '\x1B[29m'
FG_DEFAULT = '\x1B[39m'
BG_DEFAULT = '\x1B[49m'
FRAMED = '\x1B[51m'
CIRCLED = '\x1B[52m'
OVERLINE = '\x1B[53m'
NO_FRAME = '\x1B[54m'
NO_CIRCLE = '\x1B[54m'
NO_OVERLINE = '\x1B[55m'

BLACK = 0
RED = 1
GREEN = 2
YELLOW = 3
BLUE = 4
MAGENTA = 5
CYAN = 6
WHITE = 7

def fg3(n):
    return '\x1B[{}m'.format(30+n)

def fg256(n):
    return '\x1B[38;5;{}m'.format(n)

def fgrgb(r,g,b):
    return '\x1B[38;2;{};{};{}m'.format(r,g,b)

def fg(*args):
    if len(args) == 1:
        x = args[0]
        if x < 8: return fg3(x)
        else: return fg256(x)
    elif len(args) == 3:
        return fgrgb(*args)
    else:
        return FG_DEFAULT


def bg3(n):
    return '\x1B[{}m'.format(40+n)

def bg256(n):
    return '\x1B[48;5;{}m'.format(n)
def bgrgb(r,g,b):
    return '\x1B[48;2;{};{};{}m'.format(r,g,b)
def bg(*args):
    if len(args) == 1:
        x = args[0]
        if isinstance(x,(tuple,list)):
            return bg(*x)
        elif x < 8:
            return bg3(x)
        else:
            return bg256(x)
    elif len(args) == 3:
        return bgrgb(*args)
    else:
        return BG_DEFAULT

def strip(s):
    return CSI_RE.sub('',s)

def split(s):
    return CSI_RE.split(s)

def len(s):
    return len(strip(s))

def style(s, fg=None, bg=None, reset=True, bold=False, dim=False, italic=False, underline=False, strike=False, code=None):
    ss = ''
    if code is not None: ss += code
    if bold: ss += BOLD
    if dim: ss += DIM
    if italic: ss += ITALIC
    if underline: ss += UNDERLINE
    if strike: ss += STRIKE
    if fg is not None: ss += fg(fg)
    if bg is not None: ss += bg(bg)
    if reset:
        return ss + s + RESET
    else:
        return ss + s

def erase_below():
    return '\x1B[0J'
def erase_above():
    return '\x1B[1J'
def erase_all():
    return '\x1B[2J'
def erase_right():
    return '\x1B[0K'
def erase_left():
    return '\x1B[1K'
def erase_line():
    return '\x1B[2K'

def cursor_home():
    return '\x1B[H'
def cursor_save():
    return '\x1B[s'
def cursor_restore():
    return '\x1B[u'
def cursor_pos(line,column):
    return '\x1B[{};{}H'.format(line,column)
def cursor_up(lines):
    return '\x1B[{}A'.format(lines)
def cursor_down(lines):
    return '\x1B[{}B'.format(lines)
def cursor_right(columns):
    return '\x1B[{}C'.format(columns)
def cursor_left(columns):
    return '\x1B[{}D'.format(columns)
def cursor_move(lines=0,columns=0):
    s = ''
    if lines > 0:
        s += cursor_down(lines)
    elif lines < 0:
        s += cursor_up(-lines)
    if columns > 0:
        s += cursor_right(lines)
    elif columns < 0:
        s += cursor_left(-lines)
    return s

