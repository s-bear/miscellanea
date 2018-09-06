#let's screw with the import path... why not?
__path__.append(__import__('os').path.join(__path__[-1],'python'))

__all__ = ['ansi','progress']