# encoding: utf8

import sys


def g():
    frame = sys._getframe()
    print('current function is %s ' % frame.f_code.co_name)
    caller = frame.f_back
    print('caller function is %s' % caller.f_code.co_name)
    print("caller's local namespace: %s" % caller.f_locals)
    print("caller's global namespace: %s" % caller.f_globals.keys())


def f():
    a = 1
    b = 2
    g()


def show():
    f()


show()
