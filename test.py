import pandas as pd
import numpy as np
import datetime
import pmdarima
import copy

class A(object):
  def __init__(self):
    self.a = 0

a = A()
b = copy.copy(a)
print(a.a, b.a)
a.a = 1
print(a.a, b.a)

