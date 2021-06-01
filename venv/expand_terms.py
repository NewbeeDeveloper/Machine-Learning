import sympy as sym

from IPython.display import display, Math

#"Importamos la variable simbólica(incógnita) 'x'
from sympy.abc import x
from sympy import *

print("---- Declaración de los términos ----")
term1 = (4*x +5)
term2 = x

display(term1)
display(term2)

print("---- Expansión de términos ----")
expantion = sym.expand(term1*term2)
display(expantion)

print("---- Formato a la solución ----")
print("%s" % (sym.latex(expantion)))