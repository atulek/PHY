from numpy import abs, cos, exp, pi, sin, sqrt, sum
import numpy as np


class TestFunc:

    costfunc = None             # Setup.py dosyasında belirlenen maliyet fonk.
    varmin = None               # Problem minimum default sınır
    varmax = None               # Problem minimum default sınır
    name = None                 # Problem adı

    # Sphere test fonksiyonu
    def sphere(self):
        self.costfunc = lambda x: sum(x ** 2)
        self.varmin = -5.12
        self.varmax = 5.12
        self.name = "sphere"
        return self

    # Rosenbrock test fonksiyonu
    def rosenbrock(self):
        self.costfunc = lambda x: sum((1 - x[:-1]) ** 2) + 100 * sum((x[1:] - x[:-1] ** 2) ** 2)
        self.varmin = -2.048
        self.varmax = 2.048
        self.name = "rosenbrock"
        return self

    # Zakharov test fonksiyonu
    def zakharov(self):
        self.costfunc = self.__zakharowf
        self.varmin = -5
        self.varmax = 10
        self.name = "zakharov"
        return self

    # Schwefel test fonksiyonu
    def schwefel(self):
        self.costfunc = lambda x: 418.9829 * len(x) - sum(x * sin(sqrt(abs(x))))
        self.varmin = -500.0
        self.varmax = 500.0
        self.name = "schwefel"
        return self

    # Rastrigin test fonksiyonu
    def rastrigin(self):
        self.costfunc = lambda x: 10 * len(x) + sum(x**2 - 10 * cos(2 * pi * x))
        self.varmin = -5.12
        self.varmax = 5.12
        self.name = "rastrigin"
        return self

    # Ackley test fonksiyonu
    def ackley(self):
        self.costfunc = self.__ackleyf
        self.varmin = -5.12
        self.varmax = 5.12
        self.name = "ackley"
        return self

    def __zakharowf(self, x):
        n = len(x)
        j = np.arange(1., n + 1)
        s2 = sum(j * x) / 2
        return sum(x ** 2) + s2 ** 2 + s2 ** 4

    def __ackleyf(self, x):
        a = 20
        b = 0.2
        c = 2 * pi
        n = len(x)
        s1 = sum(x ** 2)
        s2 = sum(cos(c * x))
        return -a * exp(-b * sqrt(s1 / n)) - exp(s2 / n) + a + exp(1)
