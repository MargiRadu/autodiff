class DualNumber:

    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, DualNumber):
            real = self.real + other.real
            dual = self.dual + other.dual
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real + other, self.dual)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            real = self.real - other.real
            dual = self.dual - other.dual
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real - other, self.dual)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            real = self.real * other.real
            dual = (self.real * other.dual) + (other.real * self.dual)
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real * other, self.dual)

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            real = self.real / other.real
            dual = ((-1) * self.real * (self.dual + other.dual)) / (other.real ** 2)
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real / other, self.dual)
