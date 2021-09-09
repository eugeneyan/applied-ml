from gaussiandistribution import Gaussian

gaussian_one = Gaussian(22, 2)
print(gaussian_one.mean)

# class fraction:
#     def __init__(self, n, d):
#         self.numerator, self.denominator = fraction.reduce(n, d)
#
#     @classmethod
#     def reduce(cls, n1, n2):
#         gcd = cls.gcd(n1, n2)
#         return n1//gcd, n2//gcd
#
#     @staticmethod
#     def gcd(a, b):
#         while b != 0:
#             a, b = b, a % b
#         return a
#
#     def __str__(self):
#         return str(self.numerator) + '/' + str(self.denominator)
#
#
# test = fraction(8,12)
# print(test)
#
#
#
