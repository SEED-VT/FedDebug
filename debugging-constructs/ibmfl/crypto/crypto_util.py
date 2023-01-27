"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc
import random
import gmpy2 as gp
import numpy as np

from ibmfl.crypto.crypto_exceptions import CryptoException

class CryptoUtil(abc.ABC):

    @staticmethod
    def random(maximum, bits):
        """
        Generates a random number with specified bit size below the maximum.

        :param maximum: The maximum value of the \
        to be generated random number.
        :type maximum: `int`
        :param bits: The bit size of the to be generated random number. \
        :type bits: `int`
        :return: The generated random number
        :rtype: `int`
        """
        if bits < maximum.bit_length():
            raise CryptoException('Provided bits should be greater than bits of maximum')
        rand_function = random.SystemRandom()
        r = gp.mpz(rand_function.getrandbits(bits))
        while r >= maximum:
            r = gp.mpz(rand_function.getrandbits(bits))
        return r

    @staticmethod
    def random_with_maximum(maximum):
        """
        Generates a random number with maximum.

        :param maximum: The maximum value of the \
        to be generated random number.
        :type maximum: `int`
        :return: The generated random number
        :rtype: `int`
        """
        if maximum <= 0:
            raise CryptoException('Provided maximum should be greater than 0')
        bits = maximum.bit_length()
        rand_function = random.SystemRandom()
        r = gp.mpz(rand_function.getrandbits(bits))
        while r >= maximum:
            r = gp.mpz(rand_function.getrandbits(bits))
        return r

    @staticmethod
    def random_with_bits(bits):
        """
        Generates a random number with specified bits.

        :param bits: The bit size of the to be generated random number. \
        :type bits: `int`
        :return: The generated random number
        :rtype: `int`
        """
        if bits <= 0:
            raise CryptoException('Provided bit size should be great than 0')
        rand_function = random.SystemRandom()
        return gp.mpz(rand_function.getrandbits(bits))

    @staticmethod
    def crypto_encode(plaintext, 
                          precision=None, 
                          clipping_threshold=None, 
                          **kwargs):
        """
        Method to encode (preprocess) the plaintext such that it lies in a ring of integers
        (essentially the set of non-negative integers modulo a large number or prime).
        Several crypto systems (such as Paillier, Shamir's Secret Sharing, BGV/BFV 
        homomorophic encryption) require that elements of the plaintext lie in a 
        ring of integers. This method clips and quantizes elements of the plaintext,
        and maps them to a ring of integers.

        :param plaintext: Plaintext to be encoded
        :type plaintext: `list` of `np.ndarray`
        :param precision: precision for rounding off plaintext to nearest integer
        :type precision: `int`
        :param clipping_threshold: bound for clipping the plaintext
        :type clipping_threshold: `float`
        :return: Plaintext encoded into a ring of integers
        :rtype: `list` of `np.ndarray`
        """
        if not isinstance(plaintext, list):
            raise CryptoException(
                'Plaintext should be of type '
                'list of numpy.ndarray. '
                'Instead it is of type ' + str(type(plaintext)))
        
        if precision is None or not isinstance(precision, int) \
                or (precision < 0):
                raise CryptoException("Provided precision is not in the correct format. "
                                      "Precision should be a non-negative integer. ")
        
        if clipping_threshold is None or (not isinstance(clipping_threshold, float) and 
            not isinstance(clipping_threshold, int)) or (clipping_threshold <= 0):
                raise CryptoException("Provided clipping threshold is not in the correct format. "
                                      "Cliiping threshold should be a positive float. ")

        for i in range(len(plaintext)):
            if not isinstance(plaintext[i], np.ndarray):
                raise CryptoException(
                    'Provided plaintext should be of type '
                    'list of numpy.ndarray. '
                    'Instead its ' + str(i) + '-th element is of type ' +
                    str(type(plaintext[i])))
            plaintext[i] = np.clip(plaintext[i], -clipping_threshold, clipping_threshold)
            plaintext[i] = \
                (plaintext[i] * pow(10, precision)).astype(np.int64)
            plaintext[i] = plaintext[i] + int(clipping_threshold * pow(10, precision))
        
        return plaintext

    @staticmethod
    def crypto_decode(dec_weghts, 
                          precision=None, 
                          clipping_threshold=None, 
                          num_parties=1,
                          **kwargs):
        """
        Method to decode (postprocesse) the decrypted plaintext to map it from 
        a ring of integers (essentially a set of non-negative integers modulo 
        a large number or prime) to real numbers.
        Several crypto systems (such as Paillier, Shamir's Secret Sharing, BGV/BFV 
        homomorophic encryption) produce decrypted values that lie in a ring of integers. 
        This method maps the decrypted plaintext from a ring of integers to real numbers.

        :param dec_weights: Decrypted weights to be decoded
        :type dec_weights: `list` of `np.ndarray`
        :param precision: precision for rounding off plaintext to nearest integer
        :type precision: `int`
        :param clipping_threshold: bound for clipping the plaintext
        :type clipping_threshold: `float`
        :param num_parties: Number of parties involved in aggregation
        :type num_parties: `int`
        :return: Decrypted weights converted to ring of integers
        :rtype: `list` of `np.ndarray`
        """
        if not isinstance(dec_weghts, list):
            raise CryptoException(
                'Decrypted weights should be of type '
                'list of numpy.ndarray. '
                'Instead it is of type ' + str(type(dec_weghts)))
        
        if precision is None or not isinstance(precision, int) \
            or (precision < 0):
            raise CryptoException("Provided precision is not in the correct format. "
                                  "Precision should be a non-negative integer. ")
        
        if clipping_threshold is None or (not isinstance(clipping_threshold, float) and 
            not isinstance(clipping_threshold, int)) or (clipping_threshold <= 0):
                raise CryptoException("Provided clipping threshold is not in the correct format. "
                                      "Cliiping threshold should be a positive float. ")
        
        if not isinstance(num_parties, int) or num_parties < 1: 
            raise CryptoException("Provided number of parties should be a positive integer. ")

        for i in range(len(dec_weghts)):
            if not isinstance(dec_weghts[i], np.ndarray):
                raise CryptoException(
                    'Provided weights should be of type '
                    'list of numpy.ndarray. '
                    'Instead its ' + str(i) + '-th element is of type ' +
                    str(type(dec_weghts[i])))
            dec_weghts[i] = dec_weghts[i] \
                            - num_parties * int(clipping_threshold * pow(10, precision))
            dec_weghts[i] = dec_weghts[i] / pow(10, precision)
        
        return dec_weghts 
