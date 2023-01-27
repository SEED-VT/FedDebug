"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import ipaddress
from ibmfl.exceptions import InvalidConfigurationException


def validate_ip(ip):
    """
    Validates ip address using ipaddress package
    :param ip: ip address
    :type ip: `str`
    :return : ip address
    :rtype : `str`

    """
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        raise InvalidConfigurationException("Invalid IP address configuration")
    return ip


def validate_port(port):
    """
    Validates port
    :param port: port
    :return : port
    :rtype : `int`

    """
    if isinstance(port, str) and not port.isdigit():
        raise InvalidConfigurationException(
            "Invalid port configuration. Should be a number")

    if int(port) > 65535 or int(port) < 1:
        raise InvalidConfigurationException("Invalid port configuration")

    return int(port)


def validate_ip_port(ip, port):
    """
    Validates IP, Port and returns them in appropriate format
    :param ip: ip address
    :type ip: `str`

    :param port: port in either str or int format
    :return : tuple of ip and port
    :rtype : `str`, `int`
    """
    return validate_ip(ip), validate_port(port)
