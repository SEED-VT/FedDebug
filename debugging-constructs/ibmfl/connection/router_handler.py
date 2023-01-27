"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module acting as a bridge between Server and ProtocolHandler
Routes for all the requests received by the server are redirected
to designated handler in PH using routers
"""
import logging
from parse import parse
from ibmfl.exceptions import DuplicateRouteException


logger = logging.getLogger(__name__)


class Router(object):
    """
    Container used to add and match a group of routes.
    """

    def __init__(self):
        self.routes = {}

    def add_route(self, path, handler):
        """
        Add path, handler combination in a dictionary.
        Throws DuplicateRouteException when route is repeated

        :param path: special pre defined /assigned identifier which will be \
            used by the client to reach specific handler in the PH
        :param handler: method which needs to handle the request \
            carrying the special identifier
        :raises `DuplicateRouteException`: exception is raised \
            when path is added multiple times
        """
        if path not in self.routes:
            self.routes[path] = handler
        else:
            raise DuplicateRouteException

    def add_routes(self, routes):
        """
        Add multiple path, handler combinations in routes dictionary.
        Throws DuplicateRouteException when route is repeated

        :param routes: list of path, handler combination
        :raises `DuplicateRouteException`: exception is raised when \
        path is added multiple times
        """
        for route, handler in routes.items():
            self.add_route(route, handler)

    def get_handler(self, request_path):
        """
        Gets handler for path specified in the request

        :param request_path: path or identifier provided in \
            the request message
        :return: a handler which was assigned when Router object \
            was created
        """
        for path, handler in self.routes.items():
            parse_result = parse(path, request_path)
            if parse_result is not None:
                return handler, parse_result.named

        return None, None
