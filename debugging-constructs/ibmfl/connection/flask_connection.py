"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""Connection class which uses flask and request libraries to create a server
client combo.
All supported config combinations are shown below.
connection:
  name: FlaskConnection
  path: ibmfl.connection.flask_connection
  info:
    ip: <ip>
    port: <port>
    headers:
      'Content-type': 'application/json'
    tls_config:
      enable: true
      cert: <server.crt>
      key: <server.key>
      ca_cert: <ca.pem>

    Headers and tls config both are optional. By default `Content-type` and
    `Accept` header are sent for every request. Additional headers can be added
    by changing the header section of info. 
    TLS is by default False and can be enabled by switching adding tls_config to 
    info section and assigning True. If certificates are not provided, adhoc ssl_context 
    is assigned which uses Flask adhoc feature to generate certificates for connections.
    CA certificate is optional and should be provided if mutual authentication is required
    on both server and client side. 
"""
import logging
import requests
import threading

import ssl
from requests.exceptions import SSLError

from flask import Flask, request, Response
import ibmfl.util.config as fl_config
from ibmfl.connection.connection import ConnectionStatus
from ibmfl.connection.connection import FLConnection, FLSender, FLReceiver
from ibmfl.util.validate import validate_ip_port
from ibmfl.message.message import Message
from ibmfl.message.serializer_types import SerializerTypes
from ibmfl.message.serializer_factory import SerializerFactory
from ibmfl.exceptions import InvalidConfigurationException, \
    InvalidServerConfigurationException, FLException



logger = logging.getLogger(__name__)
handler = logging.FileHandler('_flask_connection.log', mode="w")
logger.addHandler(handler)


class FlaskConnection(FLConnection):

    DEFAULT_CONFIG = {'ip': '127.0.0.1', 'port': 5000}

    def __init__(self, config):
        """Initialized the connection object and validates the config
        provided to this connection instance

        :param config: Dictionary of configuration provided to connection
        :type config: `dict`
        """
        self.settings = self.process_config(config)
        # self.ssl_context = self.load_ssl_context(config)
        self.ssl_config = self.get_ssl_config(config)
        self.config = config
        self.server_thread = None
        self.receiver = None
        self.status = None
        self.sender = None
        self.app = None

    def initialize(self, **kwargs):
        """Initialize receiver and sender """
        router = kwargs.get('router')
        self.initialize_receiver(router=router)
        self.initialize_sender()

    def initialize_receiver(self, router=None):
        """Initialize flask server using the settings and handler.

        :param router: Router object describing the routes for each request \
            which are passed down to PH
        :type router: `Router`
        """

        self.receiver = FlaskReceiver(
            router, self.settings.get('ip'), self.settings.get('port'), self.ssl_config)
        logger.info('Receiver Initialized')
        self.receiver.initialize()
        self.status = ConnectionStatus.INITIALIZED

    def initialize_sender(self):
        """Initialize simple http client using the settings provided during
        connection creation
        """
        self.sender = RestSender(self.settings, self.ssl_config)
        self.sender.initialize()
        self.status = ConnectionStatus.INITIALIZED

    def process_config(self, config):
        """Validates the configuration provided to flask connection and returns
        a settings dictionary with all the information extracted from config

        :param config: configuraiton sent from application
        :type config: `dict`

        :return: settings
        :rtype: `dict`
        """

        if config:
            host = config.get('ip')
            port = config.get('port')

            if host and port:
                host, port = validate_ip_port(host, port)
                settings = {'ip': host, 'port': port}
            else:
                settings = self.DEFAULT_CONFIG

            if 'id' in config:
                settings['id'] = config.get('id')

            if 'url' in config:
                settings['url'] = config.get('url')
        else:
            raise InvalidServerConfigurationException(
                'No connection configuration found')
        return settings

    def get_ssl_config(self, config):
        """Loads tls config from connection config. Returns None when no config is provided
        :param config: Configuraiton sent from application
        :type config: `dict`

        :return: ssl config
        :rtype: `dict`
        """
        ssl_config = None
        if 'tls_config' in config and config.get('tls_config').get('enable'):
            ssl_config = {}
            tls_config = config.get('tls_config')
            if 'cert' in tls_config and 'key' in tls_config:
                try:
                    ssl_config['cert'] = fl_config.get_absolute_path(
                        tls_config.get('cert'))
                    ssl_config['key'] = fl_config.get_absolute_path(
                        tls_config.get('key'))
                    if 'ca_cert' in tls_config:
                        ssl_config['mutual_auth'] = True
                        ssl_config['ca_cert'] = fl_config.get_absolute_path(
                            tls_config.get('ca_cert'))

                except InvalidConfigurationException as invalid_file_location:
                    logger.exception(invalid_file_location)
                    logger.exception("Invalid cert file location.")
                    raise InvalidConfigurationException(
                        "Invalid Flask connection config, certificate file not found")

        return ssl_config

    def get_connection_config(self):
        """Provide a connection information such that a node can communicate 
        to other nodes on how to communicate with it.

        :return: settings
        :rtype: `dict`
        """
        return self.settings

    def start(self):
        """starts the receiver in a new thread
        """
        self.start_receiver()
        self.status = ConnectionStatus.STARTED
        self.SENDER_STATUS = ConnectionStatus.STARTED

    def start_receiver(self):
        """starts the receiver in a new thread
        """
        self.server_thread = threading.Thread(target=self.receiver.start)
        self.server_thread.start()
        self.status = ConnectionStatus.STARTED
        self.RECEIVER_STATUS = ConnectionStatus.STARTED

    def stop(self):
        """Stop and cleanup the connection
        """
        logger.info('Stopping Receiver and Sender')
        self.status = ConnectionStatus.STOPPED
        self.receiver.stop()


class FlaskReceiver(FLReceiver):

    def __init__(self, router, host, port, ssl_config=None):
        """
        Initializes server instance with reference to Router
        :param router: Router object describing the routes for each request
            which are passed down to PH
        :type router: `Router`

        :param host: Host of the webserver
        :type host: `str`
        :param port: Port on which the webserver should listen
        :type port: `int`
        """

        self.router = router
        self.host = host
        self.port = port
        self.ssl_config = ssl_config
        self.ssl_context = self.load_ssl_context(self.ssl_config)

    def load_ssl_context(self, config):
        """Loads tls certificates from specified location and creates a SSL 
        context based on TLS v 1.2.
        If certificates are not provided and ssl enabled is false, returns None
        If certificates are not provided and ssl enabled is true, returns adhoc context

        :param config: Configuration sent from application
        :type config: `dict`

        :return: context
        :rtype: `ssl.SSLContext`
        """

        if not config:
            return None

        if 'cert' in config and 'key' in config:
            try:
                context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
                context.load_cert_chain(config.get('cert'), config.get('key'))
                if config.get('mutual_auth'):
                    context.verify_mode = ssl.CERT_REQUIRED
                    context.load_verify_locations(config.get('ca_cert'))
            except Exception as ex:
                logger.exception(ex)
                raise FLException(
                    "Error occured while initializing Flask connection.")
        else:
            context = 'adhoc'

        return context

    def shutdown_server(self):
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()

    def initialize(self):
        """
        Creates a instance of Flask app and assigns a Router to route
        all request to Router object

        :return: Response object is sent back to the client
        """
        logger.info('Initializing Flask application')
        app = Flask(__name__)

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            self.shutdown_server()
            return 'Server shutting down...'

        @app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
        @app.route('/<path:path>', methods=['GET', 'POST'])
        def handle_request(path):
            logger.info('Request received for path :'+path)

            serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()

            message = serializer.deserialize(request.data)
            handler, kwargs = self.router.get_handler(
                request_path=path)
            if handler is None:
                logger.info('Invalid Request ! Routing it to default handler')
                handler, kwargs = self.router.get_handler(
                    request_path='default')
            try:
                res_message = handler(message)

            except Exception as ex:
                res_message = Message()
                data = {'status': 'error', 'message': str(ex)}
                res_message.set_data(data)

            return Response(response=serializer.serialize(res_message), status=200)

        self.app = app

    def start(self):
        """
        Start the flask app with the configs defined in the initialization
        :raises `InvalidServerConfigurationException`: exception is raised
        app is not initialized
        """
        if self.app is not None:
            # from waitress import serve
            # serve(self.app, host=self.host, port=self.port)
            from flask import cli
            cli.show_server_banner = lambda *_: None
            # REMOVE the following message
            # * Environment: production
            #    WARNING: Do not use the development server in a production environment.
            #    Use a production WSGI server instead.
            self.app.run(host=self.host, port=self.port,
                         ssl_context=self.ssl_context)
        else:
            raise InvalidServerConfigurationException()

    def stop(self):
        """Stop the flask app and cleanup
        """
        requests.post("http://{}:{}".format(
            self.host, self.port) + "/shutdown")


class RestSender(FLSender):

    """
    Basic implementation using the request package of python to send
    requests to Rest endpoints.
    """

    def __init__(self, source_info, ssl_config=None):
        """
        Initializes the connection with settings object which is passed
        from Aggregator/party

        :param source_info: Dictionary with connection details such as ip,
            port, authentication details etc.
        :type source_info: `dict`
        """
        self.source_info = source_info
        self.ssl_config = ssl_config
        self.ssl_enabled = True if ssl_config else False

    def initialize(self):
        """Does all the setups required for the sender. This method should also
        check for availability of resources, open ports, certificates etc.,
        if required
        """
        logger.info('RestSender initialized')

    def _url(self, endpoint_url, name):
        """"method to append request endpoint to the URL as needed

        :param endpoint_url: Endpoint url constructed using destination info
        :type endpoint_url: `str`
        :param name: Name of the resource identifier which needs to be \
        requested
        :type name: `str`
        :return: updated url
        :rtype: `str`
        """""
        return "{}/{}".format(endpoint_url, name)

    def get(self, endpoint, name):
        """
        send a get request to the url with resource endpoint defined in
        arguments
        :param endpoint: Endpoint url constructed using destination info
        :type endpoint: `str`
        :param name: Resource identifier name
        :type name: `str`
        :return: response object
        :rtype: `response`
        """
        return self.get_params(name, None)

    def get_params(self, endpoint, name, params=None):
        """send a get requests to the url with resource endporint and param

        :param endpoint: Endpoint url constructed using destination info
        :type endpoint: `str`
        :param name: Resource identifier name
        :type name: `str`
        :param params: Contains sessionkey and format information which
        should be passed down to server for every gety request.
        :type params: `dict`
        :return: response object
        :rtype: `Response`
        """
        if params is not None:
            # TODO: construct params, add additional information and forward
            # it to server
            pass
        url = self._url(endpoint, name)
        return requests.get(url)

    def post(self, endpoint, name, data, headers, certs=None):
        """send a post request with data formatted as json. Currently
        it only supports json format

        :param endpoint: Endpoint url constructed using destination info
        :type endpoint: `str`
        :param name: Resource identifier name
        :type name: `str`
        :param data: Message object
        :type data: `str`
        :return: response object
        :rtype: `Response`
        """
        url = self._url(endpoint, name)
        return requests.post(url, data=data, headers=headers, verify=False, cert=certs)

    def get_url_from_info(self, info):
        """Constructs a url using the destination information

        :param info: Information about the destination to which message should be forwarded
        :type info: `dict`
        :return: url
        :rtype: `str`
        """
        if not info:
            raise InvalidConfigurationException(
                'Destination info is not valid or null.')

        if 'url' in info:
            return info.get('url')
        elif info and 'ip' in info and 'port' in info:
            if self.ssl_enabled:
                protocol = "https"
            else:
                protocol = "http"
            return "{}://{}:{}".format(protocol,
                                       info.get('ip'), info.get('port'))

        else:
            raise InvalidConfigurationException(
                'Destination info does not have valid host, port or url information')

    def get_headers(self, destination_info):
        """Build header dictionary based on the info 

        :param destination_info: Information about the destination to which message should be forwarded
        :type destination_info: `dict`
        :return: header dictionary
        :rtype: `dict`
        """
        headers = {'Content-type': 'application/json',
                   'Accept': 'application/json'}

        if 'headers' in destination_info:
            custom_headers = destination_info['headers']
            for key in custom_headers:
                headers[key] = custom_headers.get(key)

        return headers

    def get_certificates_mutual_auth(self):
        """When mutual authentication is required, client side certificates are used
        while sending requests.

        :return: cert tuple
        :rtype: `tuple`
        """
        if self.ssl_config and self.ssl_config.get('mutual_auth'):
            certs = (self.ssl_config.get('cert'), self.ssl_config.get('key'))
            return certs

        return None

    def send_message(self, destination, message):
        """
        used for sending all the requests. Message object should be
        validated and endpoint should be decided based on message codes.

        :param destination: Information about the destination to which message \
        should be forwarded
        :type destination: `dict`
        :param message: Message object constructed by aggregator/party
        :type message: `Message`
        :return: response object
        :rtype: `Response`
        """
        path = message.message_type
        endpoint = self.get_url_from_info(destination)

        message.add_sender_info(self.source_info)
        serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()

        message = serializer.serialize(message)

        headers = self.get_headers(destination)
        certs = self.get_certificates_mutual_auth()
        # send request to aggregator
        logger.debug('Sending serialized message')
        try:
            response = self.post(endpoint, path, message, headers, certs)
        except SSLError as error:
            logger.exception("Error occurred while performing ssl handshake")
            # logger.exception(error)
            raise FLException(
                "SSL Handshake error occured while sending request to url: "+ str(endpoint))

        logger.debug('Received serialized message as response')
        # deserialize response
        response_message = serializer.deserialize(response.content)

        return response_message

    def cleanup(self):
        """
        Cleanup the Sender object and close the hanging connections if any
        """
        logger.info('Cleanup HTTP Client')
