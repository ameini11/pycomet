from absl import app
from absl import logging
from absl.logging import PythonFormatter

class CustomPythonFormatter(PythonFormatter):
    def format(self, record):
        return super(PythonFormatter, self).format(record)

def main(argv):

    logging.get_absl_handler().setFormatter(CustomPythonFormatter())

    logging.info('Info log')
    logging.warning('Warning log')


if __name__ == '__main__':
    app.run(main)