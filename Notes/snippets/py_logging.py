import logging
def init_logging(log_file):
    fmt = '%(asctime)-15s %(filename)s:%(lineno)s %(levelname)s %(message)s'
    logging.basicConfig(filename=log_file,
                        level=logging.DEBUG,
                        format=fmt,
                        datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=fmt)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(stream_handler)


if __name__ == '__main__':
    log_file = '/var/log/{}_{}_{}.log'.format(
        os.path.basename(__file__).split('.')[0], tag,
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    init_logging(log_file)