# This Python file uses the following encoding: utf-8
"""
download_img.py
    Downloads all the images on the supplied URL, and saves them to the
    specified output file ("/home/alant/python/images" by default)

Usage:
    python download_img.py http://example.com/ [output]
"""

from BeautifulSoup import BeautifulSoup as bs
import urlparse
from urllib2 import urlopen
from urllib import urlretrieve
import os
import sys
import signal

# Определение класса для прерывания слишком долгих процессов. Взято отсюда:
# https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def main(url, out_folder="/home/alant/python/images"):
    """Downloads all the images at 'url' to ./images/"""
    # Задаём время, после которого нужно вернуть ошибку
    with timeout(seconds=60):
        soup = bs(urlopen(url))
    parsed = list(urlparse.urlparse(url))

    for image in soup.findAll("img"):
        print "Image: %(src)s" % image
        image_url = urlparse.urljoin(url, image['src'])
        filename = image["src"].split("/")[-1]
        outpath = os.path.join(out_folder, filename)
        urlretrieve(image_url, outpath)

def _usage():
    print "usage: python download_img.py http://example.com [outpath]"

if __name__ == "__main__":
    url = sys.argv[-1]
    out_folder = "/home/alant/python/images"
    if not url.lower().startswith("http"):
        out_folder = sys.argv[-1]
        url = sys.argv[-2]
        if not url.lower().startswith("http"):
            _usage()
            sys.exit(-1)
    main(url, out_folder)