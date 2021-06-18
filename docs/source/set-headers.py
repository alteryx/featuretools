import urllib.request

opener = urllib.request.build_opener()
opener.addheaders = [("Testing", "True")]
urllib.request.install_opener(opener)
