from __future__ import print_function

import sys
from builtins import input, str
from subprocess import call

from conf import release

LOCAL_ROOT = "build/html/"
REMOTE_ROOT = "s3://docs.featuretools.com"


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def upload(root=False):
    # build html
    if not query_yes_no("Upload Release: %s" % str(release)):
        print("Not uploading")
        return

    if root and not query_yes_no("Upload to root?"):
        print("Not uploading")
        return

    call(["make", "clean", "html"])

    remote_version = "%s/v%s" % (REMOTE_ROOT, str(release))

    # # upload to AWS S3
    call(["aws", "s3", "sync", LOCAL_ROOT, remote_version])
    if root:
        remote_root = "%s" % (REMOTE_ROOT)
        call(["aws", "s3", "sync", LOCAL_ROOT, remote_root])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Upload documentation to AWS S3')
    parser.add_argument('--root',
                        default=False,
                        action='store_true',
                        help='Specifies if docs should get upload to root directory')
    args = parser.parse_args()
    upload(root=args.root)
