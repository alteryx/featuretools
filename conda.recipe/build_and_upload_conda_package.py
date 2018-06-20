import subprocess


def build_docker_img(img_id, no_cache=False):
    build_cmd = ["docker", "build"]
    if no_cache:
        build_cmd.append("--no-cache")

    build_cmd.extend(["-t", img_id, "."])

    # remove so we can catch an error more easily
    # in the case where the build command fails
    subprocess.run(['docker', 'rmi', '-f', img_id])

    subprocess.run(build_cmd)


if __name__ == '__main__':
    import argparse
    import getpass
    import sys
    import os

    parser = argparse.ArgumentParser(description='Test locally')
    parser.add_argument('--no-cache', default=False, action='store_true', help='build docker image without cache')
    parser.add_argument('--no-build', default=False, action='store_true', help='Do not build the docker image')
    parser.add_argument('--img', default="conda_featuretools_build", type=str, help='docker image to use')
    parser.add_argument('--username', type=str, help='Anaconda username')
    parser.add_argument('--password', help="Anaconda password. If this is not given, you will be prompted")
    args = parser.parse_args()
    password = args.password
    if not password:
        password = getpass.getpass(stream=sys.stderr)

    img_id = args.img

    if not args.no_build:
        build_docker_img(img_id, args.no_cache)

    featuretools_folder = os.path.dirname(os.getcwd())
    run_cmd = ["docker",
               "run",
               "-v", featuretools_folder + ":/featuretools/",
               '-i',
               '--entrypoint', '/bin/bash',
               img_id,
               "-c",
               "/featuretools/conda.recipe/build_featuretools.sh {} '{}'".format(args.username,
                                                                                 password)]
    with open('test.log', 'w') as f:
        process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            sys.stdout.buffer.write(line)
            f.write(line.decode(sys.stdout.encoding))
