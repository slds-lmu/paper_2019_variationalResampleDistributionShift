add-apt-repository ppa:jonathonf/python-3.7
apt-get update
apt-get install -y python3.7

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
# update-alternatives --config python3
